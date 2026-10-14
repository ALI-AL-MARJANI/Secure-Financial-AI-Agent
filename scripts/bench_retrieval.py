"""Retrieval ablation on the eCFR golden set or BEIR FiQA-2018.

  python scripts/bench_retrieval.py --dataset fiqa
  python scripts/bench_retrieval.py --dataset ecfr --split test
  python scripts/bench_retrieval.py --dataset ecfr --split test --with-llm      # needs Ollama + contextual cache

Writes results/retrieval/<dataset>-<split>.json (aggregates, CIs, paired tests, latency) and
results/retrieval/<dataset>-<split>_per_query.csv. Rows that need an LLM are merged into an existing file,
so the cheap rows and the LLM rows can be run at different times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np

from secfin.eval.ir_metrics import (
    bootstrap_ci,
    dedupe,
    graded_ndcg_at_k,
    hit_at_k,
    latency_summary,
    mrr_at_k,
    paired_bootstrap,
    recall_at_k,
)
from secfin.retrieval.components import BM25Index, CrossEncoderReranker, DenseIndex
from secfin.retrieval.datasets import IRDataset, load_ecfr, load_fiqa
from secfin.retrieval.llm_components import PROMPTS
from secfin.retrieval.pipeline import RetrievalPipeline
from secfin.runinfo import REPO_ROOT, load_config, run_metadata, write_json

CACHE = REPO_ROOT / "data/cache"
METRICS = ["recall@5", "recall@10", "recall@20", "hit@5", "mrr@10", "ndcg@10"]


def _device(pref: str) -> str:
    import torch

    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def query_metrics(ranking_units: list[str], gains: dict[str, float]) -> dict[str, float]:
    rel = set(gains)
    return {
        "recall@5": recall_at_k(ranking_units, rel, 5),
        "recall@10": recall_at_k(ranking_units, rel, 10),
        "recall@20": recall_at_k(ranking_units, rel, 20),
        "hit@5": hit_at_k(ranking_units, rel, 5),
        "mrr@10": mrr_at_k(ranking_units, rel, 10),
        "ndcg@10": graded_ndcg_at_k(ranking_units, gains, 10),
    }


class IndexCache:
    """One BM25 + dense index per indexed-text variant (prefix mode), embeddings cached on disk."""

    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        self._bm25: dict[str, BM25Index] = {}
        self._dense: dict[str, DenseIndex] = {}

    def _key(self, ds: IRDataset) -> str:
        h = hashlib.sha1()
        for d in sorted(ds.doc_texts):
            h.update(d.encode())
            h.update(ds.doc_texts[d].encode())
        return h.hexdigest()[:16]

    def bm25(self, ds: IRDataset) -> BM25Index:
        key = self._key(ds)
        if key not in self._bm25:
            t = time.perf_counter()
            ids = list(ds.doc_texts)
            self._bm25[key] = BM25Index(ids, [ds.doc_texts[i] for i in ids])
            print(f"  BM25 index over {len(ids)} docs in {time.perf_counter() - t:.1f}s")
        return self._bm25[key]

    def dense(self, ds: IRDataset) -> DenseIndex:
        key = self._key(ds)
        if key not in self._dense:
            ids = list(ds.doc_texts)
            model = self.cfg["dense_model"]
            path = CACHE / "emb" / f"{ds.name.split('-')[0]}_{model.replace('/', '_')}_{key}.npy"
            emb = np.load(path) if path.exists() else None
            t = time.perf_counter()
            idx = DenseIndex(
                model,
                ids,
                [ds.doc_texts[i] for i in ids],
                query_prefix=self.cfg["dense_query_prefix"],
                device=self.device,
                embeddings=emb,
            )
            if emb is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, idx.embeddings)
                print(f"  encoded {len(ids)} docs in {time.perf_counter() - t:.1f}s -> {path.name}")
            self._dense[key] = idx
        return self._dense[key]


def run_variant(ds: IRDataset, pipe: RetrievalPipeline, warmup: int) -> tuple[dict, list[float], dict]:
    per_query, lat, extra = {}, [], {"grades": {}, "abstained": 0}
    for i, (qid, q) in enumerate(ds.queries.items()):
        res = pipe.search(q)
        if i >= warmup:
            lat.append(res.timings["total_s"])
        units = dedupe([ds.doc_to_unit[d] for d in res.ranking])
        per_query[qid] = query_metrics(units, ds.qrels[qid])
        if res.grade is not None:
            extra["grades"][qid] = res.grade
            extra["abstained"] += int(not res.context_ids)
    return per_query, lat, extra


def aggregate(per_query: dict, boot: dict) -> dict:
    return {m: bootstrap_ci([v[m] for v in per_query.values()], **boot) for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval/ablation.yaml")
    ap.add_argument("--dataset", choices=["ecfr", "fiqa"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", default="", help="comma-separated subset of variant names")
    ap.add_argument("--with-llm", action="store_true", help="also run rows marked requires_llm")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N queries")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    device = _device(cfg.get("device", "cpu"))
    wanted = set(filter(None, args.variants.split(",")))
    variants = []
    for v in cfg["variants"]:
        v = {**cfg["defaults"], **v}
        if wanted and v["name"] not in wanted:
            continue
        if args.dataset not in v.get("datasets", ["ecfr", "fiqa"]):
            continue
        if v.get("requires_llm") and not args.with_llm:
            print(f"skip {v['name']} (requires --with-llm)")
            continue
        variants.append(v)

    out_dir = REPO_ROOT / "results/retrieval"
    stem = f"{args.dataset}-{args.split}"
    out_json, out_csv = out_dir / f"{stem}.json", out_dir / f"{stem}_per_query.csv"
    previous = json.loads(out_json.read_text()) if out_json.exists() else {}
    rows = {r["name"]: r for r in previous.get("variants", [])}
    per_query_all: dict[str, dict] = {}
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                per_query_all.setdefault(f"{r['variant']}|{r['seed']}", {})[r["qid"]] = {
                    m: float(r[m]) for m in METRICS
                }

    datasets: dict[str, IRDataset] = {}

    def get_ds(prefix: str) -> IRDataset:
        if prefix not in datasets:
            if args.dataset == "fiqa":
                datasets[prefix] = load_fiqa(split=args.split)
            else:
                datasets[prefix] = load_ecfr(
                    args.split, prefix, cfg["chunk_max_chars"], REPO_ROOT / cfg["contexts_path"]
                )
            if args.limit:
                ds = datasets[prefix]
                ds.queries = dict(list(ds.queries.items())[: args.limit])
        return datasets[prefix]

    cache = IndexCache(cfg, device)
    reranker = None
    for v in variants:
        ds = get_ds(v["prefix"])
        print(f"[{v['name']}] {len(ds.queries)} queries, {len(ds.doc_texts)} docs, device={device}")
        if v["rerank"] and reranker is None:
            reranker = CrossEncoderReranker(cfg["reranker"], device=device)
        seeds = cfg["hyde_llm"]["seeds"] if v["hyde"] else [None]
        seed_rows = []
        for seed in seeds:
            llm = None
            if v["hyde"] or v["crag"]:
                from secfin.llm import OllamaLLM

                llm_cfg = cfg["hyde_llm"] if v["hyde"] else cfg["llm"]
                llm = OllamaLLM(llm_cfg["model"], llm_cfg["temperature"], seed=seed)
            pipe = RetrievalPipeline(
                v,
                ds.doc_texts,
                bm25=cache.bm25(ds) if v["sparse"] else None,
                dense=cache.dense(ds) if v["dense"] else None,
                reranker=reranker if v["rerank"] else None,
                llm=llm,
            )
            t = time.perf_counter()
            per_query, lat, extra = run_variant(ds, pipe, cfg["latency"]["warmup_queries"])
            wall = time.perf_counter() - t
            per_query_all[f"{v['name']}|{seed}"] = per_query
            seed_rows.append(
                {
                    "seed": seed,
                    "metrics": aggregate(per_query, cfg["bootstrap"]),
                    "latency": latency_summary(lat),
                    "wall_s": wall,
                    **extra,
                }
            )
            print(
                "   ",
                {m: round(seed_rows[-1]["metrics"][m]["mean"], 4) for m in METRICS},
                f"p50={seed_rows[-1]['latency']['p50_ms']:.1f}ms",
            )
        row = {"name": v["name"], "config": v, "runs": seed_rows}
        if len(seed_rows) > 1:
            row["seed_mean_std"] = {
                m: {
                    "mean": float(np.mean([r["metrics"][m]["mean"] for r in seed_rows])),
                    "std": float(np.std([r["metrics"][m]["mean"] for r in seed_rows], ddof=1)),
                }
                for m in METRICS
            }
        rows[v["name"]] = row

    # Paired bootstrap on per-query nDCG@10 / Recall@5 against fixed references and the previous row.
    order = [v["name"] for v in cfg["variants"] if v["name"] in rows]

    def mean_over_seeds(name: str) -> dict[str, dict]:
        keys = [k for k in per_query_all if k.split("|")[0] == name]
        qids = per_query_all[keys[0]].keys()
        return {q: {m: float(np.mean([per_query_all[k][q][m] for k in keys])) for m in METRICS} for q in qids}

    comparisons = []
    for i, name in enumerate(order):
        refs = {"bm25", "dense_bge_small", "hybrid_rrf+rerank", "dense+title_prefix"} | (
            {order[i - 1]} if i else set()
        )
        for ref in sorted(refs & set(order)):
            if ref == name:
                continue
            a, b = mean_over_seeds(ref), mean_over_seeds(name)
            qids = sorted(set(a) & set(b))
            for m in ("ndcg@10", "recall@5"):
                comparisons.append(
                    {
                        "variant": name,
                        "reference": ref,
                        "metric": m,
                        **paired_bootstrap([a[q][m] for q in qids], [b[q][m] for q in qids]),
                    }
                )

    meta = run_metadata(
        config=cfg,
        seed=None,
        dataset=args.dataset,
        split=args.split,
        device=device,
        prompts=PROMPTS,
        n_queries=len(next(iter(datasets.values())).queries) if datasets else None,
    )
    write_json(out_json, {"meta": meta, "variants": [rows[n] for n in order], "comparisons": comparisons})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "qid", *METRICS])
        for key, pq in sorted(per_query_all.items()):
            name, seed = key.split("|")
            for qid, ms in pq.items():
                w.writerow([name, seed, qid, *[f"{ms[m]:.6f}" for m in METRICS]])
    print(f"wrote {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
