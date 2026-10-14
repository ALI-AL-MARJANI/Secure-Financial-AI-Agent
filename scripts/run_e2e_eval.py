"""End-to-end RAG evaluation with a local judge (RAGAS 0.4 collections API over Ollama's OpenAI endpoint).

  python scripts/run_e2e_eval.py --stage generate --generator mistral
  python scripts/run_e2e_eval.py --stage judge    --generator mistral
  python scripts/run_e2e_eval.py --stage summary

Stages are resumable (per-question JSONL caches). Judge failures (invalid structured output, timeouts) are
recorded as `null` with the error, never as 0, and reported as a failure rate per metric.
Outputs: results/e2e/answers_<gen>.jsonl, results/e2e/judged_<gen>__<judge>.jsonl, results/e2e/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import defaultdict

os.environ.setdefault("TQDM_DISABLE", "1")

from secfin.agent.rag import ANSWER_PROMPT, RegulationSearch, generate_answer, is_abstention  # noqa: E402
from secfin.eval.ir_metrics import bootstrap_ci, latency_summary  # noqa: E402
from secfin.guardrails.output_guard import check_output  # noqa: E402
from secfin.runinfo import REPO_ROOT, load_config, read_jsonl, run_metadata, write_json  # noqa: E402

OUT = REPO_ROOT / "results/e2e"


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_generate(cfg: dict, gen: str, limit: int) -> None:
    from secfin.llm import OllamaLLM

    golden = [g for g in read_jsonl(REPO_ROOT / "data/golden/golden.jsonl") if g["split"] == cfg["split"]]
    if limit:
        golden = golden[:limit]
    path = OUT / f"answers_{_slug(gen)}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    rcfg = load_config(REPO_ROOT / cfg["retrieval_config"])["retrieval"]
    search = RegulationSearch(rcfg)
    g = cfg["generation"]
    llm = OllamaLLM(gen, g["temperature"], seed=g["seed"], num_ctx=g["num_ctx"])
    for item in golden:
        if item["id"] in done:
            continue
        t0 = time.perf_counter()
        docs = search.search(item["question"])
        t1 = time.perf_counter()
        answer = generate_answer(llm, item["question"], docs)
        t2 = time.perf_counter()
        guard = check_output(answer, [d.text for d in docs])
        _append(
            path,
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": answer,
                "reference": item["reference_answer"],
                "gold_passages": item["gold_passages"],
                "contexts": [d.text for d in docs],
                "context_passages": [d.passage_id for d in docs],
                "citations": [d.citation for d in docs],
                "abstained": is_abstention(answer),
                "ungrounded_numbers": guard.ungrounded_numbers,
                "retrieval_s": t1 - t0,
                "generation_s": t2 - t1,
                "generator": gen,
            },
        )
        print(f"{item['id']} {'ABSTAIN' if is_abstention(answer) else 'ok'} {t2 - t0:.1f}s")


def _judge_clients(cfg: dict):
    from openai import AsyncOpenAI
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    j = cfg["judge"]
    client = AsyncOpenAI(base_url=j["base_url"], api_key="ollama")  # local Ollama, the key is ignored
    llm = llm_factory(
        j["model"], provider="openai", client=client, temperature=j["temperature"], max_tokens=j["max_tokens"]
    )
    emb = HuggingFaceEmbeddings(model=cfg["embeddings"])
    return llm, emb


async def _score(metric, **kwargs):
    try:
        res = await metric.ascore(**kwargs)
        v = float(res.value)
        return (None if math.isnan(v) else v), None
    except Exception as e:  # noqa: BLE001 - every judge failure is recorded, not hidden
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def stage_judge(cfg: dict, gen: str, limit: int) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    llm, emb = _judge_clients(cfg)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    src = OUT / f"answers_{_slug(gen)}.jsonl"
    dst = OUT / f"judged_{_slug(gen)}__{_slug(cfg['judge']['model'])}.jsonl"
    done = {r["id"] for r in read_jsonl(dst)} if dst.exists() else set()
    rows = [r for r in read_jsonl(src) if r["id"] not in done]
    if limit:
        rows = rows[:limit]

    async def run_one(r):
        out = {"id": r["id"], "scores": {}, "errors": {}}
        answerable = r["type"] != "unanswerable"
        jobs = {}
        if not r["abstained"]:
            jobs["faithfulness"] = dict(
                user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]
            )
            jobs["answer_relevancy"] = dict(user_input=r["question"], response=r["answer"])
        if answerable:
            jobs["context_precision"] = dict(
                user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"]
            )
            jobs["context_recall"] = dict(
                user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference"]
            )
        for name, kw in jobs.items():
            t = time.perf_counter()
            out["scores"][name], err = await _score(metrics[name], **kw)
            out.setdefault("judge_s", {})[name] = time.perf_counter() - t
            if err:
                out["errors"][name] = err
        return out

    async def main():
        for r in rows:
            res = await run_one(r)
            _append(dst, res)
            print(
                r["id"],
                {k: (None if v is None else round(v, 3)) for k, v in res["scores"].items()},
                "errors:" if res["errors"] else "",
                list(res["errors"]),
            )

    asyncio.run(main())


def stage_summary(cfg: dict) -> None:
    boot = cfg["bootstrap"]
    judge = cfg["judge"]["model"]
    summary = {}
    for gen in cfg["generators"]:
        apath = OUT / f"answers_{_slug(gen)}.jsonl"
        if not apath.exists():
            continue
        answers = {r["id"]: r for r in read_jsonl(apath)}
        jpath = OUT / f"judged_{_slug(gen)}__{_slug(judge)}.jsonl"
        judged = {r["id"]: r for r in read_jsonl(jpath)} if jpath.exists() else {}
        ans = [r for r in answers.values() if r["type"] != "unanswerable"]
        una = [r for r in answers.values() if r["type"] == "unanswerable"]
        s: dict = {
            "n_questions": len(answers),
            "self_judged": gen.split(":")[0] == judge.split(":")[0],
            "abstention_on_unanswerable": bootstrap_ci([float(r["abstained"]) for r in una], **boot),
            "false_abstention_on_answerable": bootstrap_ci([float(r["abstained"]) for r in ans], **boot),
            "ungrounded_number_rate": bootstrap_ci(
                [float(bool(r["ungrounded_numbers"])) for r in answers.values() if not r["abstained"]], **boot
            ),
            # Deterministic (non-LLM) context metric: share of gold passages present in the 5 contexts.
            "context_passage_recall@5": bootstrap_ci(
                [
                    len(set(r["gold_passages"]) & set(r["context_passages"])) / len(r["gold_passages"])
                    for r in ans
                ],
                **boot,
            ),
            "latency_generation": latency_summary([r["generation_s"] for r in answers.values()]),
            "latency_retrieval": latency_summary([r["retrieval_s"] for r in answers.values()]),
        }
        if judged:
            per_metric = defaultdict(list)
            failures = defaultdict(lambda: [0, 0])
            for j in judged.values():
                for m, v in j["scores"].items():
                    failures[m][1] += 1
                    if v is None:
                        failures[m][0] += 1
                    else:
                        per_metric[m].append(v)
            s["ragas"] = {m: bootstrap_ci(v, **boot) for m, v in per_metric.items()}
            s["judge_failure_rate"] = {
                m: {"failed": f, "attempted": n, "rate": f / n} for m, (f, n) in failures.items()
            }
            s["judged_n"] = len(judged)
        summary[gen] = s
    write_json(
        OUT / "summary.json",
        {
            "meta": run_metadata(config=cfg, seed=cfg["generation"]["seed"], answer_prompt=ANSWER_PROMPT),
            "generators": summary,
        },
    )
    print(
        json.dumps(
            {
                g: {k: v for k, v in s.items() if k in ("ragas", "abstention_on_unanswerable")}
                for g, s in summary.items()
            },
            indent=1,
        )[:3000]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2e.yaml")
    ap.add_argument("--stage", choices=["generate", "judge", "summary"], required=True)
    ap.add_argument("--generator", default="mistral")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    if args.stage == "generate":
        stage_generate(cfg, args.generator, args.limit)
    elif args.stage == "judge":
        stage_judge(cfg, args.generator, args.limit)
    else:
        stage_summary(cfg)


if __name__ == "__main__":
    main()
