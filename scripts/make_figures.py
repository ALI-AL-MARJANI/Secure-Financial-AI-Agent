"""Render figures (results/figures/*.png) and markdown tables (results/tables.md) from results/*.json.

Every number shown in the README tables comes from results/tables.md, which this script writes.
Rows whose result file does not exist yet are written as "TBD (run pending)".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from secfin.runinfo import REPO_ROOT, load_config  # noqa: E402

RES = REPO_ROOT / "results"
FIG = RES / "figures"
BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6a66", "#e4e3df"
TBD = "TBD (run pending)"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _row_metrics(row: dict) -> dict:
    """Aggregate over seeds (HyDE rows have several runs)."""
    runs = row["runs"]
    if len(runs) == 1:
        return runs[0]["metrics"]
    return {m: {"mean": v["mean"], "std": v["std"]} for m, v in row["seed_mean_std"].items()}


def fmt_ci(m: dict) -> str:
    if "std" in m:
        return f"{m['mean']:.3f} ± {m['std']:.3f} (3 seeds)"
    return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def retrieval_table(name: str, cfg: dict) -> tuple[list[str], dict | None]:
    data = _load(RES / f"retrieval/{name}.json")
    ds = name.split("-")[0]
    rows = {r["name"]: r for r in (data or {}).get("variants", [])}
    lines = [
        "| Variant | nDCG@10 [95% CI] | Recall@5 | Recall@20 | MRR@10 | p50 / p90 latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for v in cfg["variants"]:
        if ds not in v.get("datasets", ["ecfr", "fiqa"]):
            continue
        r = rows.get(v["name"])
        if r is None:
            lines.append(f"| {v['name']} | {TBD} | | | | |")
            continue
        m = _row_metrics(r)
        lat = r["runs"][0]["latency"]
        lines.append(
            f"| {v['name']} | {fmt_ci(m['ndcg@10'])} | {m['recall@5']['mean']:.3f} | "
            f"{m['recall@20']['mean']:.3f} | {m['mrr@10']['mean']:.3f} | "
            f"{lat['p50_ms']:.0f} / {lat['p90_ms']:.0f} |"
        )
    return lines, data


def retrieval_figure(name: str, data: dict, title: str) -> None:
    rows = data["variants"]
    labels = [r["name"] for r in rows]
    ms = [_row_metrics(r)["ndcg@10"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2), dpi=150)
    y = range(len(rows))[::-1]
    for yi, m in zip(y, ms, strict=True):
        lo = m.get("lo", m["mean"] - m.get("std", 0))
        hi = m.get("hi", m["mean"] + m.get("std", 0))
        ax.plot([lo, hi], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.scatter([m["mean"]], [yi], s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
        ax.text(hi + 0.008, yi, f"{m['mean']:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.set_xlabel("nDCG@10 (dot = mean, line = 95% bootstrap CI)", fontsize=9, color=MUTED)
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.02, ha="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / f"retrieval_{name}.png")
    plt.close(fig)


def injection(lines: list[str]) -> None:
    data = _load(RES / "guardrails/injection.json")
    lines += ["", "### Prompt-injection detection (test split)", ""]
    if not data:
        lines.append(TBD)
        return
    lines += [
        "| Detector | TPR (attacks) [95% CI] | FPR all benign | FPR Banking77 + golden [95% CI] | TPR without jackhhao |",
        "|---|---|---|---|---|",
    ]
    test, clean = data["results"]["test"], data["results"]["test_without_jackhhao"]
    pts = []
    for name, r in test.items():
        t, f = r["tpr"], r["fpr_banking"]
        lines.append(
            f"| {name} | {t['rate']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] | {r['fpr']['rate']:.4f} | "
            f"{f['rate']:.4f} [{f['lo']:.4f}, {f['hi']:.4f}] | {clean[name]['tpr']['rate']:.3f} |"
        )
        pts.append((name, f["rate"] * 100, t["rate"] * 100))
    lines += [
        "",
        "Per-source flag rates:",
        "",
        "| Detector | "
        + " | ".join(k.replace("|", " / ") for k in next(iter(test.values()))["flag_rate_by_source"])
        + " |",
        "|---" * (1 + len(next(iter(test.values()))["flag_rate_by_source"])) + "|",
    ]
    for name, r in test.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{v['rate']:.3f}" for v in r["flag_rate_by_source"].values()) + " |"
        )
    lat = data["latency_batch1"]
    lines += [
        "",
        "Latency per message (batch 1, "
        + data["meta"]["config"]["device"]
        + "): "
        + ", ".join(f"{k} p50 {v['p50_ms']:.2f} ms / p90 {v['p90_ms']:.2f} ms" for k, v in lat.items()),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for name, x, yv in pts:
        ax.scatter([x], [yv], s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        # the two 5%-FPR points nearly coincide: put the union label above, ProtectAI alone below
        offset = (-60, 10) if "OR" in name else (-40, -16) if name == "protectai@0.5" else (6, 4)
        ax.annotate(
            name.split(" (")[0], (x, yv), xytext=offset, textcoords="offset points", fontsize=8, color=INK
        )
    ax.set_xlabel("False positive rate on benign banking queries (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("True positive rate on public attacks (%)", fontsize=9, color=MUTED)
    ax.set_title(
        "Prompt-injection detectors: detection vs. blocked customers", fontsize=10.5, color=INK, loc="left"
    )
    ax.set_xlim(-0.3, max(p[1] for p in pts) * 1.35 + 0.5)
    ax.set_ylim(0, 100)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "injection_tradeoff.png")
    plt.close(fig)


def pii(lines: list[str]) -> None:
    data = _load(RES / "guardrails/pii.json")
    lines += ["", "### PII detection (entity-level, overlap match)", ""]
    if not data:
        lines.append(TBD)
        return
    r = data["results"]
    lines += [
        "| Detector | Synthetic P / R / F1 (600 msgs) | Stress P / R / F1 (30 hand-written) | Message-level FP on 200 negatives | p50 latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for name in ("regex_v0", "regex_v1_validated", "presidio"):
        a, s = r[name]["micro"], r[f"{name}|stress"]["micro"]
        ml = r[name]["message_level"]
        lines.append(
            f"| {name} | {a['precision']:.3f} / {a['recall']:.3f} / {a['f1']:.3f} | "
            f"{s['precision']:.3f} / {s['recall']:.3f} / {s['f1']:.3f} | {ml['fp']} | "
            f"{r[name]['latency_per_message']['p50_ms']:.2f} |"
        )


def router(lines: list[str]) -> None:
    data = _load(RES / "agent/router.json")
    lines += ["", "### Router (test split)", ""]
    lines += [
        "| Router | Accuracy [95% CI] | Coverage | Accuracy when answering | p50 latency (ms) |",
        "|---|---|---|---|---|",
    ]
    res = (data or {}).get("results", {})
    for name in ("keyword", "llm", "cascade"):
        r = res.get(f"{name}|test")
        if r is None:
            lines.append(f"| {name} | {TBD} | | | |")
            continue
        a = r["accuracy"]
        lines.append(
            f"| {name} | {a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | {r['coverage']:.2f} | "
            f"{r['accuracy_when_answered']:.3f} | {r['latency']['p50_ms']:.1f} |"
        )


def e2e(lines: list[str]) -> None:
    cfg = load_config(REPO_ROOT / "configs/e2e.yaml")
    data = _load(RES / "e2e/summary.json")
    agree = _load(RES / "e2e/judge_agreement.json")
    lines += ["", f"### End-to-end RAG (golden test split, judge = {cfg['judge']['model']})", ""]
    lines += [
        "| Generator | Faithfulness | Answer relevancy | Context precision | Context recall | "
        "Abstains on unanswerable | False abstention | Judge failure rate (faithfulness) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    gens = (data or {}).get("generators", {})
    for g in cfg["generators"]:
        s = gens.get(g)
        if not s or "ragas" not in s:
            lines.append(f"| {g} | {TBD} | | | | | | |")
            continue
        rg = s["ragas"]

        def c(m, rg=rg):
            return fmt_ci(rg[m]) if m in rg else "n/a"

        lines.append(
            f"| {g}{' (self-judged)' if s['self_judged'] else ''} | {c('faithfulness')} | "
            f"{c('answer_relevancy')} | {c('context_precision')} | {c('context_recall')} | "
            f"{s['abstention_on_unanswerable']['mean']:.2f} | {s['false_abstention_on_answerable']['mean']:.2f} | "
            f"{s['judge_failure_rate'].get('faithfulness', {}).get('rate', float('nan')):.2f} |"
        )
    lines.append("")
    lines.append(
        f"Judge vs. human (50 answers): weighted kappa = {agree['weighted_kappa']:.2f}, "
        f"Spearman = {agree['spearman']:.2f}"
        if agree
        else f"Judge vs. human agreement: {TBD}"
    )


def agent(lines: list[str]) -> None:
    lines += ["", "### Agent tool calling and red team (3 seeds, T=0.7)", ""]
    lines += [
        "| Model / prompt placement | Tool selection | Argument accuracy | Red-team cases defended | LLM attempted a forbidden call |",
        "|---|---|---|---|---|",
    ]
    files = sorted((RES / "agent").glob("tool_calling_*.json"))
    if not files:
        lines.append(f"| mistral / inline | {TBD} | | | |")
    for f in files:
        d = json.loads(f.read_text())
        s, m = d["summary"], d["meta"]["config"]
        lines.append(
            f"| {m['model']} / {m['tool_prompt']} | {fmt_ci(s['tool_selection_accuracy'])} | "
            f"{fmt_ci(s['argument_accuracy'])} | {fmt_ci(s['redteam_defended'])} | "
            f"{fmt_ci(s['llm_attempted_forbidden_call'])} |"
        )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO_ROOT / "configs/retrieval/ablation.yaml")
    lines = ["<!-- generated by scripts/make_figures.py; do not edit by hand -->", ""]
    for name, title in (
        ("ecfr-test", "eCFR golden set (test, 114 answerable questions)"),
        ("fiqa-test", "BEIR FiQA-2018 (test, 648 queries)"),
    ):
        tbl, data = retrieval_table(name, cfg)
        lines += [f"### Retrieval: {title}", "", *tbl, ""]
        if data:
            retrieval_figure(name, data, title)
    injection(lines)
    pii(lines)
    router(lines)
    e2e(lines)
    agent(lines)
    (RES / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
