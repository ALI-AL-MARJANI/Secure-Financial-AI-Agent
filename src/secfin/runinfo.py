"""Metadata written next to every result file: config, seed, git SHA, hardware, library versions."""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

_LIBS = [
    "torch",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "rank-bm25",
    "langgraph",
    "langchain-core",
    "langchain-ollama",
    "ragas",
    "presidio-analyzer",
    "datasets",
    "numpy",
]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def hardware() -> dict:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
    }
    mem = _sysctl("hw.memsize")
    if mem:
        info["ram_gb"] = round(int(mem) / 2**30, 1)
    try:
        import torch

        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["mps"] = bool(torch.backends.mps.is_available())
    except ImportError:
        pass
    return info


def versions() -> dict:
    out = {}
    for lib in _LIBS:
        with contextlib.suppress(md.PackageNotFoundError):
            out[lib] = md.version(lib)
    return out


def run_metadata(config: dict | None = None, seed: int | None = None, **extra: Any) -> dict:
    return {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "seed": seed,
        "config": config,
        "hardware": hardware(),
        "versions": versions(),
        **extra,
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def write_env() -> Path:
    return write_json(RESULTS / "env.json", run_metadata())


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
