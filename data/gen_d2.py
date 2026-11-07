"""Build dataset D2 from LinAlg-Bench: python data/gen_d2.py

Downloads the benchmark split at a pinned revision into cache/ and writes
data/d2.jsonl. Source, version, licence and the changes made are in data/SOURCES.md.

All 660 problems are kept. `runnable` marks those this system can check exactly:
determinant and rank, whose answers are exact. Every other task is outside the core
6 topics, or (eigenvalue) has irrational answers rounded to 4 decimals, which cannot
take an exact certificate. For rank problems the Solver also gives the RREF, because
that is what the certificate needs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import gen_d1  # noqa: E402
from verifier.exact import mat, to_json  # noqa: E402

REPO = "LinAlgBench/linalg-bench"
REVISION = "9c4c65ac347f5fe7fd602388c166956f46ad9f11"
SHARDS = ["train-00000.parquet", "train-00001.parquet", "train-00002.parquet"]
SEED = gen_d1.SEED
DEV_SHARE = 0.2
TOPIC_OF_TASK = {"determinant": "determinant", "rank": "rref_rank"}
_BMATRIX = re.compile(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", re.S)
_SIZE = re.compile(r"(\d)x\d")
_LAST_INT = re.compile(r"=\s*(-?\d+)\s*$")


def download(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard in SHARDS:
        path = folder / shard
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/benchmark/{shard}"
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        paths.append(path)
    return paths


def parse_matrix(latex: str) -> list:
    """The first bmatrix in a LaTeX string, as a list of integer rows."""
    m = _BMATRIX.search(latex)
    if m is None:
        raise ValueError("no bmatrix found")
    rows = [r for r in m.group(1).split("\\\\") if r.strip()]
    return [[int(cell.strip()) for cell in row.split("&")] for row in rows]


def _stated_integer(answer_latex: str) -> int:
    m = _LAST_INT.search(answer_latex.strip())
    if m is None:
        raise ValueError(f"no integer answer in {answer_latex!r}")
    return int(m.group(1))


def _convert(row, split: str | None) -> dict:
    task = row["Subcat"]
    rec = {
        "id": f"D2-{row['Problem_ID']}",
        "dataset": "d2",
        "source_id": row["Problem_ID"],
        "task": task,
        "topic": TOPIC_OF_TASK.get(task),
        "size": int(_SIZE.search(row["Problem_ID"]).group(1)),
        "runnable": task in TOPIC_OF_TASK,
        "problem_latex": row["problem_latex"],
        "answer_latex": row["answer_latex"],
        "split": split,
    }
    if not rec["runnable"]:
        return rec
    A = parse_matrix(row["problem_latex"])
    M = mat(A)
    stated = _stated_integer(row["answer_latex"])
    if task == "determinant":
        det = M.det(method="bareiss")
        if det != stated:
            raise ValueError(f"{row['Problem_ID']}: SymPy gives {det}, the dataset says {stated}")
        truth = {"determinant": det}
    else:
        R, pivots = M.rref()
        if len(pivots) != stated:
            raise ValueError(f"{row['Problem_ID']}: SymPy gives rank {len(pivots)}, the dataset says {stated}")
        truth = {"rref": R, "rank": len(pivots), "pivots": [p + 1 for p in pivots]}
    rec["A"] = A
    rec["problem"] = f"{gen_d1.QUESTIONS[rec['topic']]}\nA = {A}"
    rec["ground_truth"] = to_json(truth)
    return rec


def build(paths: list[Path]) -> list[dict]:
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values("Problem_ID")
    rows = [r for _, r in df.iterrows()]
    cells: dict = {}
    for row in rows:
        if row["Subcat"] in TOPIC_OF_TASK:
            cells.setdefault((row["Subcat"], _SIZE.search(row["Problem_ID"]).group(1)), []).append(row["Problem_ID"])
    splits = {}
    for (task, size), ids in sorted(cells.items()):
        order = list(ids)
        random.Random(f"{SEED}-d2-{task}-{size}").shuffle(order)
        n_dev = round(DEV_SHARE * len(order))
        splits.update({pid: "dev" if k < n_dev else "test" for k, pid in enumerate(order)})
    return [_convert(r, splits.get(r["Problem_ID"])) for r in rows]


def load(path: Path = ROOT / "data" / "d2.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=ROOT / "cache" / "linalg_bench" / REVISION)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "d2.jsonl")
    args = ap.parse_args()
    problems = build(download(args.source))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for p in problems:
            f.write(json.dumps(p, separators=(",", ":")) + "\n")
    runnable = sum(p["runnable"] for p in problems)
    print(f"wrote {len(problems)} problems to {args.out} ({runnable} runnable)")


if __name__ == "__main__":
    main()
