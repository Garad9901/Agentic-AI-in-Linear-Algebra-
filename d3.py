"""Loads D3, the real student solutions, from data/private/d3.csv (git-ignored).

data/d3_template.csv holds the columns, and docs/ANNOTATION_GUIDE.md says how to
fill them. One row per step per annotator:
  annotator   the two independent annotators, under any two names. Rows named
              "gold" are the adjudicated labels. Without them, gold is taken only
              from solutions where the two annotators agree, and the rest are
              left out and counted.
  problem     must contain the matrix, e.g. A = [[1,2],[3,4]] (and b = [5,6])
  step_value  a matrix, vector or value in JSON, or empty
  step_text   states the row operation, e.g. "R2 <- R2 - 3R1"

The split is by student, 30% dev / 70% test (PREREGISTRATION.md s4.3), taking the part
of solution_id before the first "-" as the student, so "S017-P2" belongs to S017.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
D3_PATH = ROOT / "data" / "private" / "d3.csv"
TEMPLATE_PATH = ROOT / "data" / "d3_template.csv"
COLUMNS = ["solution_id", "topic", "problem", "step_no", "step_text", "step_value", "annotator", "step_label",
           "first_error_step", "misconception"]
GOLD = "gold"
SEED = 20261024
_A = re.compile(r"\bA\s*=\s*(\[\s*\[.*?\]\s*\])", re.S)
_B = re.compile(r"\bb\s*=\s*(\[.*?\])", re.S)


class D3Unavailable(RuntimeError):
    pass


TAXONOMY_PATH = ROOT / "data" / "taxonomy.json"
TAXONOMY_TAG = "taxonomy-v1"
LABEL_COLUMNS = ["step_label", "first_error_step", "misconception"]


def taxonomy_v1_ready() -> bool:
    """True once data/taxonomy.json says frozen and matches the file at tag taxonomy-v1."""
    try:
        current = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        tagged = subprocess.run(["git", "show", f"{TAXONOMY_TAG}:data/taxonomy.json"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError, ValueError):
        return False
    return current.get("frozen") is True and json.loads(tagged) == current


def check_annotation_gate(df: pd.DataFrame) -> None:
    """Test-split labels made before the taxonomy was frozen break PREREGISTRATION.md s4.4."""
    labelled = df[LABEL_COLUMNS].apply(lambda c: c.str.strip() != "").any(axis=1)
    on_test = df["solution_id"].map(split_of) == "test"
    if (labelled & on_test).any() and not taxonomy_v1_ready():
        raise D3Unavailable("the D3 test split has labels, but the taxonomy is not frozen and tagged "
                            f"{TAXONOMY_TAG}. Test-split annotation must wait for the frozen taxonomy")


def read_rows(path: Path | None = None) -> pd.DataFrame:
    path = path or D3_PATH
    if not path.exists():
        raise D3Unavailable(f"{path.name} does not exist: no student data has been collected yet")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise D3Unavailable(f"{path.name} is missing columns {missing}")
    if df.empty:
        raise D3Unavailable(f"{path.name} has no rows")
    df["step_no"] = df["step_no"].astype(int)
    check_annotation_gate(df)
    return df


def split_of(solution_id: str) -> str:
    student = solution_id.split("-")[0]
    digest = hashlib.sha256(f"{SEED}-{student}".encode()).hexdigest()
    return "dev" if int(digest, 16) % 10 < 3 else "test"


def _step_check(value: str):
    return json.loads(value) if value.strip() else None


def load_problems(path: Path | None = None) -> list[dict]:
    """One record per solution, in the same shape as a D1 problem plus student_steps."""
    df = read_rows(path)
    problems = []
    for solution_id, rows in df.groupby("solution_id", sort=True):
        first = rows.iloc[0]
        a = _A.search(first["problem"])
        if a is None:
            raise D3Unavailable(f"{solution_id}: the problem column needs a matrix written as A = [[...]]")
        b = _B.search(first["problem"])
        steps = []
        for _, row in rows.drop_duplicates("step_no").sort_values("step_no").iterrows():
            steps.append({"step_no": int(row["step_no"]), "step_text": row["step_text"],
                          "step_value": row["step_value"], "check": _step_check(row["step_value"])})
        A = json.loads(a.group(1))
        problems.append({"id": solution_id, "dataset": "d3", "topic": first["topic"], "size": len(A[0]),
                         "problem": first["problem"], "A": A, "b": json.loads(b.group(1)) if b else None,
                         "student_steps": steps, "split": split_of(solution_id)})
    return problems


def _first_error(rows: pd.DataFrame):
    value = rows.iloc[0]["first_error_step"].strip()
    return int(value) if value else None


def gold_labels(df: pd.DataFrame) -> tuple[dict, int]:
    """Returns ({solution_id: {first_error_step, misconception, step_labels}}, n_left_out)."""
    gold, left_out = {}, 0
    for solution_id, rows in df.groupby("solution_id", sort=True):
        chosen = rows[rows["annotator"] == GOLD]
        if chosen.empty:
            names = sorted(rows["annotator"].unique())
            if len(names) != 2:
                left_out += 1
                continue
            one, two = (rows[rows["annotator"] == name] for name in names)
            same_step = _first_error(one) == _first_error(two)
            same_label = one.iloc[0]["misconception"] == two.iloc[0]["misconception"]
            if not (same_step and same_label):
                left_out += 1
                continue
            chosen = one
        gold[solution_id] = {
            "first_error_step": _first_error(chosen),
            "misconception": chosen.iloc[0]["misconception"].strip() or None,
            "step_labels": dict(zip(chosen["step_no"], chosen["step_label"].str.strip())),
        }
    return gold, left_out
