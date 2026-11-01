"""Build a SYNTHETIC stand-in for D3, used only to dry-run the D3 pipeline before real data exists.

It is not student data and must never be reported as such. Output goes to results/dryrun/d3_synthetic/.
Each "solution" is a row-reduction chain built from a D1 dev problem (rref, inverse, lsq, det topics),
with one arithmetic slip injected at a random step in about 70% of them (gold = that step).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latutor.exact import mat, to_json, vec  # noqa: E402
from latutor.generator import load  # noqa: E402

SEED = 20261024


def chain(M, rng, k):
    """k elimination-style row operations; returns list of (text, matrix)."""
    out, M = [], M.copy()
    for _ in range(k):
        i, j = rng.sample(range(M.rows), 2)
        c = rng.choice([-3, -2, -1, 1, 2, 3])
        M = M.copy()
        M[i, :] = M[i, :] + c * M[j, :]
        out.append((f"R{i + 1} <- R{i + 1} + ({c})R{j + 1}", M))
    return out


def main():
    rng = random.Random(SEED)
    probs = [p for p in load(ROOT / "data/d1/problems.jsonl") if p["split"] == "dev"
             and p["topic"] in ("rref", "inverse", "lsq", "det")]
    out_dir = ROOT / "results/dryrun/d3_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    sols = []
    for si in range(20):
        for p in rng.sample(probs, 4):
            A = mat(p["A"])
            if p["topic"] == "inverse":
                start, ctx = A.row_join(sp.eye(A.rows)), "augmented_A_I"
            elif p["topic"] == "lsq":
                start, ctx = (A.T * A).row_join(A.T * vec(p["b"])), "normal_augmented"
            else:
                start, ctx = A, "A"
            steps = chain(start, rng, rng.randint(3, 5))
            gold = None
            if rng.random() < 0.7:
                k = rng.randrange(len(steps))
                for _ in range(20):
                    i, j = rng.randrange(start.rows), rng.randrange(start.cols)
                    bad = steps[k][1].copy(); bad[i, j] += rng.choice([-1, 1])
                    if bad.col_join(start).rank() != start.rank() or bad.rank() != start.rank():
                        break
                else:
                    bad = None
                if bad is not None:
                    gold = k + 1
                    # propagate the slip: later steps apply the same operations to the wrong matrix
                    cur = bad
                    new = steps[:k] + [(steps[k][0], bad)]
                    for text, _m in steps[k + 1:]:
                        a, b = [int(x) for x in __import__("re").findall(r"R(\d+)", text)[:2]]
                        c = int(__import__("re").search(r"\((-?\d+)\)", text).group(1))
                        cur = cur.copy(); cur[a - 1, :] = cur[a - 1, :] + c * cur[b - 1, :]
                        new.append((text, cur))
                    steps = new
            sols.append({"solution_id": f"SYN{si:02d}-{p['id']}", "student_id": f"SYN{si:02d}",
                         "problem_id": p["id"], "synthetic": True,
                         "steps": [{"text": t, "claim": {"type": "intermediate_matrix", "context": ctx,
                                                         "matrix": to_json(m)}} for t, m in steps],
                         "gold": {"first_error_step": gold, "misconception": "ROW_OP_ARITHMETIC" if gold else None}})
    students = sorted({s["student_id"] for s in sols})
    rng.shuffle(students)
    dev = set(students[: round(0.3 * len(students))])
    for s in sols:
        s["split"] = "dev" if s["student_id"] in dev else "test"
    with open(out_dir / "solutions.jsonl", "w") as f:
        for s in sols:
            f.write(json.dumps(s) + "\n")
    with open(out_dir / "problems.jsonl", "w") as f:
        for p in probs:
            f.write(json.dumps(p) + "\n")
    print(f"{len(sols)} synthetic solutions -> {out_dir}")


if __name__ == "__main__":
    main()
