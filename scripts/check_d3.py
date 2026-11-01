"""Validate D3 files and assign the dev/test split by student (PREREGISTRATION.md s4.3).

  python scripts/check_d3.py data/private/d3            # validate and report
  python scripts/check_d3.py data/private/d3 --write    # also write the split into solutions.jsonl

Split: 30% dev / 70% test of students, seed 20261024, every solution of a student on one side.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latutor.generator import load  # noqa: E402
from latutor.verifier import Verifier  # noqa: E402

SEED = 20261024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d = Path(a.folder)
    probs = {p["id"]: p for p in load(d / "problems.jsonl")}
    sols = load(d / "solutions.jsonl")
    errors = []
    for s in sols:
        for k in ("solution_id", "student_id", "problem_id", "steps", "gold"):
            if k not in s:
                errors.append(f"{s.get('solution_id')}: missing {k}")
        if s.get("problem_id") not in probs:
            errors.append(f"{s.get('solution_id')}: unknown problem {s.get('problem_id')}")
            continue
        v = Verifier(probs[s["problem_id"]])
        for i, st in enumerate(s.get("steps", []), 1):
            c = st.get("claim")
            if c is not None and v.check(c).reason.startswith("malformed"):
                errors.append(f"{s['solution_id']} step {i}: malformed claim ({v.check(c).reason})")
        g = s.get("gold", {}).get("first_error_step")
        if g is not None and not (1 <= int(g) <= len(s.get("steps", []))):
            errors.append(f"{s['solution_id']}: gold first_error_step {g} out of range")
    students = sorted({s["student_id"] for s in sols})
    rng = random.Random(SEED)
    shuffled = students[:]
    rng.shuffle(shuffled)
    n_dev = round(0.3 * len(students))
    dev = set(shuffled[:n_dev])
    for s in sols:
        s["split"] = "dev" if s["student_id"] in dev else "test"
    usable = [s for s in sols if not s.get("excluded")]
    test_err = sum(1 for s in usable if s["split"] == "test" and s["gold"].get("first_error_step") is not None)
    print(f"students {len(students)}  solutions {len(sols)}  usable {len(usable)}  "
          f"excluded {len(sols) - len(usable)} {dict(Counter(s.get('exclusion_reason') for s in sols if s.get('excluded')))}")
    print(f"split: dev students {len(dev)}, test students {len(students) - len(dev)}")
    print(f"erroneous solutions in TEST split: {test_err} (plan target >= 100; below that RQ2 is reported as underpowered)")
    if errors:
        print(f"{len(errors)} problems:")
        for e in errors[:50]:
            print("  -", e)
    if a.write and not errors:
        with open(d / "solutions.jsonl", "w", encoding="utf-8") as f:
            for s in sols:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
        print("split written")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
