"""The human audit of claim extraction (PREREGISTRATION.md s3).

  python tools/claim_audit.py sample --split dev
  python tools/claim_audit.py agree --split dev

`sample` draws 10% of the delivered claims, stratified by system and topic, and writes
the same rows to one sheet for each of two raters, in audit/. Each rater compares the
model's text with the extracted claim and fills in extraction_correct with y or n. Raters
work independently and never see each other's sheet.

`agree` reports percent agreement and Cohen's kappa once both sheets are filled.

The decision of 2026-11-07 requires this on dev outputs before FROZEN exists, and again
on 10% of test outputs after the test run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score  # noqa: E402
from data import gen_d1, gen_d2  # noqa: E402
from extraction import extract  # noqa: E402
from run import RESULTS  # noqa: E402

AUDIT_DIR = ROOT / "audit"
SHARE = 0.10
SEED = 20261024
COLUMNS = ["audit_id", "model", "system", "topic", "problem_id", "where",
           "claim_text", "extracted_claim", "extraction_correct", "notes"]
RATERS = (1, 2)


def _source_text(item: dict, trace, answer, hints) -> str:
    """What the model actually wrote, for the rater to compare with the extraction."""
    if item["where"] == "step":
        return json.dumps(trace[item["index"]], ensure_ascii=False)
    if item["where"] == "final":
        return json.dumps(answer, ensure_ascii=False)
    return str(hints[item["index"]].get("text", ""))


def _printable(claim: dict) -> str:
    return json.dumps({k: v for k, v in claim.items() if k != "previous"}, ensure_ascii=False)


def all_claims(split: str, results: Path = RESULTS) -> list:
    problems = {p["id"]: p for p in gen_d1.load()}
    problems.update({p["id"]: p for p in gen_d2.load() if p["runnable"]})
    rows = []
    for (system, dataset, tag), frame in score.load_results(split, results).items():
        if dataset == "d3":
            continue
        for _, row in frame[frame["status"] == "answered"].iterrows():
            problem = problems[row["problem_id"]]
            trace, answer = json.loads(row["trace_json"]), json.loads(row["answer_json"])
            hints = score.delivered_hints(row)
            for item in extract(problem, trace, answer, hints):
                rows.append({"model": tag, "system": system, "topic": problem["topic"],
                             "problem_id": f"{row['problem_id']} run{row['run']}", "where": f"{item['where']} {item['index']}",
                             "claim_text": _source_text(item, trace, answer, hints),
                             "extracted_claim": _printable(item["claim"])})
    return rows


def draw(claims: list, share: float = SHARE, seed: int = SEED) -> list:
    """share of the claims in each (system, topic) stratum, rounded up, so no stratum is skipped."""
    strata: dict = {}
    for claim in claims:
        strata.setdefault((claim["system"], claim["topic"]), []).append(claim)
    chosen = []
    for (system, topic), group in sorted(strata.items()):
        rng = random.Random(f"{seed}-{system}-{topic}")
        chosen.extend(rng.sample(group, math.ceil(share * len(group))))
    return chosen


def sheet_path(split: str, rater: int, folder: Path = AUDIT_DIR) -> Path:
    return folder / f"claims_{split}_rater{rater}.csv"


def write_sheets(chosen: list, split: str, folder: Path = AUDIT_DIR) -> None:
    rows = [{**c, "audit_id": i, "extraction_correct": "", "notes": ""} for i, c in enumerate(chosen, 1)]
    folder.mkdir(parents=True, exist_ok=True)
    for rater in RATERS:
        path = sheet_path(split, rater, folder)
        if path.exists():
            sys.exit(f"{path.name} already exists. It may hold a rater's work, so it was not overwritten.")
    for rater in RATERS:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(sheet_path(split, rater, folder), index=False)


def _ratings(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    values = frame.set_index("audit_id")["extraction_correct"].str.strip().str.lower()
    unfinished = values[~values.isin(["y", "n"])]
    if len(unfinished):
        sys.exit(f"{path.name}: {len(unfinished)} rows are not y or n, for example audit_id {unfinished.index[0]}")
    return values


def agreement(split: str, folder: Path = AUDIT_DIR) -> dict:
    one, two = (_ratings(sheet_path(split, r, folder)) for r in RATERS)
    if set(one.index) != set(two.index):
        sys.exit("the two sheets do not hold the same audit_id values")
    two = two.loc[one.index]
    a, b = one.tolist(), two.tolist()
    agreed = sum(x == y for x, y in zip(a, b))
    both_correct = sum(x == y == "y" for x, y in zip(a, b))
    return {"claims": len(a), "agreed": agreed, "percent_agreement": agreed / len(a),
            "cohen_kappa": score.cohen_kappa(a, b), "both_say_correct": both_correct,
            "both_say_correct_share": both_correct / len(a),
            "disagreements": [i for i, x, y in zip(one.index, a, b) if x != y]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit claim extraction with two human raters.")
    ap.add_argument("command", choices=["sample", "agree"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    if args.command == "sample":
        claims = all_claims(args.split)
        if not claims:
            sys.exit(f"no delivered claims on the {args.split} split yet")
        chosen = draw(claims)
        write_sheets(chosen, args.split)
        print(f"sampled {len(chosen)} of {len(claims)} claims into {AUDIT_DIR}")
    else:
        result = agreement(args.split)
        kappa = result["cohen_kappa"]
        print(f"claims: {result['claims']}")
        print(f"percent agreement: {result['percent_agreement']:.3f}")
        print("cohen kappa: " + ("undefined (one category only)" if kappa is None else f"{kappa:.3f}"))
        print(f"both raters say correct: {result['both_say_correct_share']:.3f}")
        print(f"disagreements, audit_id: {result['disagreements']}")


if __name__ == "__main__":
    main()
