"""Hand out blank D3 annotation sheets: python tools/annotation_sheet.py --split dev --annotator NAME

Reads the anonymised transcriptions in data/private/d3.csv (rows with no annotator) and
writes data/private/annotate_<split>_<annotator>.csv with the label columns empty.

The test split is refused until the taxonomy is frozen: data/taxonomy.json has
"frozen": true and matches the git tag taxonomy-v1 (PREREGISTRATION.md s4.4). d3.py
also refuses to load test-split labels made before that, so a sheet made by hand is
caught too.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import d3  # noqa: E402

LABELS = ["step_label", "first_error_step", "misconception"]


def make_sheet(split: str, annotator: str, source: Path | None = None) -> Path:
    if split == "test" and not d3.taxonomy_v1_ready():
        sys.exit("refused: the D3 test split cannot be annotated until data/taxonomy.json is frozen and tagged "
                 "taxonomy-v1. Build the taxonomy on the dev split first.")
    rows = pd.read_csv(source or d3.D3_PATH, dtype=str, keep_default_na=False)
    transcribed = rows[rows["annotator"].str.strip() == ""].copy()
    transcribed = transcribed[transcribed["solution_id"].map(d3.split_of) == split]
    if transcribed.empty:
        sys.exit(f"no transcribed {split} solutions to annotate")
    transcribed["annotator"] = annotator
    for column in LABELS:
        transcribed[column] = ""
    out = d3.D3_PATH.parent / f"annotate_{split}_{annotator}.csv"
    if out.exists():
        sys.exit(f"{out.name} already exists and may hold work, so it was not overwritten")
    transcribed.to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    print(f"wrote {make_sheet(args.split, args.annotator)}")


if __name__ == "__main__":
    main()
