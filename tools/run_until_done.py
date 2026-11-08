"""Run systems until every problem has a final row, surviving crashes, sleep and daily caps.

  python tools/run_until_done.py --provider cerebras --dataset d1 --split dev --systems B1,B2,B3,OURS

Each attempt calls run.py with --wait-daily, so a daily cap pauses the run until it resets.
If run.py stops for any other reason (a crash, the machine sleeping, the network dropping,
or provider refusals), the next attempt resumes from the cache, so finished calls cost
nothing. A system is done when no row is a provider refusal (status api_error). Truncated
rows are final: they are logged and set aside, never scored and never retried here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run import model_for, result_path  # noqa: E402

PAUSE_SECONDS = 300
MAX_STALLED_ATTEMPTS = 5


def refused_rows(path: Path, expected: int) -> int:
    """Rows still to do: provider refusals plus rows never written."""
    if not path.exists():
        return expected
    frame = pd.read_csv(path, keep_default_na=False)
    return int((frame["status"] == "api_error").sum()) + max(0, expected - len(frame))


def run_system(system: str, args, expected: int, launch=subprocess.run, sleep=time.sleep) -> bool:
    path = result_path(system, args.dataset, args.split, args.run, args.provider, model_for(args.provider))
    command = [sys.executable, str(ROOT / "run.py"), "--provider", args.provider, "--system", system,
               "--dataset", args.dataset, "--split", args.split, "--run", str(args.run),
               "--workers", str(args.workers), "--wait-daily"]
    if args.limit:
        command += ["--limit", str(args.limit)]
    remaining, stalled = refused_rows(path, expected), 0
    while remaining:
        launch(command, cwd=ROOT)
        now = refused_rows(path, expected)
        stalled = stalled + 1 if now >= remaining else 0
        remaining = now
        if not remaining:
            break
        if stalled >= MAX_STALLED_ATTEMPTS:
            print(f"{system}: {remaining} rows still refused after {stalled} attempts with no progress; "
                  f"see logs/refusals.jsonl", flush=True)
            return False
        print(f"{system}: {remaining} rows to go; trying again in {PAUSE_SECONDS // 60} min", flush=True)
        sleep(PAUSE_SECONDS)
    print(f"{system}: complete", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", required=True)
    ap.add_argument("--dataset", required=True, choices=["d1", "d2"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--systems", required=True, help="comma-separated, run in this order")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--expected", type=int, required=True, help="number of problems in the split, or the limit")
    args = ap.parse_args()
    results = [run_system(s, args, args.expected) for s in args.systems.split(",")]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
