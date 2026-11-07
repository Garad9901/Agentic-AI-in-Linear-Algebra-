"""Summarise the cost probe and extrapolate to the full prereg-v1 run.

  python scripts/cost_probe.py

Reads results/<system>_d1_dev_run1.csv and writes results/cost_probe.md. Tokens are
total tokens as the provider reports them, including thinking. Means are taken over
(topic, size) cells, so a cell that happens to hold two probe problems does not count
twice in the extrapolation.

"Logical" tokens charge every system for every call. "Live" tokens are what the API
actually bills, because identical prompts in a run are served from the cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import gen_d1, gen_d2  # noqa: E402
from run import RESULTS  # noqa: E402
from systems import SYSTEMS  # noqa: E402

RUN = 1
SEEDS = 3
D1_PROBLEMS = 270


def _cell_mean(frame: pd.DataFrame, column: str) -> float:
    return frame.groupby(["topic", "size"])[column].mean().mean()


def main() -> None:
    frames = {}
    for system in SYSTEMS:
        path = RESULTS / f"{system}_d1_dev_run{RUN}.csv"
        if path.exists():
            frames[system] = pd.read_csv(path, keep_default_na=False)
    missing = [s for s in SYSTEMS if s not in frames]
    lines = ["# Cost probe", "", f"D1 dev, run {RUN}. Model and dates: see RUN_INFO.md.", ""]
    if missing:
        lines += [f"**Incomplete: no results for {missing}.**", ""]

    rows = []
    for system, f in frames.items():
        ok = f[f["status"] != "api_error"]
        calls = int(ok["llm_calls"].sum())
        rows.append({
            "system": system,
            "problems": len(ok),
            "api_errors": len(f) - len(ok),
            "mean_calls": f"{_cell_mean(ok, 'llm_calls'):.2f}",
            "mean_tokens": f"{_cell_mean(ok, 'total_tokens'):.0f}",
            "mean_live_tokens": f"{_cell_mean(ok, 'live_tokens'):.0f}",
            "mean_seconds": f"{(ok['llm_seconds'] + ok['local_seconds']).mean():.1f}",
            "parse_failure_rate_calls": f"{ok['parse_failures'].sum() / calls:.3f} ({int(ok['parse_failures'].sum())}/{calls})" if calls else "",
            "no_usable_reply_rate_problems": f"{(ok['status'] == 'parse_error').mean():.3f} ({int((ok['status'] == 'parse_error').sum())}/{len(ok)})" if len(ok) else "",
        })
    table = pd.DataFrame(rows)
    lines += [_markdown(table), ""]

    if frames and not missing:
        lines += _extrapolation(frames, gen_d2.load())
    else:
        lines += ["## Extrapolation", "", "Blank: it needs all 5 systems on all probe problems, and this probe is incomplete."]
    out = RESULTS / "cost_probe.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


def _markdown(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)])


def _extrapolation(frames: dict, d2: list) -> list:
    d1_each = {s: _cell_mean(f[f["status"] != "api_error"], "total_tokens") for s, f in frames.items()}
    d1_live = {s: _cell_mean(f[f["status"] != "api_error"], "live_tokens") for s, f in frames.items()}
    base = sum(d1_each.values()) * D1_PROBLEMS * SEEDS
    base_live = sum(d1_live.values()) * D1_PROBLEMS * SEEDS
    ours = d1_each.get("OURS", 0.0) * D1_PROBLEMS * SEEDS
    # Ablations are unmeasured. A1 repeats B4 and A3 repeats OURS, so with the cache they cost nothing.
    # A2 and A4 are charged at the OURS rate, an assumption.
    abl_logical, abl_live = 4 * ours, 2 * ours

    runnable = [p for p in d2 if p["runnable"]]
    per_problem = {}
    for system, f in frames.items():
        ok = f[(f["status"] != "api_error") & f["topic"].isin(["determinant", "rref_rank"])]
        per_problem[system] = _cell_mean(ok, "total_tokens") if len(ok) else 0.0
    d2_each = sum(per_problem.values()) * SEEDS
    d2_runnable, d2_all = d2_each * len(runnable), d2_each * len(d2)

    total_low = base_live + abl_live + d2_runnable
    total_high = base + abl_logical + d2_all
    rows = pd.DataFrame([
        {"part": f"D1: {D1_PROBLEMS} x {SEEDS} seeds x {len(frames)} systems (logical)", "tokens": f"{base:,.0f}"},
        {"part": "D1, same, live tokens after the cache", "tokens": f"{base_live:,.0f}"},
        {"part": "Ablations A1-A4 at the OURS rate (logical)", "tokens": f"{abl_logical:,.0f}"},
        {"part": "Ablations, live: A2 and A4 only", "tokens": f"{abl_live:,.0f}"},
        {"part": f"LinAlg-Bench, runnable subset: {len(runnable)} x {SEEDS} x {len(frames)}", "tokens": f"{d2_runnable:,.0f}"},
        {"part": f"LinAlg-Bench, if all {len(d2)} could run", "tokens": f"{d2_all:,.0f}"},
        {"part": "**Total, low: live tokens, runnable LinAlg-Bench**", "tokens": f"**{total_low:,.0f}**"},
        {"part": "**Total, high: logical tokens, all LinAlg-Bench**", "tokens": f"**{total_high:,.0f}**"},
    ])
    return ["## Extrapolation", "", _markdown(rows), "",
            "- One model only. A second model, as prereg-v1 s5 plans, would double these totals.",
            "- D3 is excluded: no student data exists yet.",
            f"- LinAlg-Bench cost is estimated from the D1 determinant and rref_rank cells. Sizes {gen_d1.SIZES}.",
            "- Probe means come from a few problems per cell, so these are order-of-magnitude figures."]


if __name__ == "__main__":
    main()
