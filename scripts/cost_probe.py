"""Summarise a cost probe and extrapolate tokens and USD to the full design.

  python scripts/cost_probe.py --provider groq

Reads results/<system>_d1_dev_run1__<provider>__<model>.csv and writes
results/cost_probe_<provider>.md. Means are taken over (topic, size) cells, so a cell
holding two probe problems does not count twice. Extrapolations use live tokens: what
the API bills, after identical prompts in a run are served from the cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run import RESULTS, RUNS, model_for, result_path  # noqa: E402
from systems import SYSTEMS  # noqa: E402

RUN = 1
D1_PROBLEMS = 270
D2_PROBLEMS = 210
D2_TOPICS = ["determinant", "rref_rank"]
GEMINI_FREE_PROBE = RESULTS / "archive" / "gemini_free_tier_probe" / "B1_d1_dev_run1.csv"

# USD per million tokens, as (input, output). Read on 2026-11-07.
GROQ_PRICE = (0.15, 0.60)  # Groq API, GET /models/openai/gpt-oss-120b
GEMINI_BATCH_PRICES = {  # https://aistudio.google.com/docs/pricing, gemini-3.8-flash batch
    "through 2026-12-31": (0.375, 1.875),
    "from 2027-01-01": (0.75, 3.75),
}


def cell_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby(["topic", "size"])[column].mean().mean())


def usd(input_tokens: float, output_tokens: float, price: tuple) -> float:
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


def design_tokens(per_problem: dict, d2_per_problem: dict, runs: int = len(RUNS)) -> dict:
    """Total (input, output) tokens for one model.

    per_problem and d2_per_problem map system -> (input, output) per problem. On D1:
    270 x runs x each system. On D2: 210 x runs x each system, costed from the D1
    determinant and rref_rank cells. Ablations: A1 repeats B4 and A3 repeats OURS, so
    the cache makes them free; A2 and A4 are charged at the OURS rate, an assumption.
    """
    total_in = total_out = 0.0
    for system, (i, o) in per_problem.items():
        total_in += D1_PROBLEMS * runs * i
        total_out += D1_PROBLEMS * runs * o
    for system, (i, o) in d2_per_problem.items():
        total_in += D2_PROBLEMS * runs * i
        total_out += D2_PROBLEMS * runs * o
    ours_in, ours_out = per_problem["OURS"]
    total_in += 2 * D1_PROBLEMS * runs * ours_in
    total_out += 2 * D1_PROBLEMS * runs * ours_out
    return {"input": total_in, "output": total_out}


def markdown(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)])


def summary_row(system: str, f: pd.DataFrame) -> dict:
    ok = f[f["status"] != "api_error"]
    calls = int(ok["llm_calls"].sum())
    if ok.empty:
        return {"system": system, "problems": 0, "api_errors": len(f)}
    return {
        "system": system,
        "problems": len(ok),
        "api_errors": len(f) - len(ok),
        "mean_calls": f"{cell_mean(ok, 'llm_calls'):.2f}",
        "mean_input_tokens": f"{cell_mean(ok, 'input_tokens'):.0f}",
        "mean_output_tokens": f"{cell_mean(ok, 'output_tokens'):.0f}",
        "mean_seconds": f"{(ok['llm_seconds'] + ok['local_seconds']).mean():.1f}",
        "parse_failures": f"{int(ok['parse_failures'].sum())}/{calls}",
        "truncation_rate": f"{ok['truncated_calls'].sum() / calls:.3f} ({int(ok['truncated_calls'].sum())}/{calls})" if calls else "",
        "rate_limit_retries": int(ok["rate_limit_errors"].sum()),
    }


def live_means(frames: dict) -> tuple:
    per_problem, d2 = {}, {}
    for system, f in frames.items():
        ok = f[f["status"] != "api_error"]
        per_problem[system] = (cell_mean(ok, "live_input_tokens"), cell_mean(ok, "live_output_tokens"))
        cell = ok[ok["topic"].isin(D2_TOPICS)]
        d2[system] = (cell_mean(cell, "live_input_tokens"), cell_mean(cell, "live_output_tokens"))
    return per_problem, d2


def gemini_ratio(groq_b1: pd.DataFrame) -> tuple:
    """Gemini tokens per call / Groq tokens per call, for input and output, on B1 calls only."""
    g = pd.read_csv(GEMINI_FREE_PROBE, keep_default_na=False)
    g = g[g["status"] != "api_error"]
    calls = g["llm_calls"].sum()
    gem_in = g["prompt_tokens"].sum() / calls
    gem_out = (g["total_tokens"].sum() - g["prompt_tokens"].sum()) / calls
    ok = groq_b1[groq_b1["status"] != "api_error"]
    calls = ok["llm_calls"].sum()
    return gem_in / (ok["input_tokens"].sum() / calls), gem_out / (ok["output_tokens"].sum() / calls), int(g["llm_calls"].sum())


def extrapolate(frames: dict) -> list:
    per_problem, d2 = live_means(frames)
    groq = design_tokens(per_problem, d2)
    ratio_in, ratio_out, gemini_calls = gemini_ratio(frames["B1"])
    gemini = {"input": groq["input"] * ratio_in, "output": groq["output"] * ratio_out}
    rows = [{"run": "Groq, synchronous", "input_tokens": f"{groq['input']:,.0f}", "output_tokens": f"{groq['output']:,.0f}",
             "total_tokens": f"{groq['input'] + groq['output']:,.0f}", "price_per_M": f"${GROQ_PRICE[0]} in, ${GROQ_PRICE[1]} out",
             "usd": f"${usd(groq['input'], groq['output'], GROQ_PRICE):,.2f}"}]
    for label, price in GEMINI_BATCH_PRICES.items():
        rows.append({"run": f"Gemini Batch, {label}", "input_tokens": f"{gemini['input']:,.0f}",
                     "output_tokens": f"{gemini['output']:,.0f}", "total_tokens": f"{gemini['input'] + gemini['output']:,.0f}",
                     "price_per_M": f"${price[0]} in, ${price[1]} out", "usd": f"${usd(gemini['input'], gemini['output'], price):,.2f}"})
    per_system = pd.DataFrame([{"system": s, "D1 input per problem": f"{v[0]:.0f}", "D1 output per problem": f"{v[1]:.0f}",
                                "D2 input per problem": f"{d2[s][0]:.0f}", "D2 output per problem": f"{d2[s][1]:.0f}"}
                               for s, v in per_problem.items()])
    return ["## Extrapolation", "",
            "Live tokens per problem, used below:", "", markdown(per_system), "",
            f"Formula: total = {D1_PROBLEMS} x {len(RUNS)} x (sum over 5 systems of D1 tokens) "
            f"+ {D2_PROBLEMS} x {len(RUNS)} x (sum over 5 systems of D2 tokens) "
            f"+ 2 x {D1_PROBLEMS} x {len(RUNS)} x (OURS D1 tokens), for input and output separately. "
            "USD = input/1e6 x input price + output/1e6 x output price.", "",
            f"Gemini tokens = Groq tokens x ({ratio_in:.3f} for input, {ratio_out:.3f} for output), the ratio of "
            f"tokens per B1 call on {gemini_calls} Gemini calls from the free-tier probe.", "",
            markdown(pd.DataFrame(rows)), "",
            "- A2 and A4 are charged at the OURS rate, an assumption. A1 and A3 repeat B4 and OURS, so the cache makes them free.",
            "- D3 is excluded: no student data exists yet.",
            "- LinAlg-Bench cost is estimated from the D1 determinant and rref_rank cells.",
            "- The Gemini ratio comes from few B1 calls and is applied to every agent, so the Gemini figures are rough.",
            "- These are one probe of 20 problems: order-of-magnitude figures, not measurements of the full run."]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", required=True, choices=["groq", "gemini"])
    args = ap.parse_args()
    model = model_for(args.provider)
    frames = {}
    for system in SYSTEMS:
        path = result_path(system, "d1", "dev", RUN, args.provider, model)
        if path.exists():
            frames[system] = pd.read_csv(path, keep_default_na=False)
    missing = [s for s in SYSTEMS if s not in frames]
    incomplete = [s for s, f in frames.items() if (f["status"] == "api_error").any() or len(f) < 20]
    lines = ["# Cost probe", "", f"Provider {args.provider}, model {model}, D1 dev, run {RUN}.", ""]
    if missing:
        lines += [f"**Incomplete: no results for {missing}.**", ""]
    if incomplete:
        lines += [f"**Incomplete: {incomplete} did not finish all 20 problems.**", ""]
    if frames:
        table = pd.DataFrame([summary_row(s, f) for s, f in frames.items()]).astype(object)
        table = table.where(table.notna(), "")
        lines += [markdown(table), ""]
    if args.provider == "groq" and not missing and not incomplete:
        lines += extrapolate(frames)
    else:
        lines += ["## Extrapolation", "", "Blank: it needs all 5 Groq systems on all 20 problems, and this probe "
                  "did not finish. Extrapolating from what did finish would not be a measurement."]
    out = RESULTS / f"cost_probe_{args.provider}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
