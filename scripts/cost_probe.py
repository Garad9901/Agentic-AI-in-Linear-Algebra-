"""Summarise the cost probe and project tokens and USD for the full design.

  python scripts/cost_probe.py

Reads results/<system>_d1_dev_run1__groq__<model>.csv and writes results/cost_probe_groq.md.
All reported results come from this one model (decision of 2026-11-07). Means are taken
over (topic, size) cells, so a cell holding two probe problems does not count twice.
Projections use live tokens: what the API bills, after identical prompts in a run are
served from the cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run import RESULTS, RUNS, model_for, result_path  # noqa: E402
from systems import SYSTEMS  # noqa: E402

PROVIDER = "groq"
RUN = 1
PROBE_PROBLEMS = 20
D1_PROBLEMS = 270
D2_PROBLEMS = 210
D2_TOPICS = ["determinant", "rref_rank"]
# USD per million tokens, (input, output). Groq API, GET /models/openai/gpt-oss-120b, read 2026-11-07.
GROQ_PRICE = (0.15, 0.60)


def cell_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby(["topic", "size"])[column].mean().mean())


def usd(input_tokens: float, output_tokens: float, price: tuple = GROQ_PRICE) -> float:
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


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


def per_problem(frames: dict, topics=None) -> dict:
    """system -> live (input, output) per problem in run 1, and in runs without agent hints."""
    out = {}
    for system, f in frames.items():
        ok = f[f["status"] != "api_error"]
        if topics:
            ok = ok[ok["topic"].isin(topics)]
        run1 = (cell_mean(ok, "live_input_tokens"), cell_mean(ok, "live_output_tokens"))
        hints = (cell_mean(ok, "live_hint_input_tokens"), cell_mean(ok, "live_hint_output_tokens"))
        out[system] = {"run1": run1, "later": (run1[0] - hints[0], run1[1] - hints[1])}
    return out


def design_tokens(d1: dict, d2: dict, runs: int = len(RUNS)) -> dict:
    """Total live (input, output) tokens for the full design.

    D1: 270 problems; run 1 with hints, the other runs without agent hints (decision of 2026-11-07).
    D2: 210 problems x runs, with hints in every run, costed from the D1 determinant and rref_rank cells.
    Ablations: A1 repeats B4 and A3 repeats OURS, so the cache makes them free. A2 and A4 run on D1
    and are charged at the OURS rate, an assumption.
    """
    def d1_cost(rates):
        return tuple(D1_PROBLEMS * (rates["run1"][k] + (runs - 1) * rates["later"][k]) for k in (0, 1))

    total = [0.0, 0.0]
    parts = {}
    for system, rates in d1.items():
        parts[f"D1 {system}"] = d1_cost(rates)
    for system, rates in d2.items():
        parts[f"D2 {system}"] = tuple(D2_PROBLEMS * runs * rates["run1"][k] for k in (0, 1))
    ours = d1_cost(d1["OURS"])
    parts["Ablations A2 + A4"] = (2 * ours[0], 2 * ours[1])
    for i, o in parts.values():
        total[0] += i
        total[1] += o
    return {"input": total[0], "output": total[1], "parts": parts}


def projection(frames: dict) -> list:
    d1, d2 = per_problem(frames), per_problem(frames, D2_TOPICS)
    design = design_tokens(d1, d2)
    rates = pd.DataFrame([{"system": s, "run 1 input": f"{v['run1'][0]:.0f}", "run 1 output": f"{v['run1'][1]:.0f}",
                           "later runs input": f"{v['later'][0]:.0f}", "later runs output": f"{v['later'][1]:.0f}",
                           "D2 input": f"{d2[s]['run1'][0]:.0f}", "D2 output": f"{d2[s]['run1'][1]:.0f}"}
                          for s, v in d1.items()])
    parts = pd.DataFrame([{"part": k, "input_tokens": f"{i:,.0f}", "output_tokens": f"{o:,.0f}",
                           "usd": f"${usd(i, o):,.2f}"} for k, (i, o) in design["parts"].items()])
    total = design["input"] + design["output"]
    return ["## Projection for the full run", "",
            "Live tokens per problem. Later runs have no agent hints on D1; B1 and B3 still write hints in their one call:",
            "", markdown(rates), "",
            f"Formula, for input and output separately: D1 = {D1_PROBLEMS} x (run 1 + {len(RUNS) - 1} x later runs), "
            f"summed over the 5 systems; D2 = {D2_PROBLEMS} x {len(RUNS)} x D2 rate; ablations = 2 x the D1 cost of "
            f"OURS. USD = input / 1e6 x ${GROQ_PRICE[0]} + output / 1e6 x ${GROQ_PRICE[1]}.", "",
            markdown(parts), "",
            f"**Total: {design['input']:,.0f} input + {design['output']:,.0f} output = {total:,.0f} tokens, "
            f"${usd(design['input'], design['output']):,.2f}.**", "",
            "- A1 and A3 repeat B4 and OURS, so the cache makes them free. A2 and A4 are charged at the OURS rate.",
            "- D3 is excluded: no student data exists yet.",
            "- D2 cost is estimated from the D1 determinant and rref_rank cells.",
            "- One probe of 20 problems: an order-of-magnitude projection, not a measurement of the full run."]


def main() -> None:
    model = model_for(PROVIDER)
    frames = {}
    for system in SYSTEMS:
        path = result_path(system, "d1", "dev", RUN, PROVIDER, model)
        if path.exists():
            frames[system] = pd.read_csv(path, keep_default_na=False)
    missing = [s for s in SYSTEMS if s not in frames]
    incomplete = [s for s, f in frames.items() if (f["status"] == "api_error").any() or len(f) < PROBE_PROBLEMS]
    lines = ["# Cost probe", "", f"Provider {PROVIDER}, model {model}, D1 dev, run {RUN}.", ""]
    if missing:
        lines += [f"**Incomplete: no results for {missing}.**", ""]
    if incomplete:
        lines += [f"**Incomplete: {incomplete} did not finish all {PROBE_PROBLEMS} problems.**", ""]
    if frames:
        table = pd.DataFrame([summary_row(s, f) for s, f in frames.items()]).astype(object)
        lines += [markdown(table.where(table.notna(), "")), ""]
    if not missing and not incomplete:
        lines += projection(frames)
    else:
        lines += ["## Projection for the full run", "", f"Blank: it needs all 5 systems on all {PROBE_PROBLEMS} "
                  "problems, and this probe did not finish. Projecting from what did finish would not be a measurement."]
    out = RESULTS / f"cost_probe_{PROVIDER}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
