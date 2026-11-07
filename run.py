"""Run one system on one dataset split and write one row per problem.

  python run.py --provider groq --system OURS --dataset d1 --split dev --run 1 --limit 5
  python run.py --provider gemini --batch --system OURS --dataset d1 --split dev --run 1

The test split refuses to run unless a file named FROZEN exists (prompts frozen).
Rows hold what the system produced, never grades: score.py grades them. --limit
takes problems round-robin across topic and size, so a short run still covers
several of each. Reruns are free for anything already in the LLM cache. --batch uses
the Gemini Batch API: rerun the same command to resume after a crash or sleep.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

import batch as batch_mode
import d3 as d3_data
from agents.diagnostic import load_taxonomy
from data import gen_d1, gen_d2
from llm import BatchPending, LLMError, _settings, configure, track, use_batch, use_run
from systems import ABLATIONS, STUDENT_ONLY, SYSTEMS, run_system

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FROZEN = ROOT / "FROZEN"
RUN_INFO = RESULTS / "RUN_INFO.md"
MODELS = ROOT / "config" / "models.yaml"
RUNS = [1, 2, 3]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
MAX_CONSECUTIVE_API_ERRORS = 3


def round_robin(problems: list[dict]) -> list[dict]:
    """One problem from each (topic, size) cell, then a second from each, and so on.

    Consecutive cells change both topic and size, so a short --limit spans the sizes too.
    """
    cells: dict = {}
    for p in problems:
        cells.setdefault((p["topic"], p["size"]), []).append(p)
    wanted = [(gen_d1.TOPICS[k % 6], gen_d1.SIZES[(k + k // 6) % 3]) for k in range(18)]
    keys = [k for k in wanted if k in cells]
    ordered = []
    for i in range(max((len(c) for c in cells.values()), default=0)):
        ordered.extend(cells[k][i] for k in keys if i < len(cells[k]))
    return ordered


def load_dataset(dataset: str, split: str) -> list[dict]:
    if dataset == "d1":
        problems = gen_d1.load()
    elif dataset == "d2":
        problems = [p for p in gen_d2.load() if p["runnable"]]
    else:
        try:
            problems = d3_data.load_problems()
        except d3_data.D3Unavailable as e:
            sys.exit(f"cannot run D3: {e}. Fill data/private/d3.csv from data/d3_template.csv with real solutions.")
    return [p for p in problems if p["split"] == split]


def result_path(system: str, dataset: str, split: str, run: int, provider: str, model: str) -> Path:
    slug = model.replace("/", "-")
    return RESULTS / f"{system}_{dataset}_{split}_run{run}__{provider}__{slug}.csv"


def model_for(provider: str) -> str:
    """The exact model ID registered for this provider in config/models.yaml (tag models-v1)."""
    entries = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["models"]
    found = [m["model_id"] for m in entries if m["provider"] == provider]
    if len(found) != 1:
        raise SystemExit(f"config/models.yaml must list exactly one model for {provider}, found {found}")
    return found[0]


def log_run(system: str, dataset: str, split: str, run: int, problems: int, outcome: str, mode: str = "sync") -> None:
    cfg = _settings()
    if not RUN_INFO.exists():
        RUN_INFO.write_text("# Run log\n\nOne row per run.py command that finished or stopped.\n\n"
                            "| date | system | dataset | split | run | problems | provider | model | temperature "
                            "| python | mode | outcome |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(RUN_INFO, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | {system} | {dataset} | {split} | {run} | {problems} "
                f"| {cfg['provider']} | {cfg['model']} | {cfg['temperature']} | {platform.python_version()} | {mode} | {outcome} |\n")


def _tokens(call: dict) -> int:
    reported = call.get("total_tokens")
    return reported if reported is not None else (call.get("prompt_tokens") or 0) + (call.get("completion_tokens") or 0)


def _output_tokens(call: dict) -> int:
    """Everything but the prompt. Gemini reports thinking only in the total."""
    return _tokens(call) - (call.get("prompt_tokens") or 0)


def _usage(calls: list[dict]) -> dict:
    live = [c for c in calls if not c["cached"]]
    return {
        "llm_calls": len(calls),
        "api_calls": len(live),
        "total_tokens": sum(_tokens(c) for c in calls),
        "live_tokens": sum(_tokens(c) for c in live),
        "input_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
        "output_tokens": sum(_output_tokens(c) for c in calls),
        "live_input_tokens": sum(c.get("prompt_tokens") or 0 for c in live),
        "live_output_tokens": sum(_output_tokens(c) for c in live),
        "parse_failures": sum(bool(c.get("parse_failed")) for c in calls),
        "truncated_calls": sum(c.get("finish_reason") == "length" for c in calls),
        "llm_seconds": round(sum(c.get("latency_s") or 0 for c in calls), 3),
        "rate_limit_errors": sum(c.get("rate_limit_errors") or 0 for c in calls),
        "timeouts": sum(c.get("timeouts") or 0 for c in calls),
    }


def run_one(system: str, problem: dict, taxonomy: list, run: int, mode: str = "sync") -> dict:
    t0 = time.time()
    error = ""
    with track() as calls, use_run(run):
        try:
            outcome = run_system(system, problem, taxonomy)
        except LLMError as e:
            error = str(e)[:300]
            outcome = {"status": "api_error"}
        except BatchPending:
            outcome = {"status": "pending"}
    usage = _usage(calls)
    local = max(0.0, time.time() - t0 - sum(c.get("request_s") or 0 for c in calls))
    return {
        "problem_id": problem["id"],
        "run": run,
        "mode": mode,
        "topic": problem["topic"],
        "size": problem["size"],
        "status": outcome["status"],
        "answer_json": json.dumps(outcome.get("answer")),
        "trace_json": json.dumps(outcome.get("trace")),
        "hints_json": json.dumps(outcome.get("hints", [])),
        "self_check": outcome.get("self_check") or "",
        "retries": outcome.get("retries", 0),
        "verify_reasons": json.dumps(outcome.get("verify_reasons", [])),
        "unchecked_steps": outcome.get("unchecked_steps"),
        "weak_steps": outcome.get("weak_steps"),
        "pred_first_wrong_step": outcome.get("pred_first_wrong_step"),
        "pred_misconception": outcome.get("pred_misconception") or "",
        "flags_json": json.dumps({"steps": outcome["step_flags"], "answer": outcome["answer_flag"]})
        if outcome.get("step_flags") is not None else "null",
        "error": error,
        **usage,
        "local_seconds": round(local, 3),
    }


def run_batch(args, problems: list, taxonomy: list, path: Path) -> None:
    def replay(collector):
        with use_batch(collector):
            return [run_one(args.system, p, taxonomy, args.run, "batch") for p in problems]

    rows = batch_mode.drive(_settings(), path.stem, replay)
    pd.DataFrame(rows).to_csv(path, index=False)
    log_run(args.system, args.dataset, args.split, args.run, len(rows), "complete", "batch")
    print(f"wrote {path}")


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs. It lapses when the process exits."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    keep_awake()
    ap = argparse.ArgumentParser(description="Run a system on a dataset split.")
    ap.add_argument("--provider", required=True, choices=["groq", "gemini"])
    ap.add_argument("--batch", action="store_true", help="Gemini Batch API; rerun the command to resume")
    ap.add_argument("--system", required=True, choices=SYSTEMS + ABLATIONS + STUDENT_ONLY)
    ap.add_argument("--dataset", required=True, choices=["d1", "d2", "d3"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--run", type=int, choices=RUNS, default=1, help="seed index, prereg-v1 s6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="problems at once; requests stay within LLM_RPM")
    args = ap.parse_args()

    if args.system in STUDENT_ONLY and args.dataset != "d3":
        sys.exit(f"{args.system} needs student work: use --dataset d3")
    if args.batch and args.provider != "gemini":
        sys.exit("--batch is for --provider gemini only")
    if args.split == "test" and not FROZEN.exists():
        sys.exit("refusing to run the test split: create the file FROZEN once the prompts are frozen")
    if args.dataset == "d3" and args.split == "test" and not d3_data.taxonomy_v1_ready():
        sys.exit("refusing to run the D3 test split: the taxonomy must be frozen and tagged taxonomy-v1")

    problems = round_robin(load_dataset(args.dataset, args.split))
    if args.limit is not None:
        problems = problems[:args.limit]
    taxonomy = load_taxonomy()
    model = model_for(args.provider)
    configure(args.provider, model)
    path = result_path(args.system, args.dataset, args.split, args.run, args.provider, model)
    RESULTS.mkdir(exist_ok=True)
    if args.batch:
        run_batch(args, problems, taxonomy, path)
        return

    order = {p["id"]: i for i, p in enumerate(problems)}
    rows: list[dict] = []
    consecutive_errors = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, args.system, p, taxonomy, args.run) for p in problems]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: order[r["problem_id"]])  # the file order does not depend on --workers
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"[{len(rows)}/{len(problems)}] {row['problem_id']}: {row['status']} (calls {row['llm_calls']}, "
                  f"live {row['api_calls']}, timeouts {row['timeouts']}) {row['error']}", flush=True)
            consecutive_errors = consecutive_errors + 1 if row["status"] == "api_error" else 0
            if consecutive_errors >= MAX_CONSECUTIVE_API_ERRORS:
                pool.shutdown(wait=False, cancel_futures=True)
                log_run(args.system, args.dataset, args.split, args.run, len(rows), "stopped: API errors")
                sys.exit(f"stopping after {consecutive_errors} API errors in a row: {row['error']}\n"
                         "Finished calls are cached, so rerunning the same command resumes at no cost.")
    log_run(args.system, args.dataset, args.split, args.run, len(rows), "complete")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
