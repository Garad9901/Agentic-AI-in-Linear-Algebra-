"""Run systems on D1 or D3 and write one JSONL record per (item, system, seed).

Examples
  # dry run of the plumbing, no API key needed (records are tagged DRYRUN and cannot become paper tables)
  python scripts/run_benchmark.py --dataset D1 --split dev --backend mock --systems S0,B1,B2,B3,A1,A2 --seeds 0

  # prompt tuning on the dev split with a real model (model 0 in config/models.yaml)
  set GROQ_API_KEY=...        (Windows)   |   export GROQ_API_KEY=...   (macOS/Linux)
  python scripts/run_benchmark.py --dataset D1 --split dev --model 0 --systems S0 --seeds 0 --limit 10

  # the pre-registered test run (refused unless every pre-registration gate passes)
  python scripts/run_benchmark.py --dataset D1 --split test --model 0 --systems S0,B1,B2,B3,A1,A2 --seeds 0,1,2

Output: results/<split>/<dataset>/<model>/<system>/seed<k>.jsonl  (resumable; finished items are skipped,
so a configuration is never run twice on the test split).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latutor import prompts  # noqa: E402
from latutor.generator import load  # noqa: E402
from latutor.llm import MockLLM, OpenAICompatible, load_models  # noqa: E402
from latutor.systems import D1_SYSTEMS, D3_SYSTEMS, Ctx, run_d1, run_d3  # noqa: E402

SEED_BASE = 20261024


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def test_gates(dataset, models_problems):
    """Pre-registration gates for any test-split run. Returns a list of failures."""
    fails = []
    tags = set(git("tag", "--merged", "HEAD").split())
    if models_problems:
        fails.append("config/models.yaml is incomplete (provider, model_id, family, release_date, access_date)")
    if "models-v1" not in tags:
        fails.append("git tag models-v1 is missing or not an ancestor of HEAD (PREREGISTRATION.md s5)")
    if prompts.PROMPT_VERSION not in tags:
        fails.append(f"git tag {prompts.PROMPT_VERSION} is missing: freeze the dev-tuned prompts first")
    if git("status", "--porcelain", "--untracked-files=no"):
        fails.append("working tree has uncommitted changes: commit so every row records an exact build")
    if dataset == "D3":
        tax = json.loads((ROOT / "config/taxonomy.json").read_text())
        if not tax.get("frozen") or "taxonomy-v1" not in tags:
            fails.append("misconception taxonomy is not frozen (config/taxonomy.json frozen=true + tag taxonomy-v1)")
        if not all(r.get("name") for r in _annotators()):
            fails.append("docs/ANNOTATORS.md names are not filled in")
    return fails


def _annotators():
    rows = []
    for line in (ROOT / "docs/ANNOTATORS.md").read_text().splitlines():
        if line.startswith("| Annotator "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append({"name": cells[2] if len(cells) > 2 else ""})
    return rows


def load_d3(path, split):
    """D3 records: see data/d3/README.md. Problems and solutions live outside the repo (data/private/)."""
    probs = {p["id"]: p for p in load(Path(path) / "problems.jsonl")}
    sols = [s for s in load(Path(path) / "solutions.jsonl") if s.get("split") == split and not s.get("excluded")]
    return probs, sols


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                keys.add(r.get("solution_id") or r["problem_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["D1", "D3"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--model", type=int, help="index into config/models.yaml")
    ap.add_argument("--backend", choices=["live", "mock"], default="live")
    ap.add_argument("--limit", type=int, help="first N items only (dev tuning)")
    ap.add_argument("--d1", default="data/d1/problems.jsonl")
    ap.add_argument("--d3", default="data/private/d3")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-repairs", type=int, default=2)
    args = ap.parse_args()

    systems = args.systems.split(",")
    allowed = D1_SYSTEMS if args.dataset == "D1" else D3_SYSTEMS
    for s in systems:
        if s not in allowed:
            sys.exit(f"{s} is not defined for {args.dataset}; allowed: {allowed}")
    seeds = [int(x) for x in args.seeds.split(",")]

    models, problems_in_cfg = load_models(ROOT / "config/models.yaml")
    if args.backend == "mock":
        if args.split == "test":
            sys.exit("Refused: the mock backend may only run on the dev split.")
        model_cfg = {"provider": "mock", "model_id": "mock-dryrun", "temperature": 0.7}
        out_root = Path(args.out) / "dryrun"
    else:
        if args.model is None:
            sys.exit("--model is required with the live backend")
        if args.model in problems_in_cfg or args.model >= len(models):
            sys.exit(f"config/models.yaml entry {args.model} is incomplete; fill it in first")
        model_cfg = models[args.model]
        out_root = Path(args.out)
        if args.split == "test":
            fails = test_gates(args.dataset, problems_in_cfg)
            if fails:
                sys.exit("Refused: test-split run blocked by pre-registration gates:\n  - " + "\n  - ".join(fails))

    if args.dataset == "D1":
        items = [p for p in load(ROOT / args.d1) if p["split"] == args.split]
        problems = {p["id"]: p for p in items}
    else:
        problems, items = load_d3(ROOT / args.d3, args.split)
    if args.limit:
        items = items[: args.limit]

    taxonomy = json.loads((ROOT / "config/taxonomy.json").read_text()).get("labels", [])
    build = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
             "python": sys.version.split()[0]}
    slug = f"{model_cfg['provider']}__{str(model_cfg['model_id']).replace('/', '_').replace(':', '_')}"

    for system in systems:
        for seed in seeds:
            if args.backend == "mock":
                llm = MockLLM(problems=problems)
            else:
                llm = OpenAICompatible(model_cfg["provider"], model_cfg["model_id"],
                                       temperature=float(model_cfg.get("temperature", 0.7)))
            path = out_root / args.split / args.dataset / slug / system / f"seed{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = done_keys(path)
            todo = [it for it in items if (it.get("solution_id") or it["id"]) not in done]
            print(f"[{system} seed{seed}] {len(todo)} to run, {len(done)} already done -> {path}")
            t0 = time.time()
            with open(path, "a", encoding="utf-8") as f:
                for k, it in enumerate(todo, 1):
                    ctx = Ctx(llm=llm, seed=SEED_BASE + seed, max_repairs=args.max_repairs, taxonomy=taxonomy)
                    if args.dataset == "D1":
                        rec = run_d1(system, it, ctx)
                    else:
                        rec = run_d3(system, problems[it["problem_id"]], it, ctx)
                    rec.update(build=build, access_date=model_cfg.get("access_date"),
                               model_family=model_cfg.get("family"), seed_index=seed,
                               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    f.flush()
                    if k % 10 == 0 or k == len(todo):
                        print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
