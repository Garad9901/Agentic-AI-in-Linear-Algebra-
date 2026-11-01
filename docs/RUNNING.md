# How to run the experiments

Commands are for Windows PowerShell; on macOS/Linux use `export NAME=value` instead of `$env:NAME="value"`.

## 0. Install and self-check (once)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q          # 10 tests; also confirms the problem bank regenerates identically (~3 min)
```

## 1. What is already done (no LLM needed)

| Step | Command | Output |
|---|---|---|
| Problem bank D1 (270 problems, seed 20261024) | `python -c "from latutor import generator as g; from pathlib import Path; g.write(Path('data/d1/problems.jsonl'))"` | `data/d1/problems.jsonl` |
| D1 statistics (T1, D1 part) | `python scripts/d1_stats.py` | `results/T1_d1_bank.csv` |
| Verifier validation (engineering check) | `python scripts/validate_verifier.py` | `results/verifier_validation/` |
| Pipeline dry run (mock model, dev only) | `python scripts/run_benchmark.py --dataset D1 --split dev --backend mock --systems S0,B1,B2,B3,A1,A2 --seeds 0` | `results/dryrun/...` (never results) |

## 2. Before any LLM run (gates the runner enforces for the test split)

1. Choose the two models (two families, free tier, released after 2025-04-24), fill `config/models.yaml`,
   commit, and tag: `git tag models-v1`.
2. Approve the rows in `docs/PROPOSED_DEVIATIONS.md` and copy them into `docs/DEVIATIONS.md`.
3. Set the API key in the shell only (never in a file in the repo):
   `$env:GROQ_API_KEY="..."` or `$env:GEMINI_API_KEY="..."`.

## 3. Dev split: tune prompts (allowed to repeat)

```powershell
python scripts/run_benchmark.py --dataset D1 --split dev --model 0 --systems S0,B1 --seeds 0 --limit 12
python scripts/analyse.py --split dev
```
Edit `latutor/prompts.py` only on dev results. When done: commit, then `git tag prompts-v1`.

## 4. Test split: the pre-registered run (once per configuration)

```powershell
python scripts/run_benchmark.py --dataset D1 --split test --model 0 --systems S0,B1,B2,B3,A1,A2 --seeds 0,1,2
python scripts/run_benchmark.py --dataset D1 --split test --model 1 --systems S0,B1,B2,B3,A1,A2 --seeds 0,1,2
python scripts/analyse.py --split test
```
The runner refuses the test split if `models.yaml` is incomplete, the `models-v1` / `prompts-v1` tags are
missing, or the working tree has uncommitted changes. It resumes where it stopped and never re-runs a
finished item, so rate-limit interruptions are safe. Spread runs over days to stay within free-tier limits
(about 11,700 calls per model on D1).

## 5. Student data (D3), after collection closes on 4 Nov 2026

1. Transcribe into `data/private/d3/` using `data/d3_template/README.md` (git-ignored; never commit).
2. `python scripts/check_d3.py data/private/d3 --write`  (validates and assigns the 30/70 split by student).
3. Annotate the dev split, build the taxonomy in `config/taxonomy.json`, set `"frozen": true`, commit, `git tag taxonomy-v1`.
4. Fill `docs/ANNOTATORS.md`, then annotate the test split.
5. Run `--dataset D3` with systems `S0,B1,B4,A3,A4`, then `python scripts/analyse.py --split test`.

## Output tables

| File | Paper table |
|---|---|
| `results/T1_d1_bank.csv`, `T1b_kappa.csv` | T1 |
| `T2_wrong_claim_rate.csv` | T2 (RQ1) |
| `T3_coverage_claims.csv`, `T3b_coverage_student_steps.csv` | T3 (RQ4) |
| `T4_first_error.csv`, `T4b_detection_and_labels.csv` | T4 (RQ2) |
| `T5_ablations_d1.csv` (+ A3/A4 rows in T4/T6) | T5 |
| `T6_hint_leakage_auto.csv` (+ human rubric) | T6 (RQ3) |
| `T7_cost_d1.csv` | T7 |
| `tests.csv` | McNemar, odds ratios, Holm-adjusted p |
