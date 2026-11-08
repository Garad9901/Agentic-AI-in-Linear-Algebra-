# Running the benchmark

Python 3.10.9, CPU only. The plan is [PREREGISTRATION.md](PREREGISTRATION.md), and every change to it is in [DEVIATIONS.md](DEVIATIONS.md).

No number in any table comes from anywhere but a real model reply, a dataset file or an exact computation. There is no mock model. A cell that cannot be computed stays blank, with the reason under the table.

## Design for this paper

The reduced design decided on 2026-11-08 (see [DEVIATIONS.md](DEVIATIONS.md)):

- **Systems:** OURS, B1, B2 and B3. These are the RQ1 primary comparisons.
- **Data:** the D1 test split, 216 problems, with dev for tuning.
- **Runs:** run 1 only, so there is no run-to-run variance. The registered sample-size reasoning assumed 3 runs, so the paper reports this as a limitation.
- **Hints:** none. No system makes hints on D1 or D2, and the B1 and B3 prompts no longer ask for them.
- **Statistics:** RQ4 coverage comes from the same OURS outputs. The Holm family is the 3 RQ1 comparisons.
- **Not run:** runs 2 and 3, LinAlg-Bench (D2), B4, the ablations and hints. RQ2 and RQ3 need student data (D3) and are deferred to a later paper.

Provider refusals and truncated responses are logged in `logs/refusals.jsonl` and `logs/truncations.jsonl`. They are set aside and counted, never scored as wrong answers or abstentions.

## Setup (PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env          # GROQ_API_KEY and GEMINI_API_KEY
python data/gen_d1.py                # 270 problems, seed 20261024
python data/gen_d2.py                # LinAlg-Bench, pinned revision
python -m pytest -q
```

`.env` is git-ignored. `--provider` chooses the provider, and the model always comes from `config/models.yaml` (tag `models-v1`).

## Dev split

```powershell
python tools/run_until_done.py --provider cerebras --dataset d1 --split dev --systems B1,B2,B3,OURS --expected 54
python score.py --split dev
python tools/claim_audit.py sample --split dev
```

`run_until_done.py` calls `run.py` with `--wait-daily`, so a daily token cap pauses the run until it resets. After a crash, sleep or network loss, the next attempt resumes from the cache, so finished calls cost nothing. It keeps going until no row is a provider refusal.

Only Cerebras results are scored. The Groq probe is kept in `results/archive/groq_probe/` as dev-only cost evidence (`python scripts/cost_probe.py`). The Gemini provider and Batch mode stay in the repository but produce no reported results. For reference, Batch mode writes the same rows, with no latency, and resumes when the same command is rerun:

```powershell
python run.py --provider gemini --batch --system OURS --dataset d1 --split dev --run 1
```

The Batch API answers only the calls it is given, and the agents make calls in sequence, so a batch run works in rounds: see `batch.py`. Each round can take up to 24 hours. A job left over 48 hours expires, and its requests are submitted again.

Result files are named `<system>_<dataset>_<split>_run<k>__<provider>__<model>.csv`, and `score.py` writes one folder of tables per model.

`--system` takes B1, B2, B3, B4, OURS, A1 to A4, and DET (D3 only). `--dataset` takes d1, d2 or d3. `--run` takes 1, 2 or 3.

`--limit` takes problems round-robin across topic and size: 5 problems cover 5 topics and all 3 sizes, and 18 cover every cell once. `--workers` runs several problems at once. Requests stay within `LLM_RPM`, and a single call can take minutes on larger matrices, so a sequential run is slow.

Tune `agents/prompts.py` on the dev split only.

## Test split

`run.py` refuses the test split until a file named `FROZEN` exists. **Do not create it yet.** First resolve every row marked pending in [DEVIATIONS.md](DEVIATIONS.md).

Every response is cached under `cache/` by (provider, model, run, prompt hash), so rerunning a stopped command resumes at no API cost, and each run still gets a fresh response. Every call is logged to `logs/calls.jsonl`, and every completed command to `results/RUN_INFO.md`.

## Claim audit

Two humans check 10% of the claim extractions on dev outputs before `FROZEN` exists, and again on 10% of test outputs after the test run:

```powershell
python tools/claim_audit.py sample --split dev   # writes audit/claims_dev_rater1.csv and rater2.csv
python tools/claim_audit.py agree --split dev    # percent agreement and Cohen's kappa
```

Raters fill in `extraction_correct` with y or n, and work independently. `sample` never overwrites a sheet.

## Layout

| Path | What it is |
|---|---|
| `llm.py` | `call_llm(role, prompt)`: temperature 0.7, disk cache, backoff, logging |
| `batch.py` | Gemini Batch API rounds, with resume |
| `extraction.py` | Turns a response into the claims it makes, for RQ1 and RQ4 |
| `data/gen_d1.py`, `data/gen_d2.py` | Build `data/d1.jsonl` and `data/d2.jsonl` |
| `data/SOURCES.md` | Source, version and licence of each dataset |
| `d3.py` | Loads student data from `data/private/d3.csv` |
| `verifier/certificates.py` | Final-answer certificates |
| `verifier/steps.py` | Step checker and row-operation parser |
| `verifier/leak.py` | Leak checker |
| `verifier/claims.py` | Checks the claims in a hint |
| `agents/` | Solver, Verification, LLM verifier, Diagnostic, Hint, JSON parser, prompts |
| `systems.py` | All systems and ablations |
| `grader.py` | Grades against stored ground truth; independent of the certificates |
| `run.py`, `score.py` | Runner, and tables T1 to T8 |
| `latutor/stats.py` | McNemar, odds ratio, Holm, bootstrap. Still in use by `score.py` |
| `scripts/cost_probe.py` | Cost summary and extrapolation |

The rest of `latutor/` and the older scripts are kept for the record, but they no longer produce results.

## Systems

| ID | What it does on D1 | What reaches the student |
|---|---|---|
| B1 | One prompt solves and writes hints | Everything, unchecked |
| B2 | Solver, then an LLM verifier judges the solution and each hint | What the LLM approves. Otherwise it abstains or withholds |
| B3 | B1, then a self-check prompt | Everything, unchecked |
| B4 | Solver, then a Hint call per level | Everything, unchecked |
| OURS | Solver with up to 2 repairs, then checked hints with up to 2 regenerations each | Everything except claims still refuted. Claims it cannot certify, including weakly checked ones, carry an "unverified" flag. Hints are withheld only if they leak or hold a refuted claim |
| A1 | OURS without verification: the same configuration as B4, reported as one result | Everything, unchecked |
| A2 | OURS verifying the final answer only | As OURS, with every step flagged unverified |
| A3 | OURS without the Diagnostic Agent: the same as OURS on D1 and D2, reported as one result there | As OURS |
| A4 | OURS with one hint, not three levels | As OURS |
| DET | Zero-shot first-error detector, D3 only | No answer or hints |

## Definitions

- **RQ1 wrong-claim rate** (T2b): the share of responses with at least 1 delivered claim that contradicts the ground truth, graded by `grader.py`. Reported as mean, SD and range over runs with a clustered bootstrap CI. A response that abstains delivers no claims, so read it with the abstention rate.
- **RQ4 coverage** (T3d): certified claims / all extracted claims in what OURS delivers, from the verifier. The decidable share adds refuted claims. A row operation is certified only if applying its stated operation reproduces the next matrix exactly. Claims that pass on row-equivalence alone are reported as `weakly_checked_share` and never count as certified.
- **Primary tests** (T8): McNemar, exact when discordant pairs < 25, on the per-item majority of 3 runs, with Holm across the 4 registered comparisons. Per-run tests are a sensitivity analysis. With one model family, there is no cross-model family.
- **Accuracy** is correct / all problems, so an abstention counts as not correct. T3b also gives accuracy when answered.
- **Consistent computation error**: wrong, but the response's own working arrives at it.
- **Fabricated**: wrong, not derived from the working (none shown, or it differs from the last value the working computes), and passes a surface check: eigenvalues sum to trace(A) for eigenpairs, trace(D) == trace(A) for diagonalisation. Other topics report n/a.
- **Hint leak**: `verifier/leak.py`, run against the stored ground truth in `score.py` and against the verified Solver answer inside OURS. It cannot detect a hint that makes the answer obvious without stating it. That needs human rating.
- **Hint verified**: every declared claim is certified, and every number in the text is in the problem, in a certified claim, or is 0 or +-1. A hint with no computed values counts as verified, so read `verified_rate` next to `verified_with_certified_claim`.
- **Step F1** is for the class "wrong". Steps after the predicted first error are not scored, because every system predicts only the first error.
- **Abstain** (B2 only) means no answer and no hints reach the student. **Answer withheld** (OURS) means the final answer was still refuted after repair; the rest of the response is delivered.
- **D1 hints** are exploratory, made and scored in run 1 only. RQ3 is measured on D3.
- **No usable reply** means the system never produced usable JSON. It counts as not correct, because it is a failure to answer. API errors are different: they say nothing about a system, so those rows are left out, and the tables warn when that makes systems incomparable.
- **Tokens** are `total_tokens` as the provider reports them. Gemini counts thinking there and not in `completion_tokens`.

## Limits of the checks

- **Row-equivalence never certifies.** Every invertible n x n matrix is row-equivalent to every other, so a slip in reducing an invertible matrix cannot be seen that way. The step checker applies a stated row operation exactly, so D3 annotators must write it in `step_text`: see [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). A step that passes on row-equivalence alone is weakly checked. It is delivered, because nothing refutes it, but it is not counted as certified.
- **Intermediate determinant values are not checked** unless they carry arithmetic in `expr`. Only the last value is compared with det(A), so a wrong minor is located at the final value.
- **Runs are samples, not seeds.** The Gemini endpoint rejects a `seed` field, so a run can be reproduced from the local cache but not from the API.

## D3 (student data)

Copy `data/d3_template.csv` to `data/private/d3.csv`, which is git-ignored, and follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). `data/taxonomy.json` keeps its `categories` list empty until the authors fill it from real data.

Until D3 exists, `run.py --dataset d3` refuses to run, and T4, T4b and the D3 row of T1 stay blank.

The D3 test split cannot be annotated until the taxonomy is frozen: set `"frozen": true` in `data/taxonomy.json`, commit, and `git tag taxonomy-v1`. Until then, `tools/annotation_sheet.py --split test` refuses to make sheets, `d3.py` refuses to load test-split labels, and `run.py` refuses the D3 test split.
