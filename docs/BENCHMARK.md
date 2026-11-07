# Running the benchmark

Python 3.10.9, CPU only. The plan is [PREREGISTRATION.md](PREREGISTRATION.md), and every change to it is in [DEVIATIONS.md](DEVIATIONS.md).

No number in any table comes from anywhere but a real model reply, a dataset file or an exact computation. There is no mock model. A cell that cannot be computed stays blank, with the reason under the table.

## Setup (PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env          # provider and model; the key comes from GEMINI_API_KEY
python data/gen_d1.py                # 270 problems, seed 20261024
python data/gen_d2.py                # LinAlg-Bench, pinned revision
python -m pytest -q
```

`.env` is git-ignored. The runner refuses any model that is not listed in `config/models.yaml` (tag `models-v1`).

## Dev split

```powershell
python run.py --system OURS --dataset d1 --split dev --run 1 --limit 20
python score.py --split dev
python scripts/cost_probe.py
```

`--system` takes B1, B2, B3, B4, OURS, A1 to A4, and DET (D3 only). `--dataset` takes d1, d2 or d3. `--run` takes 1, 2 or 3.

`--limit` takes problems round-robin across topic and size: 5 problems cover 5 topics and all 3 sizes, and 18 cover every cell once. `--workers` runs several problems at once. Requests stay within `LLM_RPM`, and a single call can take minutes on larger matrices, so a sequential run is slow.

Tune `agents/prompts.py` on the dev split only.

## Test split

`run.py` refuses the test split until a file named `FROZEN` exists. **Do not create it yet.** First resolve every row marked pending in [DEVIATIONS.md](DEVIATIONS.md). In particular, `score.py` does not yet compute the pre-registered wrong-claim rate, RQ4 coverage or statistics.

Every response is cached under `cache/` by (model, prompt, run), so rerunning a stopped command resumes at no API cost, and each run still gets a fresh response. Every call is logged to `logs/calls.jsonl`, and every completed command to `results/RUN_INFO.md`.

## Layout

| Path | What it is |
|---|---|
| `llm.py` | `call_llm(role, prompt)`: temperature 0.7, disk cache, backoff, logging |
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
| `run.py`, `score.py` | Runner and tables T1 to T7 |
| `scripts/cost_probe.py` | Cost summary and extrapolation |

`latutor/` and the older scripts are kept for the record, but they no longer produce results.

## Systems

| ID | What it does on D1 | What reaches the student |
|---|---|---|
| B1 | One prompt solves and writes hints | Everything, unchecked |
| B2 | Solver, then an LLM verifier judges the solution and each hint | What the LLM approves. Otherwise it abstains or withholds |
| B3 | B1, then a self-check prompt | Everything, unchecked |
| B4 | Solver, then a Hint call per level | Everything, unchecked |
| OURS | Solver with up to 2 retries, then checked hints with up to 2 regenerations each | Verified answers and verified hints. Otherwise it abstains or withholds |
| A1 | OURS without verification: the same configuration as B4 | Everything, unchecked |
| A2 | OURS verifying the final answer only | As OURS |
| A3 | OURS without the Diagnostic Agent: the same as OURS on D1 and D2 | As OURS |
| A4 | OURS with one hint, not three levels | As OURS |
| DET | Zero-shot first-error detector, D3 only | No answer or hints |

## Definitions

- **Accuracy** is correct / all problems, so an abstention counts as not correct. T3b also gives accuracy when answered.
- **Consistent computation error**: wrong, but the response's own working arrives at it.
- **Fabricated**: wrong, not derived from the working (none shown, or it differs from the last value the working computes), and passes a surface check: eigenvalues sum to trace(A) for eigenpairs, trace(D) == trace(A) for diagonalisation. Other topics report n/a.
- **Hint leak**: `verifier/leak.py`, run against the stored ground truth in `score.py` and against the verified Solver answer inside OURS. It cannot detect a hint that makes the answer obvious without stating it. That needs human rating.
- **Hint verified**: every declared claim is certified, and every number in the text is in the problem, in a certified claim, or is 0 or +-1. A hint with no computed values counts as verified, so read `verified_rate` next to `verified_with_certified_claim`.
- **Step F1** is for the class "wrong". Steps after the predicted first error are not scored, because every system predicts only the first error.
- **Abstain** means no answer and no hints reach the student.
- **No usable reply** means the system never produced usable JSON. It counts as not correct, because it is a failure to answer. API errors are different: they say nothing about a system, so those rows are left out, and the tables warn when that makes systems incomparable.
- **Tokens** are `total_tokens` as the provider reports them. Gemini counts thinking there and not in `completion_tokens`.

## Limits of the checks

- **Row-equivalence is weak for full-rank square matrices.** Every invertible n x n matrix is row-equivalent to every other, so a slip in reducing an invertible matrix cannot be seen that way. The step checker applies a stated row operation exactly, so D3 annotators must write it in `step_text`: see [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). Steps that pass only on the weak check are counted in the `weak_steps` column.
- **Intermediate determinant values are not checked** unless they carry arithmetic in `expr`. Only the last value is compared with det(A), so a wrong minor is located at the final value.
- **Runs are samples, not seeds.** The Gemini endpoint rejects a `seed` field, so a run can be reproduced from the local cache but not from the API.

## D3 (student data)

Copy `data/d3_template.csv` to `data/private/d3.csv`, which is git-ignored, and follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). `data/taxonomy.json` keeps its `categories` list empty until the authors fill it from real data.

Until D3 exists, `run.py --dataset d3` refuses to run, and T4, T4b and the D3 row of T1 stay blank.
