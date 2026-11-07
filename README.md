# Design and Evaluation of a Verification-Guided Agentic AI Tutor for Linear Algebra Education

This repository holds the conference version of the project. It covers an error dataset and misconception taxonomy built from real student solutions, a multi-agent tutor in which a SymPy Verification Agent checks every mathematical claim and student step against an exact certificate, and a benchmark against unverified LLM baselines. A controlled learning-gain study is out of scope, and this work makes no claim about learning gains.

The analysis plan was frozen before any experiment: [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). Changes after the freeze are logged in [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

Student data is never stored in this repository.

License: MIT, see [LICENSE](LICENSE).

## Running the benchmark

The current code is at the repository root: `run.py`, `batch.py`, `score.py`, `systems.py`, `grader.py`, `extraction.py`, `agents/` and `verifier/`. See [docs/BENCHMARK.md](docs/BENCHMARK.md). Dataset sources and licences are in [data/SOURCES.md](data/SOURCES.md), and D3 annotators should follow [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md).

## Earlier code (added 1 Nov 2026, before any LLM run)

This code is kept for the record. It no longer produces results, except `latutor/stats.py`, which `score.py` uses.

| Path | What it is |
|---|---|
| `latutor/verifier.py` | Verification Agent: SymPy certificates for every claim type and every student step |
| `latutor/generator.py` | Certified problem bank D1 (270 problems, seed 20261024) |
| `latutor/grader.py` | Ground-truth grader (stored solution only; never calls the verifier) |
| `latutor/systems.py` | S0, B1–B4, A1–A4 |
| `latutor/llm.py`, `latutor/prompts.py` | OpenAI-compatible LLM backend; prompts-v1 |
| `latutor/stats.py`, `scripts/analyse.py` | McNemar, matched-pairs OR, Holm, Wilcoxon, cluster bootstrap; tables T2–T7 |
| `scripts/run_benchmark.py` | Runner with pre-registration gates for the test split |

How to run: [docs/RUNNING.md](docs/RUNNING.md). Open choices awaiting author approval: [docs/PROPOSED_DEVIATIONS.md](docs/PROPOSED_DEVIATIONS.md).
