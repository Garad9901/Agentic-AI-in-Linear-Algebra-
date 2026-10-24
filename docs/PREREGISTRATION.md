# Pre-registered Analysis Plan

**Project:** Design and Evaluation of a Verification-Guided Agentic AI Tutor for Linear Algebra Education (conference version)
**Plan version:** prereg-v1 — frozen 2026-10-24, before any experiment, data collection or prompt tuning
**Repository:** https://github.com/Garad9901/Agentic-AI-in-Linear-Algebra-

Any change after the freeze goes in `docs/DEVIATIONS.md` with date, reason and who approved it. Results are reported against this plan, including null and negative results.

---

## 1. Scope

This conference paper covers the error dataset and misconception taxonomy, the verified multi-agent tutor, and the benchmark against unverified baselines. **A controlled learning-gain study is out of scope** and is reserved for a later journal paper. This paper makes no claim about learning gains.

**Topics (core 6):** row reduction and rank (RREF), determinant, matrix inverse, eigenpairs, diagonalisation, least squares.

**System under test:** four agents.
- **Diagnostic Agent** (LLM) labels the misconception behind a failed step.
- **Hint Agent** (LLM) writes graded hints at levels 1–3.
- **Solver Agent** (LLM) produces worked solutions.
- **Verification Agent** (SymPy, no LLM) checks every mathematical claim and every student step against an exact certificate.

Only claims the Verification Agent certifies reach the student. Claims it cannot check are shown with an "unverified" flag.

## 2. Research questions

- **RQ1 (tutor correctness).** Does the Verification Agent reduce the share of tutor responses that contain a mathematically wrong claim, compared with unverified tutors?
- **RQ2 (first-error localisation).** Does certificate-based step checking locate the first incorrect step in real student solutions more accurately than an LLM-only detector?
- **RQ3 (hint quality).** Do graded hints from the verified system leak final answers less often, at equal or better rated pedagogical quality, than single-LLM hints?
- **RQ4 (coverage).** What share of tutor claims and student steps can the Verification Agent actually certify, by topic and matrix size?

## 3. Metrics

**Primary metrics:**

| RQ | Primary metric | Unit of analysis |
|---|---|---|
| RQ1 | Wrong-claim rate: share of responses with ≥1 claim that contradicts the ground-truth SymPy solution | Tutor response |
| RQ2 | Exact first-error accuracy: the predicted first wrong step equals the gold first wrong step | Erroneous student solution |
| RQ3 | Answer-leakage rate: the hint states the final answer, or a value that determines it in one step | Hint |
| RQ4 | Certified-claim coverage: certified claims ÷ all extracted claims | Claim |

**Secondary metrics:**
- Error / no-error detection precision, recall and F1 at solution level.
- Misconception-label macro-F1 against the gold taxonomy labels.
- Human pedagogical rubric score, 1–5 on 4 criteria.
- Latency per response and API calls per response.

**Fixed rule (no LLM judge for correctness):** correctness is decided only by comparing extracted claims against the precomputed ground truth. All systems must output claims in the same JSON schema. Two humans audit a random 10% of claim extractions for fidelity, and we report their agreement.

**Circularity guard:** the Verification Agent never grades itself. Ground truth comes from the generator's stored SymPy solution, which is computed before any tutor runs, and from the human gold labels.

## 4. Datasets

### 4.1 Certified problem bank (D1, synthetic)
- Generated with SymPy using a fixed seed of **20261024**.
- 6 topics × 3 sizes (3×3, 4×4, 5×5) × **15 problems = 270 problems**.
- Integer entries in [−9, 9]. Eigen and diagonalisation problems are built as P·D·P⁻¹ with unimodular P, so eigenvalues are exact integers and certificates stay exact.
- **Split, fixed now:** dev 20% (54 problems), test 80% (216 problems), stratified by topic and size. Prompts are tuned on dev only. Test is run once per configuration.
- Licence: released with the code under the repository licence (decided in Phase 3).

### 4.2 External check set (D2)
- LinAlg-Bench (Agarwal et al., 2026, arXiv 2605.16675).
- Used only if its released data licence permits reuse; the licence and version are recorded before use. Otherwise this set is dropped and the drop is logged as a deviation.

### 4.3 Real student solutions (D3)
- **Collection:** a new class session at SVPCET. Consenting undergraduate students solve **4 set problems each** on paper, drawn from the core 6 topics (dev and test problems differ).
- **Ethics:** written informed consent, plus Institutional Ethics Committee approval or written institutional permission, **before** collection. Students under 18 are excluded. Participation is voluntary, ungraded, and can be withdrawn.
- **Privacy:**
  - Scans and consent forms stay in a private store, never in this repository.
  - Solutions are transcribed step by step with names removed.
  - Only anonymised transcriptions are sent to any API. Check the provider's free-tier terms first: some free tiers may use inputs to improve their models, so the consent form must state which provider is used.
- **Target sample:** 60 students × 4 problems = 240 solutions.
- **Sample-size reasoning:** see `scripts/power_analysis.py`.
  - The primary RQ2 test is McNemar.
  - Assumed discordant proportions are p10 = 0.25 and p01 = 0.05, with power 0.80. These are assumptions, not results.
  - Under these assumptions we need 57 erroneous solutions at α = 0.05, and 81 under the Holm worst case (α/4).
  - Target: **≥100 erroneous solutions in the test split**. Assuming about 60% of solutions contain an error, 240 solutions gives about 144 erroneous solutions, of which about 100 fall in the 70% test split.
- **Stopping rule:** collection ends at 240 usable solutions or on **4 November 2026**, whichever comes first. If fewer than 100 test erroneous solutions result, RQ2 is reported as underpowered.
- **Exclusions:** blank or illegible solutions, and withdrawn consent. Exclusion counts are reported.
- **Split, fixed now:** dev 30% / test 70%, **split by student** (all of one student's solutions go to one side), seed 20261024.

### 4.4 Gold labels and misconception taxonomy
- Two annotators label each D3 solution independently: the first wrong step and its misconception label.
  - **Annotator 1:** a faculty member who has taught linear algebra.
  - **Annotator 2:** a trained teaching assistant or senior student who has passed the course.
  - Neither annotator builds or tunes the tutor.
  - Their names go in `docs/ANNOTATORS.md` before annotation starts. That file is committed and dated, so the record is public.
- The taxonomy is built by open coding on the **dev split only** and then frozen (tag `taxonomy-v1`) **before** the test split is annotated.
- We report Cohen's κ on the test split, for first-error step and for misconception label. Disagreements are resolved by discussion, and the pre-discussion κ is the one reported.

## 5. Systems and baselines

**LLM selection rule, fixed now:**
- Two instruction-tuned models from two different model families.
- Each is available on a free API tier on the run-freeze date and was released after **2025-04-24** (within 18 months).
- Exact model IDs, provider, and access date are written to `config/models.yaml` and committed (tag `models-v1`) **before** the first test-split run.
- Temperature 0.7 for sampled runs.

| ID | System | What it tests |
|---|---|---|
| S0 | **Full system**: Diagnostic + Hint + Solver + Verification (per-step) | Proposed |
| B1 | Single-LLM tutor (one prompt does everything) | Single-model baseline |
| B2 | Multi-agent tutor with an LLM verifier agent instead of SymPy | "More LLM roles, no independent check" |
| B3 | Single LLM with a self-verification prompt | Self-correction baseline |
| B4 | Zero-shot LLM first-error detector (RQ2 only) | LLM-only diagnosis |

**Ablations (one per component):**

| ID | Change from S0 |
|---|---|
| A1 | Remove the Verification Agent: LLM output reaches the student unchecked |
| A2 | Verify the final answer only, not each step |
| A3 | Remove the Diagnostic Agent: step checks only, hints written without a misconception label |
| A4 | Single-level hint instead of graded levels 1–3 |

## 6. Repetition, statistics, reporting

- **Seeds:** 3 sampled runs per configuration on D1-test and D3-test. This is the minimum allowed; we chose 3, not 5, because of free-tier rate limits. A run where the API fails is retried with the same seed and the retry is logged.
- **Reporting:** mean ± SD over the 3 runs, with the range, and a 95% CI. Confidence intervals come from a bootstrap with 10,000 resamples, clustered by student for D3 and by problem for D1.
- **Tests:**

| Comparison | Test | Effect size |
|---|---|---|
| Binary paired outcomes (RQ1, RQ2, RQ3 leakage) | McNemar, exact when discordant pairs < 25 | Matched-pairs odds ratio with 95% CI |
| Rubric scores (RQ3) | Wilcoxon signed-rank | Matched-pairs rank-biserial r |

- **Primary family:** S0 vs B1, S0 vs B2, S0 vs B3 on RQ1, and S0 vs B4 on RQ2. Holm–Bonferroni correction is applied across these 4 comparisons at α = 0.05. Secondary and ablation results are reported with uncorrected p-values and labelled exploratory.
- **What would refute each claim:** see the claim ledger in Section 8.

**Hint-quality rubric (RQ3):**
- A stratified sample of 120 hints (S0 vs B1, 60 each, balanced by topic) is rated by the same two annotators, blind to the system.
- Four criteria, each 1–5: mathematical correctness, targets the actual error, does not give away the answer, and clarity.
- We report κ (quadratic-weighted) per criterion.

## 7. Hardware and environment to record

- Machine: CPU model, RAM, OS. **Intel(R) Core(TM) Ultra 9 185H (16 cores, 22 threads); 32 GB RAM (31.4 GiB usable); Microsoft Windows 11 Home Single Language, version 26H2, 64-bit (OS build 10.0.26300.9550)**
- Python, SymPy, NumPy and SciPy versions, pinned in `requirements.txt` (no "latest").
- API provider, model ID, access date, region; per-call latency, token counts, and rate-limit errors.
- Total API calls, wall-clock time, and cost (expected ₹0 on free tiers; any paid call is logged).

**Planned call budget:** about 216 test problems × 3 seeds × 6 configurations × about 3 calls per response ≈ 11,700 calls on D1, plus D3. Check each provider's daily limit and schedule runs across days.

## 8. Claim ledger (all planned; nothing measured yet)

| # | Claim | Evidence file | Build tag | n | Runs | Test | What would refute it | Status |
|---|---|---|---|---|---|---|---|---|
| C1 | S0 has a lower wrong-claim rate than B1, B2 and B3 | results/rq1_*.csv | [NEEDED] | 216 problems | 3 | McNemar + Holm | Holm-adjusted p ≥ 0.05, or S0 rate ≥ baseline rate | planned |
| C2 | Certificate step-checking locates the first student error more accurately than an LLM-only detector | results/rq2_*.csv | [NEEDED] | ≥100 erroneous test solutions | 3 | McNemar + Holm | Holm-adjusted p ≥ 0.05, or S0 accuracy ≤ B4 | planned |
| C3 | S0 hints leak answers less often than B1, with rubric quality not lower | results/rq3_*.csv | [NEEDED] | 120 rated hints | 3 | McNemar; Wilcoxon | Leakage not lower; or rubric median lower with p < 0.05 | planned |
| C4 | The Verification Agent certifies ≥ X% of claims per topic and size | results/rq4_*.csv | [NEEDED] | all claims | 3 | descriptive, 95% CI | Reported as measured; no threshold claimed in advance | planned |
| C5 | Each component contributes (A1–A4 worse than S0 on their target metric) | results/ablation_*.csv | [NEEDED] | as C1–C3 | 3 | exploratory | Ablation equal to or better than S0 | planned |

## 9. Tables the paper will fill

| Table | Content |
|---|---|
| T1 | Dataset statistics: D1 by topic and size; D3 students, solutions, erroneous share, κ |
| T2 | Wrong-claim rate by system × topic × size (RQ1) |
| T3 | Coverage of certified claims and steps by topic × size (RQ4) |
| T4 | First-error accuracy, detection P/R/F1, misconception macro-F1 (RQ2) |
| T5 | Ablations A1–A4 |
| T6 | Hint leakage and rubric scores with κ (RQ3) |
| T7 | Latency, calls and cost per response |

## 10. Known threats, stated in advance

- The synthetic problems (D1) use small integer matrices, so they may be easier to certify than real coursework.
- D3 comes from one institution, which limits generalisation.
- Free-tier models can change silently, so model IDs and access dates are logged and runs are kept close together.
- Claim extraction to JSON may miss claims, which is why the 10% human audit is reported.
