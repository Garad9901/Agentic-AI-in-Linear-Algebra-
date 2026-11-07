## T1. Dataset statistics

| dataset | topic | size_3 | size_4 | size_5 | dev | test | total |
|---|---|---|---|---|---|---|---|
| D1 | rref_rank | 15 | 15 | 15 | 9 | 36 | 45 |
| D1 | determinant | 15 | 15 | 15 | 9 | 36 | 45 |
| D1 | inverse | 15 | 15 | 15 | 9 | 36 | 45 |
| D1 | eigenpairs | 15 | 15 | 15 | 9 | 36 | 45 |
| D1 | diagonalisation | 15 | 15 | 15 | 9 | 36 | 45 |
| D1 | least_squares | 15 | 15 | 15 | 9 | 36 | 45 |
| D2 | determinant | 50 | 50 | 50 | 30 | 120 | 150 |
| D2 | rref_rank | 20 | 20 | 20 | 12 | 48 | 60 |
| D3 | all |  |  |  |  |  |  |

- Least squares: size n means A is (n+1) x n.
- D2 is the 210 of 660 LinAlg-Bench problems that can be checked exactly. See data/SOURCES.md.
- D3 row is blank: d3.csv does not exist: no student data has been collected yet.

## T2. Accuracy by topic x system

| topic | B1 |
|---|---|
| rref_rank | 1.000 (2/2) |
| determinant | 1.000 (1/1) |
| inverse | 0.000 (0/1) |
| eigenpairs | 1.000 (1/1) |
| diagonalisation | 1.000 (1/1) |
| least_squares |  |
| all | 0.833 (5/6) |

- Accuracy is correct / all problems, so an abstention counts as not correct. Split: dev.
- No usable D1 results for ['B2', 'B3', 'B4', 'OURS'].
- Systems are NOT comparable here: API errors removed problems unevenly {'B1': 4, 'B2': 3, 'B3': 3, 'B4': 3, 'OURS': 3}. Rerun to fill them in.

## T3. Accuracy by size x system

| size | B1 |
|---|---|
| 3 | 1.000 (2/2) |
| 4 | 1.000 (3/3) |
| 5 | 0.000 (0/1) |
| all | 0.833 (5/6) |

- Accuracy is correct / all problems, so an abstention counts as not correct. Split: dev.
- No usable D1 results for ['B2', 'B3', 'B4', 'OURS'].
- Systems are NOT comparable here: API errors removed problems unevenly {'B1': 4, 'B2': 3, 'B3': 3, 'B4': 3, 'OURS': 3}. Rerun to fill them in.

## T2b. RQ1 wrong-claim rate

| system | topic | size | problems | runs | mean | sd | min_run | max_run | ci95_low | ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | all | all | 6 | 1 | 0.167 | 0.000 | 0.167 | 0.167 | 0.000 | 0.500 |
| B1 | determinant | 4 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | diagonalisation | 4 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | eigenpairs | 3 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | inverse | 5 | 1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B1 | rref_rank | 3 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 | rref_rank | 4 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- No usable D1 results for ['B2', 'B3', 'B4', 'OURS'].
- Systems are NOT comparable here: API errors removed problems unevenly {'B1': 4, 'B2': 3, 'B3': 3, 'B4': 3, 'OURS': 3}. Rerun to fill them in.
- Share of responses with at least 1 delivered claim that contradicts the ground truth, graded by grader.py and never by the verifier. Mean, SD and range are over runs; the 95% CI comes from a bootstrap of 10000 resamples clustered by problem.
- A response that abstains or gives no usable reply delivers no claims, so it cannot be wrong. Read this with the abstention rate in T3b.
- The human audit of 10% of claim extractions (PREREGISTRATION.md s3) has not been done.

## T3d. RQ4 certified-claim coverage (OURS)

| topic | size | responses | claims | certified_coverage | ci95_low | ci95_high | decidable_share | weakly_checked_share |
|---|---|---|---|---|---|---|---|---|

- Certified claims / all extracted claims in the delivered version, from the verifier. A row operation is certified only if applying its stated operation reproduces the next matrix exactly. Claims that pass on row-equivalence alone are weakly checked: they never count as certified, and are reported separately. Decidable share is (certified + refuted) / all.
- Blank if OURS has not been run. Coverage of student steps (prereg-v1 T3) needs D3.

## T3b. Wrong answers, fabricated answers and abstention

| system | n | api_errors_excluded | answered | wrong_delivered_rate | consistent_computation_error_rate | fabricated_rate | abstention_rate | no_parseable_reply | accuracy_when_answered |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 6 | 4 | 6 | 0.167 (1/6) | 0.167 (1/6) | 0.000 (0/2) | 0.000 (0/6) | 0 | 0.833 (5/6) |

- Consistent computation error: wrong, but the response's own working arrives at it. Over all problems.
- Fabricated: wrong, not derived from the working (none shown, or it differs from the last value the working computes), and passes the surface check. Over eigenpairs and diagonalisation only: eigenvalues must sum to trace(A), or trace(D) must equal trace(A).

## T3c. Fabricated rate by topic

| topic | B1 |
|---|---|
| rref_rank | n/a |
| determinant | n/a |
| inverse | n/a |
| eigenpairs | 0.000 (0/1) |
| diagonalisation | 0.000 (0/1) |
| least_squares | n/a |

- n/a: the topic has no surface check.

## T4. First wrong step, step F1, misconception macro-F1 (D3)

| system | solutions | first_wrong_step_accuracy | step_f1 | misconception_macro_f1 |
|---|---|---|---|---|
| B1 |  |  |  |  |
| B2 |  |  |  |  |
| B3 |  |  |  |  |
| B4 |  |  |  |  |
| OURS |  |  |  |  |
| DET |  |  |  |  |

- Blank: d3.csv does not exist: no student data has been collected yet. T4 needs real student solutions, which must not be invented.

## T4b. Cohen's kappa between the two D3 annotators

| measure | items | kappa |
|---|---|---|
| step_label |  |  |
| first_error_step |  |  |
| misconception |  |  |

- Blank: d3.csv does not exist: no student data has been collected yet.

## T5. Hint leak rate and hint-verified rate

| system | hints_delivered | hints_withheld | leak_rate | verified_rate | verified_with_certified_claim |
|---|---|---|---|---|---|
| B1 | 18 | 0 | 0.000 (0/18) | 0.889 (16/18) | 0.389 (7/18) |

- Both rates are over delivered hints, measured by the same checks for every system. Leaks are checked against the stored ground truth. B1, B2 and B3 are measured but never filtered. A hint with no computed values counts as verified, so the last column shows how many passed with at least one certified claim, i.e. checked mathematics rather than a vacuous pass.

## T6. Ablations (D1)

| configuration | n | accuracy | abstention_rate | wrong_delivered_rate | hint_leak_rate |
|---|---|---|---|---|---|
| OURS |  |  |  |  |  |
| A1 |  |  |  |  |  |
| A2 |  |  |  |  |  |
| A3 |  |  |  |  |  |
| A4 |  |  |  |  |  |

- A blank row means that configuration has not been run on this split.
- A1 is identical to B4 and A3 to OURS on D1 by construction: the Diagnostic Agent needs student work, which only D3 has.

## T7. Mean cost per problem

| system | problems | mean_llm_calls | mean_input_tokens | mean_output_tokens | mean_seconds | parse_failures | rate_limit_retries |
|---|---|---|---|---|---|---|---|
| B1 | 6 | 1.00 | 710 | 7030 | 15.29 | 0 | 2 |

- Calls, tokens and latency are those of the original API call, so cached reruns report the same cost. Seconds are the latency of the successful API call plus local compute. Pacing and rate-limit waits are excluded.
- Batch runs record no latency, so T7 seconds come only from Groq synchronous runs. They are blank for Gemini Batch runs.
- Output tokens are all tokens except the prompt, so they include thinking.

## T7b. Truncation rate by system and size

| system | size | calls | truncated | truncation_rate |
|---|---|---|---|---|
| B1 | 3 | 2 | 0 | 0.000 (0/2) |
| B1 | 4 | 3 | 0 | 0.000 (0/3) |
| B1 | 5 | 1 | 0 | 0.000 (0/1) |
| B1 | all | 6 | 0 | 0.000 (0/6) |

- A call is truncated if its finish_reason is length. Flag threshold: 1%.
- No cell is above the threshold.

## T8. McNemar tests with Holm correction (primary family)

| rq | comparison | pairing | n_pairs | s0_only | baseline_only | p | method | odds_ratio | or_ci_low | or_ci_high | holm_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RQ1 | OURS vs B1 | majority |  |  |  |  |  |  |  |  |  |
| RQ1 | OURS vs B2 | majority |  |  |  |  |  |  |  |  |  |
| RQ1 | OURS vs B3 | majority |  |  |  |  |  |  |  |  |  |
| RQ2 | OURS vs DET | majority |  |  |  |  |  |  |  |  |  |

- Primary pairing is per-item majority: an event in at least 2 of 3 runs. It stays blank until both systems have all 3 runs. Per-run rows are the sensitivity analysis and take no correction.
- s0_only and baseline_only are the discordant pairs. Odds ratio is s0_only / baseline_only, with an exact 95% CI, so below 1 favours OURS.
- holm_p is blank: Holm runs across all 4 registered comparisons, and at least one has no majority p-value yet. RQ2 needs D3.
