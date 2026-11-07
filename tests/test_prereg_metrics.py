"""RQ1, RQ4, McNemar, Holm, providers, truncation and batch mode.

Every expected number here is worked out by hand in the comment beside it.
Model replies are scripted stand-ins for control flow, and nothing here reaches results/.
"""
import io
import json
import sys
import urllib.error
from collections import Counter

import pandas as pd
import pytest

import batch
import grader
import llm
import run
import score
from extraction import extract
from latutor.stats import holm, mcnemar
from scripts import cost_probe
from tools import annotation_sheet, claim_audit
import d3
from verifier import check_claim, step_statuses, verify_hint

DET = {"id": "d", "topic": "determinant", "size": 2, "A": [[1, 2], [3, 4]], "ground_truth": {"determinant": -2}}
RREF = {"id": "r", "topic": "rref_rank", "size": 2, "A": [[1, 2], [2, 4]],
        "ground_truth": {"rref": [[1, 2], [0, 0]], "rank": 1, "pivots": [1]}}
INV = {"id": "i", "topic": "inverse", "size": 2, "A": [[1, 2], [3, 4]],
       "ground_truth": {"inverse": [[-2, 1], ["3/2", "-1/2"]]}}
EIG = {"id": "e", "topic": "eigenpairs", "size": 2, "A": [[2, 1], [1, 2]], "ground_truth": {
    "eigenvalues": [{"value": 1, "multiplicity": 1}, {"value": 3, "multiplicity": 1}],
    "eigenspaces": [{"eigenvalue": 1, "basis": [[-1, 1]]}, {"eigenvalue": 3, "basis": [[1, 1]]}]}}
LSQ = {"id": "l", "topic": "least_squares", "size": 2, "A": [[1, 0], [0, 1], [1, 1]], "b": [1, 1, 0],
       "ground_truth": {"x": ["1/3", "1/3"]}}


# ---- extraction
def test_each_step_is_one_claim_and_text_is_a_statement():
    trace = [{"text": "plan the work"}, {"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}, {"value": -2}]
    claims = extract(DET, trace, {"determinant": -2}, [])
    assert [c["where"] for c in claims] == ["step", "step", "step", "final"]
    assert [c["claim"]["type"] for c in claims] == ["statement", "intermediate_matrix", "determinant", "determinant"]


def test_hint_claims_are_extracted_and_withheld_answers_are_not_invented():
    hints = [{"text": "t", "claims": [{"type": "rank", "value": 1}, {"type": "trace", "value": 5}]}]
    claims = extract(RREF, [], {"rref": [[1, 2], [0, 0]], "rank": 1}, hints)
    assert [c["where"] for c in claims] == ["final", "final", "hint", "hint"]


def test_eigenpairs_give_a_list_claim_and_one_claim_per_pair():
    answer = {"eigenpairs": [{"eigenvalue": 1, "eigenvector": [1, -1]}, {"eigenvalue": 3, "eigenvector": [1, 1]}]}
    assert [c["claim"]["type"] for c in extract(EIG, None, answer, [])] == ["eigenvalues", "eigenvector", "eigenvector"]


# ---- grading single claims
@pytest.mark.parametrize("problem,claim,want", [
    (DET, {"type": "determinant", "value": -2}, "correct"),
    (DET, {"type": "determinant", "value": 2}, "wrong"),
    (RREF, {"type": "rank", "value": 1}, "correct"),
    (RREF, {"type": "rref", "matrix": [[1, 3], [0, 0]]}, "wrong"),
    (INV, {"type": "inverse", "matrix": [[1, 0], [0, 1]]}, "wrong"),
    (EIG, {"type": "eigenvalue", "value": 3}, "correct"),
    (EIG, {"type": "eigenvalue", "value": 2}, "wrong"),
    (EIG, {"type": "eigenvalues", "values": [1, 3]}, "correct"),
    (EIG, {"type": "eigenvalues", "values": [3, 3]}, "wrong"),
    (EIG, {"type": "eigenvector", "eigenvalue": 3, "vector": [2, 2]}, "correct"),
    (EIG, {"type": "eigenvector", "eigenvalue": 3, "vector": [1, -1]}, "wrong"),
    (EIG, {"type": "charpoly", "coefficients": [1, -4, 3]}, "correct"),  # (x-1)(x-3) = x^2 - 4x + 3
    (EIG, {"type": "charpoly", "coefficients": [1, -4, 4]}, "wrong"),
    (LSQ, {"type": "AtA", "matrix": [[2, 1], [1, 2]]}, "correct"),
    (LSQ, {"type": "Atb", "vector": [1, 1]}, "correct"),
    (LSQ, {"type": "normal_vector", "vector": ["1/3", "1/3"]}, "correct"),
    (LSQ, {"type": "normal_vector", "vector": [9, 9]}, "wrong"),
    (LSQ, {"type": "lsq_solution", "vector": [1, 1]}, "wrong"),
    (DET, {"type": "trace", "value": 5}, "correct"),
    (DET, {"type": "minor", "row": 1, "col": 1, "value": 4}, "correct"),
    (DET, {"type": "cofactor", "row": 1, "col": 2, "value": 3}, "wrong"),  # the cofactor is -3
    (DET, {"type": "arithmetic", "expr": "3*4", "value": 13}, "wrong"),
    (DET, {"type": "statement", "text": "so"}, "unscorable"),
    (DET, {"type": "vibes"}, "unscorable"),
    (DET, {"type": "determinant"}, "unscorable"),
    (DET, "not a claim", "unscorable"),
])
def test_grade_claim(problem, claim, want):
    assert grader.grade_claim(problem, claim) == want


SHIFTED = {"type": "row_op", "start": [[1, 1], [1, 1]], "op": "R2 <- R2 - R1", "matrix": [[1, 1], [0, 0]]}  # A - I


def test_a_row_op_on_a_minus_lambda_i_is_a_valid_claim():
    assert grader.grade_claim(EIG, SHIFTED) == "correct"
    assert check_claim(EIG, SHIFTED) == "certified"


def test_a_shift_by_a_non_eigenvalue_stays_refused():
    bad = {"type": "row_op", "start": [[1, 0], [0, 1]], "op": "R1 <-> R2", "matrix": [[0, 1], [1, 0]]}
    singular = {**EIG, "A": [[1, 2], [2, 4]], "ground_truth": {"eigenvalues": [{"value": 0, "multiplicity": 1},
                                                                               {"value": 5, "multiplicity": 1}],
                                                               "eigenspaces": []}}
    assert grader.grade_claim(singular, bad) == "wrong"
    assert check_claim(singular, bad) == "refuted"


def test_arithmetic_slip_on_a_full_rank_matrix_counts_as_a_wrong_claim():
    slip = {"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}  # the true row is [0, -2]
    claim = extract(DET, [slip], None, [])[0]["claim"]
    assert grader.grade_claim(DET, claim) == "wrong"
    good = {"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}
    assert grader.grade_claim(DET, extract(DET, [good], None, [])[0]["claim"]) == "correct"


def test_each_step_is_judged_against_the_step_before_it():
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}, {"op": "R2 <- (-1/2)R2", "matrix": [[1, 2], [0, 1]]}]
    grades = [grader.grade_claim(DET, c["claim"]) for c in extract(DET, trace, None, [])]
    assert grades == ["correct", "correct"]


def test_grader_is_independent_of_the_verifier(monkeypatch):
    import verifier.certificates as certificates
    monkeypatch.setattr(certificates, "verify", lambda *a: pytest.fail("the grader must not call the verifier"))
    assert grader.grade_claim(DET, {"type": "determinant", "value": -2}) == "correct"


# ---- RQ1 for one response
RIGHT = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}, {"value": -2}]


def test_rq1_one_wrong_claim_makes_the_response_wrong():
    # One right step and a wrong final answer: 2 claims, 1 wrong. The event is any wrong claim.
    stats = score.claim_stats(DET, True, [{"matrix": [[1, 2], [0, -2]]}], {"determinant": 9}, [])
    assert stats["n_claims"] == 2 and stats["n_wrong_claims"] == 1 and stats["wrong_claim"] == 1


def test_rq1_a_wrong_step_counts_even_when_the_answer_is_right():
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}, {"value": -2}]
    stats = score.claim_stats(DET, True, trace, {"determinant": -2}, [])
    assert stats["wrong_claim"] == 1 and stats["n_wrong_claims"] == 1


def test_rq1_a_wrong_hint_claim_counts():
    hints = [{"text": "t", "claims": [{"type": "determinant", "value": 7}]}]
    stats = score.claim_stats(DET, True, RIGHT, {"determinant": -2}, hints)
    assert stats["wrong_claim"] == 1


def test_rq1_abstention_delivers_no_claims():
    stats = score.claim_stats(DET, False, RIGHT, {"determinant": 9}, [])
    assert stats["n_claims"] == 0 and stats["wrong_claim"] == 0


def test_rq1_rate_by_hand():
    # 4 problems, run 1: 1 wrong of 4 = 0.25. Run 2: 3 wrong of 4 = 0.75.
    # Mean = 0.5. SD = sqrt(((.25-.5)^2 + (.75-.5)^2) / 1) = sqrt(0.125) = 0.354.
    rows = [{"run": 1, "problem_id": f"p{i}", "wrong_claim": int(i == 0)} for i in range(4)]
    rows += [{"run": 2, "problem_id": f"p{i}", "wrong_claim": int(i != 0)} for i in range(4)]
    row = score.rate_row(pd.DataFrame(rows), "wrong_claim", bootstrap=200)
    assert (row["problems"], row["runs"], row["mean"], row["sd"]) == (4, 2, "0.500", "0.354")
    assert (row["min_run"], row["max_run"]) == ("0.250", "0.750")


# ---- RQ4
def test_step_statuses_cover_every_step_and_do_not_stop():
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]},  # wrong arithmetic
             {"text": "think"},                                        # nothing to check
             {"matrix": [[1, 2], [0, 1]]},                             # invertible, no op: weak
             {"value": -2}]                                            # right
    assert step_statuses("determinant", DET, trace) == ["refuted", "uncheckable", "weak", "certified"]


def test_rq4_coverage_by_hand():
    # Claims: step 1 weak, step 2 certified, step 3 statement, final determinant certified: 4 claims.
    # Certified = 2, because weak never counts -> coverage 2/4. Refuted 0 -> decidable 2/4. Weak 1/4.
    trace = [{"matrix": [[1, 2], [0, -2]]}, {"value": -2}, {"text": "done"}]
    stats = score.claim_stats(DET, True, trace, {"determinant": -2}, [])
    assert (stats["n_claims"], stats["n_certified"], stats["n_weak"], stats["n_refuted"]) == (4, 2, 1, 0)
    frame = pd.DataFrame([{"problem_id": "d", **stats}])
    row = score.coverage_row(frame, bootstrap=100)
    assert row["certified_coverage"] == "0.500" and row["decidable_share"] == "0.500"
    assert row["weakly_checked_share"] == "0.250"


# ---- decision of 2026-11-07: row-equivalence alone never certifies
SWAP = {"op": "R1 <-> R2", "matrix": [[2, 4], [1, 2]]}


def test_a_stated_operation_reproducing_the_next_matrix_is_certified():
    assert step_statuses("rref_rank", RREF, [SWAP]) == ["certified"]
    assert grader.grade_claim(RREF, extract(RREF, [SWAP], None, [])[0]["claim"]) == "correct"


def test_no_stated_operation_is_weak_even_when_rank_deficient():
    step = {"matrix": [[1, 2], [0, 0]]}  # rank 1, so row-equivalence means something, but it still never certifies
    assert step_statuses("rref_rank", RREF, [step]) == ["weak"]
    assert grader.grade_claim(RREF, extract(RREF, [step], None, [])[0]["claim"]) == "weak"


def test_an_unreadable_operation_is_weak_not_certified():
    step = {"op": "eliminate the second row", "matrix": [[1, 2], [0, 0]]}
    assert step_statuses("rref_rank", RREF, [step]) == ["weak"]


def test_a_wrong_stated_operation_is_refuted_even_if_row_equivalent():
    step = {"op": "R2 <- R2 + R1", "matrix": [[1, 2], [0, 0]]}  # row-equivalent to A, but R2 + R1 = [3, 6]
    assert step_statuses("rref_rank", RREF, [step]) == ["refuted"]
    assert grader.grade_claim(RREF, extract(RREF, [step], None, [])[0]["claim"]) == "wrong"


def test_restating_the_previous_matrix_is_exact_not_weak():
    assert step_statuses("rref_rank", RREF, [{"matrix": [[1, 2], [2, 4]]}]) == ["certified"]


def test_a_lambda_i_reductions_follow_the_same_rule():
    # A - 1I = [[1, 1], [1, 1]]. R2 <- R2 - R1 gives [[1, 1], [0, 0]].
    good = {"eigenvalue": 1, "op": "R2 <- R2 - R1", "matrix": [[1, 1], [0, 0]]}
    assert step_statuses("eigenpairs", EIG, [good]) == ["certified"]
    assert grader.grade_claim(EIG, extract(EIG, [good], None, [])[0]["claim"]) == "correct"
    no_op = {"eigenvalue": 1, "matrix": [[1, 1], [0, 0]]}
    assert step_statuses("eigenpairs", EIG, [no_op]) == ["weak"]
    assert grader.grade_claim(EIG, extract(EIG, [no_op], None, [])[0]["claim"]) == "weak"
    wrong_op = {"eigenvalue": 1, "op": "R2 <- R2 + R1", "matrix": [[1, 1], [0, 0]]}
    assert step_statuses("eigenpairs", EIG, [wrong_op]) == ["refuted"]


def test_each_eigenvalue_keeps_its_own_chain():
    trace = [{"eigenvalue": 1, "op": "R2 <- R2 - R1", "matrix": [[1, 1], [0, 0]]},
             {"eigenvalue": 3, "op": "R2 <- R2 + R1", "matrix": [[-1, 1], [0, 0]]}]  # A - 3I = [[-1, 1], [1, -1]]
    assert step_statuses("eigenpairs", EIG, trace) == ["certified", "certified"]
    assert [grader.grade_claim(EIG, c["claim"]) for c in extract(EIG, trace, None, [])] == ["correct", "correct"]


def test_row_equivalent_hint_claims_are_weak_and_fail_hint_verification():
    claim = {"type": "row_equivalent", "matrix": [[1, 0], [0, 1]]}
    problem = {"A": [[1, 2], [3, 4]]}
    assert check_claim(problem, claim) == "weak"
    assert not verify_hint(problem, "No numbers here.", [claim]).verified
    assert check_claim(RREF, {"type": "row_equivalent", "matrix": [[1, 3], [0, 0]]}) == "refuted"
    assert grader.grade_claim(DET, claim) == "weak"


def test_a_weak_claim_is_not_a_wrong_claim():
    stats = score.claim_stats(DET, True, [{"matrix": [[1, 2], [0, -3]]}], {"determinant": -2}, [])
    assert stats["wrong_claim"] == 0 and stats["n_weakly_graded"] == 1 and stats["n_certified"] == 1


def test_final_rref_is_still_certified_because_rref_is_unique():
    stats = score.claim_stats(RREF, True, [], {"rref": [[1, 2], [0, 0]], "rank": 1}, [])
    assert stats["n_certified"] == 2 and stats["n_weak"] == 0


def test_rq4_refuted_is_decidable_but_not_certified():
    # Step refuted, final refuted: 0 of 2 certified, 2 of 2 decidable.
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}]
    stats = score.claim_stats(DET, True, trace, {"determinant": 9}, [])
    frame = pd.DataFrame([{"problem_id": "d", **stats}])
    row = score.coverage_row(frame, bootstrap=100)
    assert (row["certified_coverage"], row["decidable_share"]) == ("0.000", "1.000")


def test_rq4_pools_claims_not_responses():
    # Response A: 1 of 1 certified. Response B: 0 of 3. Pooled 1/4, not the mean of shares (0.5).
    frame = pd.DataFrame([{"problem_id": "a", "n_claims": 1, "n_certified": 1, "n_refuted": 0, "n_weak": 0},
                          {"problem_id": "b", "n_claims": 3, "n_certified": 0, "n_refuted": 0, "n_weak": 0}])
    assert score.coverage_row(frame, bootstrap=100)["certified_coverage"] == "0.250"


def test_coverage_is_blank_without_ours():
    assert score.t3d_coverage(None, 100).empty


# ---- McNemar and Holm
def test_mcnemar_exact_by_hand():
    # b = 1, c = 9, n = 10 < 25: exact binomial. p = 2 x (C(10,0) + C(10,1)) / 2^10 = 22/1024.
    x = [1] + [0] * 9
    y = [0] + [1] * 9
    result = mcnemar(x, y)
    assert result["method"] == "exact binomial"
    assert result["p"] == pytest.approx(22 / 1024)
    assert result["odds_ratio"] == pytest.approx(1 / 9)


def test_mcnemar_chi_square_by_hand():
    # b = 20, c = 5, n = 25, so not exact. chi2 = (|20-5| - 1)^2 / 25 = 14^2 / 25 = 7.84, df 1, p = 0.00511.
    x = [1] * 20 + [0] * 5
    y = [0] * 20 + [1] * 5
    result = mcnemar(x, y)
    assert result["method"].startswith("chi-square")
    assert result["p"] == pytest.approx(0.00511, abs=1e-5)


def test_mcnemar_ignores_concordant_pairs_and_handles_none_discordant():
    assert mcnemar([1, 0, 1], [1, 0, 1])["p"] == 1.0
    assert mcnemar([1, 1, 0, 0], [1, 0, 0, 1])["n_pairs"] == 4


def test_holm_by_hand():
    # Sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04.
    # Holm keeps the running maximum, so the last becomes 0.06.
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert out == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})


def test_there_is_no_across_8_table_any_more():
    assert not hasattr(score, "t8b_all_comparisons")


def test_only_the_groq_model_is_scored(tmp_path):
    for tag in ("groq__openai-gpt-oss-120b", "gemini__gemini-3.8-flash"):
        (tmp_path / f"B1_d1_dev_run1__{tag}.csv").write_text("problem_id\n", encoding="utf-8")
    assert score.model_tags("dev", tmp_path) == ["groq__openai-gpt-oss-120b"]


def test_holm_needs_all_four_registered_comparisons():
    three = {"a": {"p": 0.01}, "b": {"p": 0.02}, "c": {"p": 0.03}, "d": None}
    assert score.holm_family(three) == {}
    four = {"a": {"p": 0.01}, "b": {"p": 0.02}, "c": {"p": 0.03}, "d": {"p": 0.04}}
    assert score.holm_family(four)["a"] == "0.0400"  # 0.01 x 4


def _runs(values):
    return pd.DataFrame([{"run": r, "problem_id": f"p{i}", "m": v} for r, col in values.items() for i, v in enumerate(col)])


def test_majority_rule_by_hand():
    # p0: OURS event in 1 of 3 (no), baseline in 2 of 3 (yes): discordant, baseline only.
    # p1: OURS event in 2 of 3 (yes), baseline in 3 of 3 (yes): concordant.
    a = _runs({1: [1, 1], 2: [0, 1], 3: [0, 0]})
    b = _runs({1: [1, 1], 2: [1, 1], 3: [0, 1]})
    majority, per_run = score.paired_tests(a, b, "m")
    assert (majority["b"], majority["c"], majority["n_pairs"]) == (0, 1, 2)
    assert sorted(per_run) == [1, 2, 3]


def test_majority_is_blank_until_three_runs_exist():
    a, b = _runs({1: [1, 0]}), _runs({1: [0, 0]})
    majority, per_run = score.paired_tests(a, b, "m")
    assert majority is None and per_run[1]["b"] == 1


def test_t8_without_d3_leaves_rq2_and_holm_blank():
    a, b = _runs({1: [1, 0]}), _runs({1: [0, 1]})
    a["wrong_claim"] = a.pop("m")
    b["wrong_claim"] = b.pop("m")
    table, notes = score.t8_tests({"OURS": a, "B1": b}, {})
    assert (table["holm_p"] == "").all() and any("needs D3" in n for n in notes)
    assert list(table[table["pairing"] == "run1"]["comparison"]) == ["OURS vs B1"]


def test_rq2_event_is_a_wrong_first_error_on_erroneous_solutions():
    frame = pd.DataFrame([{"run": 1, "problem_id": "s1", "status": "answered", "pred_first_wrong_step": 2},
                          {"run": 1, "problem_id": "s2", "status": "answered", "pred_first_wrong_step": None},
                          {"run": 1, "problem_id": "s3", "status": "answered", "pred_first_wrong_step": None}])
    gold = {"s1": {"first_error_step": 2}, "s2": {"first_error_step": 1}, "s3": {"first_error_step": None}}
    out = score.rq2_frames({"OURS": frame}, gold)["OURS"]
    assert out["problem_id"].tolist() == ["s1", "s2"]  # s3 has no error, so it is not in RQ2
    assert out["first_error_wrong"].tolist() == [0, 1]


# ---- providers and result files
def test_cache_key_uses_provider_model_run_and_prompt():
    base = llm.cache_key("groq", "m", 1, "p")
    assert base != llm.cache_key("gemini", "m", 1, "p")
    assert base != llm.cache_key("groq", "n", 1, "p")
    assert base != llm.cache_key("groq", "m", 2, "p")
    assert base != llm.cache_key("groq", "m", 1, "q")
    assert base == llm.cache_key("groq", "m", 1, "p")


def test_models_come_from_models_v1():
    assert run.model_for("groq") == "openai/gpt-oss-120b"
    assert run.model_for("gemini") == "gemini-3.8-flash"


def test_result_filename_has_provider_and_model_and_parses_back():
    path = run.result_path("OURS", "d1", "dev", 2, "groq", "openai/gpt-oss-120b")
    assert path.name == "OURS_d1_dev_run2__groq__openai-gpt-oss-120b.csv"
    assert score.parse_name(path, "dev") == ("OURS", "d1", "groq__openai-gpt-oss-120b")
    assert score.parse_name(path, "test") is None


def test_batch_needs_gemini(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--provider", "groq", "--batch", "--system", "B1", "--dataset", "d1", "--split", "dev"])
    with pytest.raises(SystemExit, match="gemini only"):
        run.main()


def test_usage_splits_input_and_output_and_counts_truncation():
    calls = [{"cached": False, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 90, "finish_reason": "length",
              "rate_limit_errors": 2, "latency_s": 1.0, "timeouts": 0},
             {"cached": True, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": None, "finish_reason": "stop",
              "rate_limit_errors": 0, "latency_s": 1.0, "timeouts": 0}]
    usage = run._usage(calls)
    assert (usage["input_tokens"], usage["output_tokens"]) == (20, 85)  # (90 - 10) + (15 - 10)
    assert (usage["live_input_tokens"], usage["live_output_tokens"]) == (10, 80)
    assert usage["truncated_calls"] == 1 and usage["rate_limit_errors"] == 2


def test_truncation_table_flags_a_high_rate():
    f = pd.DataFrame([{"size": 3, "llm_calls": 100, "truncated_calls": 0}, {"size": 5, "llm_calls": 50, "truncated_calls": 5}])
    table, notes = score.t7b_truncation({"B1": f}, ["B1"])
    assert table[table["size"] == 5]["truncation_rate"].iloc[0] == "0.100 (5/50)"
    assert any(n.startswith("FLAG") and "B1 size 5" in n for n in notes)
    clean = pd.DataFrame([{"size": 5, "llm_calls": 50, "truncated_calls": 0}])
    assert not any(n.startswith("FLAG") for n in score.t7b_truncation({"B1": clean}, ["B1"])[1])


def test_t7_seconds_come_only_from_synchronous_runs():
    f = pd.DataFrame([{"mode": "batch", "llm_calls": 1, "input_tokens": 1, "output_tokens": 1, "llm_seconds": 0,
                       "local_seconds": 0, "parse_failures": 0, "rate_limit_errors": 0}])
    assert score.t7_cost({"B1": f}, ["B1"]).iloc[0]["mean_seconds"] == ""


def test_score_says_where_seconds_come_from(tmp_path, monkeypatch):
    tables = {t[0]: t for t in score.build("dev", None, tmp_path, bootstrap=100)}
    assert any("only from Groq synchronous runs" in n for n in tables["T7"][3])


def test_a_reply_survives_a_file_briefly_held_by_windows(tmp_path, monkeypatch):
    real, failures = llm.os.replace, []

    def held(src, dst):
        if len(failures) < 3:
            failures.append(1)
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(llm.os, "replace", held)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm.atomic_write(tmp_path / "x.json", "saved")
    assert (tmp_path / "x.json").read_text(encoding="utf-8") == "saved" and len(failures) == 3


def test_a_file_held_for_good_still_raises(tmp_path, monkeypatch):
    def held(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(llm.os, "replace", held)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        llm.atomic_write(tmp_path / "x.json", "saved")


def test_error_messages_carry_no_organisation_id(tmp_path, monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError("u", 413, "big", {}, io.BytesIO(b"too large in organization `org_01abc9` on tokens"))

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(llm, "_next_slot", 0.0)
    monkeypatch.setenv("LLM_RPM", "100000")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.LLMError) as e:
        llm.call_llm("solver", "p")
    assert "org_01abc9" not in str(e.value) and "org_***" in str(e.value)


def test_score_skips_a_system_with_no_usable_rows(tmp_path):
    cols = dict(run=1, mode="sync", topic="determinant", size=3, status="api_error", answer_json="null", trace_json="null",
                hints_json="[]", self_check="", pred_first_wrong_step="", pred_misconception="", error="x", llm_calls=0)
    pd.DataFrame([{"problem_id": gen_id, **cols} for gen_id in ("a", "b")]).to_csv(
        tmp_path / "B2_d1_dev_run1__groq__m.csv", index=False)
    tables = {t[0]: t for t in score.build("dev", "groq__m", tmp_path, bootstrap=100)}
    assert tables["T2b"][2].empty
    assert any("B2" in n for n in tables["T2"][3])


def test_cost_summary_handles_a_system_with_no_usable_rows():
    frame = pd.DataFrame([{"status": "api_error", "llm_calls": 0}])
    assert cost_probe.summary_row("B2", frame) == {"system": "B2", "problems": 0, "api_errors": 1}


# ---- cost arithmetic
def test_usd_by_hand():
    assert cost_probe.usd(2_000_000, 1_000_000, (0.15, 0.60)) == pytest.approx(0.90)  # 2 x .15 + 1 x .60


def test_design_tokens_by_hand():
    # OURS on D1: run 1 costs (10 in, 100 out); later runs without hints cost (4, 40).
    # D1 = 270 x (10 + 2 x 4) = 4,860 in and 270 x (100 + 2 x 40) = 48,600 out.
    # D2 = 210 x 3 x (1, 10) = 630 in and 6,300 out. Ablations = 2 x D1 = 9,720 in and 97,200 out.
    # Totals: 4,860 + 630 + 9,720 = 15,210 in; 48,600 + 6,300 + 97,200 = 152,100 out.
    d1 = {"OURS": {"run1": (10, 100), "later": (4, 40)}}
    d2 = {"OURS": {"run1": (1, 10), "later": (1, 10)}}
    out = cost_probe.design_tokens(d1, d2)
    assert (out["input"], out["output"]) == (15210, 152100)


def test_per_problem_takes_hint_calls_out_of_later_runs():
    frame = pd.DataFrame([{"status": "answered", "topic": "determinant", "size": 3, "live_input_tokens": 10,
                           "live_output_tokens": 100, "live_hint_input_tokens": 6, "live_hint_output_tokens": 60}])
    rates = cost_probe.per_problem({"OURS": frame})["OURS"]
    assert rates == {"run1": (10.0, 100.0), "later": (4.0, 40.0)}


def test_hint_calls_are_counted_separately():
    calls = [{"cached": False, "role": "solver", "prompt_tokens": 5, "total_tokens": 50},
             {"cached": False, "role": "hint", "prompt_tokens": 7, "total_tokens": 20},
             {"cached": False, "role": "hint_verifier", "prompt_tokens": 3, "total_tokens": 10},
             {"cached": True, "role": "hint", "prompt_tokens": 9, "total_tokens": 99}]
    usage = run._usage(calls)
    assert (usage["live_hint_input_tokens"], usage["live_hint_output_tokens"]) == (10, 20)  # (20-7) + (10-3)


# ---- decision of 2026-11-07: OURS withholds only refuted claims
def _ours_row(status, answer, trace, flags, run=1, pid="D1-d"):
    return {"problem_id": pid, "run": run, "topic": "determinant", "size": 3, "status": status, "mode": "sync",
            "answer_json": json.dumps(answer), "trace_json": json.dumps(trace), "hints_json": "[]",
            "flags_json": json.dumps(flags), "pred_misconception": ""}


def test_rq1_counts_delivered_claims_and_rq4_counts_withheld_ones_too():
    # Step 1 is a wrong row operation, withheld. Step 2 is the right determinant, delivered.
    # RQ1 sees only step 2 and the certified answer: no wrong claim. RQ4 sees all 3 claims.
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}, {"value": -2}]
    flags = {"steps": ["withheld", "certified"], "answer": "certified"}
    graded = score.grade_d1(pd.DataFrame([_ours_row("answered", {"determinant": -2}, trace, flags)]), {"D1-d": DET})
    row = graded.iloc[0]
    assert (row["n_claims"], row["n_delivered_claims"], row["n_withheld_claims"]) == (3, 2, 1)
    assert row["wrong_claim"] == 0 and row["n_refuted"] == 1


def test_without_withholding_the_same_response_would_be_wrong():
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}, {"value": -2}]
    stats = score.claim_stats(DET, True, trace, {"determinant": -2}, [])
    assert stats["wrong_claim"] == 1


def test_a_withheld_answer_is_not_delivered_but_its_steps_are():
    trace = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}]
    flags = {"steps": ["certified"], "answer": "withheld"}
    graded = score.grade_d1(pd.DataFrame([_ours_row("answer_withheld", {"determinant": 9}, trace, flags)]), {"D1-d": DET})
    row = graded.iloc[0]
    assert not row["delivered"] and not row["correct"] and not row["wrong"]
    assert (row["n_delivered_claims"], row["n_withheld_claims"], row["wrong_claim"]) == (1, 1, 0)
    table = score.t3b_reliability({"OURS": graded}, ["OURS"], {"OURS": graded})
    assert table.iloc[0]["answer_withheld_rate"] == "1.000 (1/1)"


def test_d1_hints_count_in_run_1_only():
    assert score.hints_counted({"problem_id": "D1-x", "run": 1})
    assert not score.hints_counted({"problem_id": "D1-x", "run": 2})
    assert score.hints_counted({"problem_id": "D2-x", "run": 3})
    assert score.hints_counted({"problem_id": "S017-P2", "run": 2})


def test_single_prompt_hints_in_later_runs_are_set_aside():
    hints = [{"text": "det A is 7", "claims": [{"type": "determinant", "value": 7}]}]
    rows = [{**_ours_row("answered", {"determinant": -2}, [], None, run=r), "hints_json": json.dumps(hints),
             "flags_json": "null"} for r in (1, 2)]
    graded = score.grade_d1(pd.DataFrame(rows), {"D1-d": DET})
    assert graded["wrong_claim"].tolist() == [1, 0] and graded["hints_delivered"].tolist() == [1, 0]


def test_verifier_computes_the_answer_itself_and_it_matches_the_truth():
    from data import gen_d1
    from verifier import compute_answer
    for p in gen_d1.generate(n_per_cell=1):
        assert grader.grade(p, compute_answer(p["topic"], p)), p["id"]


def test_verifier_answer_is_none_when_there_is_no_exact_one():
    from verifier import compute_answer
    assert compute_answer("inverse", {"A": [[1, 2], [2, 4]]}) is None
    assert compute_answer("eigenpairs", {"A": [[0, 1], [2, 0]]}) is None  # eigenvalues +-sqrt(2)


# ---- decision of 2026-11-07: taxonomy-v1 gate on D3 test annotation
def _solution_on(split):
    return next(f"S{i:03d}-P1" for i in range(1000) if d3.split_of(f"S{i:03d}-P1") == split)


def _d3_rows(split, labelled):
    sid = _solution_on(split)
    return pd.DataFrame([{"solution_id": sid, "topic": "determinant", "problem": "A = [[1,2],[3,4]]", "step_no": "1",
                          "step_text": "", "step_value": "-2", "annotator": "a" if labelled else "",
                          "step_label": "correct" if labelled else "", "first_error_step": "",
                          "misconception": ""}])


def test_taxonomy_is_not_frozen_yet():
    assert json.loads((run.ROOT / "data" / "taxonomy.json").read_text(encoding="utf-8"))["frozen"] is False
    assert not d3.taxonomy_v1_ready()


def test_labelled_test_rows_are_refused_until_taxonomy_v1(monkeypatch):
    with pytest.raises(d3.D3Unavailable, match="taxonomy"):
        d3.check_annotation_gate(_d3_rows("test", labelled=True))
    d3.check_annotation_gate(_d3_rows("dev", labelled=True))     # dev annotation builds the taxonomy
    d3.check_annotation_gate(_d3_rows("test", labelled=False))   # transcription alone is fine
    monkeypatch.setattr(d3, "taxonomy_v1_ready", lambda: True)
    d3.check_annotation_gate(_d3_rows("test", labelled=True))


def test_annotation_sheets_for_the_test_split_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="taxonomy-v1"):
        annotation_sheet.make_sheet("test", "a", tmp_path / "d3.csv")


def test_annotation_sheet_for_dev_has_blank_labels(tmp_path, monkeypatch):
    source = tmp_path / "d3.csv"
    _d3_rows("dev", labelled=False).to_csv(source, index=False)
    monkeypatch.setattr(d3, "D3_PATH", tmp_path / "d3.csv")
    out = annotation_sheet.make_sheet("dev", "rater1", source)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert (sheet["annotator"] == "rater1").all() and (sheet["step_label"] == "").all()


def test_runner_refuses_the_d3_test_split_before_taxonomy_v1(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "FROZEN", tmp_path / "FROZEN")
    (tmp_path / "FROZEN").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run.py", "--provider", "groq", "--system", "OURS", "--dataset", "d3", "--split", "test"])
    with pytest.raises(SystemExit, match="taxonomy-v1"):
        run.main()


# ---- claim audit
def _claims(system, topic, n):
    return [{"model": "m", "system": system, "topic": topic, "problem_id": f"p{i}", "where": "step 0",
             "claim_text": "t", "extracted_claim": "{}"} for i in range(n)]


def test_audit_takes_ten_percent_of_every_stratum_rounded_up():
    claims = _claims("B1", "inverse", 100) + _claims("B1", "rref_rank", 25) + _claims("OURS", "inverse", 3)
    counts = Counter((c["system"], c["topic"]) for c in claim_audit.draw(claims))
    assert counts == {("B1", "inverse"): 10, ("B1", "rref_rank"): 3, ("OURS", "inverse"): 1}


def test_audit_sample_is_reproducible():
    claims = _claims("B1", "inverse", 100)
    assert claim_audit.draw(claims) == claim_audit.draw(claims)


def test_audit_sheets_are_identical_and_have_the_required_columns(tmp_path):
    claim_audit.write_sheets(_claims("B1", "inverse", 3), "dev", tmp_path)
    one, two = (pd.read_csv(claim_audit.sheet_path("dev", r, tmp_path)) for r in (1, 2))
    assert one.equals(two) and len(one) == 3
    assert {"claim_text", "extracted_claim", "extraction_correct", "notes"} <= set(one.columns)


def test_audit_never_overwrites_a_sheet(tmp_path):
    claim_audit.write_sheets(_claims("B1", "inverse", 3), "dev", tmp_path)
    with pytest.raises(SystemExit, match="not overwritten"):
        claim_audit.write_sheets(_claims("B1", "inverse", 3), "dev", tmp_path)


def _fill(tmp_path, one, two):
    claim_audit.write_sheets(_claims("B1", "inverse", len(one)), "dev", tmp_path)
    for rater, values in ((1, one), (2, two)):
        path = claim_audit.sheet_path("dev", rater, tmp_path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["extraction_correct"] = values
        frame.to_csv(path, index=False)


def test_audit_agreement_by_hand(tmp_path):
    # 10 claims, 8 agreed, so 0.8. Rater 1 says y 7 times, rater 2 y 7 times.
    # Chance = 0.7^2 + 0.3^2 = 0.58. Kappa = (0.8 - 0.58) / 0.42 = 0.5238.
    one = ["y"] * 6 + ["y", "n"] + ["n", "n"]
    two = ["y"] * 6 + ["n", "y"] + ["n", "n"]
    _fill(tmp_path, one, two)
    result = claim_audit.agreement("dev", tmp_path)
    assert result["percent_agreement"] == pytest.approx(0.8)
    assert result["cohen_kappa"] == pytest.approx(0.22 / 0.42)
    assert result["both_say_correct"] == 6 and len(result["disagreements"]) == 2


def test_audit_refuses_unfinished_sheets(tmp_path):
    _fill(tmp_path, ["y", ""], ["y", "n"])
    with pytest.raises(SystemExit, match="not y or n"):
        claim_audit.agreement("dev", tmp_path)


def test_audit_reads_real_result_rows(tmp_path):
    row = {"problem_id": "D1-x", "run": 1, "status": "answered", "answer_json": json.dumps({"determinant": -2}),
           "trace_json": json.dumps([{"text": "ad-bc", "value": -2}]),
           "hints_json": json.dumps([{"text": "try it", "claims": [{"type": "trace", "value": 5}]}]),
           "pred_misconception": ""}
    pd.DataFrame([row]).to_csv(tmp_path / "B1_d1_dev_run1__groq__m.csv", index=False)
    original = claim_audit.gen_d1.load
    claim_audit.gen_d1.load = lambda: [{**DET, "id": "D1-x"}]
    try:
        claims = claim_audit.all_claims("dev", tmp_path)
    finally:
        claim_audit.gen_d1.load = original
    assert [c["where"] for c in claims] == ["step 0", "final 0", "hint 0"]
    assert claims[2]["claim_text"] == "try it" and '"trace"' in claims[2]["extracted_claim"]


# ---- Gemini batch mode
GEM = {"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.7, "max_tokens": 65536}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "logs" / "calls.jsonl")
    monkeypatch.setattr(batch, "STATE_DIR", tmp_path / "batches")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    return tmp_path


def gemini_reply(text, reason="STOP"):
    return {"response": {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": text}]},
                                         "finishReason": reason}],
                         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 85,
                                           "totalTokenCount": 100}}}


class FakeGemini:
    """Answers submit and poll, and records what it was asked."""

    def __init__(self, polls_until_done=1, shape="response", fail_keys=()):
        self.jobs, self.polls, self.limit = {}, {}, polls_until_done
        self.shape, self.fail_keys = shape, set(fail_keys)

    def __call__(self, cfg, method, url, body=None):
        if method == "POST":
            name = f"batches/{len(self.jobs) + 1}"
            self.jobs[name] = body["batch"]["input_config"]["requests"]["requests"]
            return {"name": name}
        name = url.split("/v1beta/")[1]
        self.polls[name] = self.polls.get(name, 0) + 1
        if self.polls[name] <= self.limit:
            return {"name": name, "metadata": {"state": "JOB_STATE_RUNNING"}}
        results = []
        for item in self.jobs[name]:
            key = item["metadata"]["key"]
            result = {"error": {"code": 500}} if key in self.fail_keys else gemini_reply(f"reply {len(results)}")
            results.append({**result, "metadata": {"key": key}})
        if self.shape == "dest":
            return {"name": name, "state": "JOB_STATE_SUCCEEDED", "dest": {"inlinedResponses": results}}
        return {"name": name, "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {"inlinedResponses": {"inlinedResponses": results}}}


def _replay(prompts):
    """A pipeline of sequential calls: each stops at the first reply it does not have."""
    def replay(collector):
        rows = []
        with llm.use_batch(collector):
            for prompt_chain in prompts:
                try:
                    rows.append([llm.call_llm("role", p) for p in prompt_chain])
                except llm.BatchPending:
                    rows.append(None)
        return rows
    return replay


def test_request_body_shape():
    body = batch.request_body(GEM, "hello")
    assert body["contents"][0]["parts"][0]["text"] == "hello"
    assert body["generationConfig"] == {"temperature": 0.7, "maxOutputTokens": 65536}


def test_cache_entry_counts_thinking_and_keeps_no_latency():
    entry = batch.to_cache_entry(gemini_reply("answer"))
    assert entry["text"] == "answer"  # the thought part is dropped
    assert entry["total_tokens"] == 100 and entry["latency_s"] is None and entry["batch"] is True
    assert entry["finish_reason"] == "stop"
    assert batch.to_cache_entry(gemini_reply("cut", "MAX_TOKENS"))["finish_reason"] == "length"


def test_failed_results_give_no_entry():
    assert batch.to_cache_entry({"error": {"code": 500}}) is None
    assert batch.to_cache_entry({"response": {"candidates": []}}) is None


@pytest.mark.parametrize("job,want", [({"metadata": {"state": "JOB_STATE_RUNNING"}}, "RUNNING"),
                                      ({"state": "JOB_STATE_SUCCEEDED"}, "SUCCEEDED"),
                                      ({"metadata": {"state": "BATCH_STATE_EXPIRED"}}, "EXPIRED")])
def test_both_documented_state_paths(job, want):
    assert batch.job_state(job) == want


def test_both_documented_result_paths():
    item = [{"x": 1}]
    assert batch.inline_results({"response": {"inlinedResponses": {"inlinedResponses": item}}}) == item
    assert batch.inline_results({"dest": {"inlinedResponses": item}}) == item
    assert batch.inline_results({}) == []


@pytest.mark.parametrize("shape", ["response", "dest"])
def test_sequential_calls_finish_over_several_rounds(sandbox, monkeypatch, shape):
    fake = FakeGemini(shape=shape)
    monkeypatch.setattr(batch, "_http", fake)
    rows = batch.drive(llm._settings(), "t", _replay([["a1", "a2"], ["b1"]]), poll_seconds=0)
    assert rows == [["reply 0", "reply 0"], ["reply 1"]]
    assert [len(v) for v in fake.jobs.values()] == [2, 1]  # a1, b1 then a2
    assert not (batch.STATE_DIR / "t.json").exists()


def test_resume_after_a_crash_uses_the_same_job(sandbox, monkeypatch):
    fake = FakeGemini(polls_until_done=1)
    monkeypatch.setattr(batch, "_http", fake)
    original = batch._wait

    def crash(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_wait", crash)
    with pytest.raises(KeyboardInterrupt):
        batch.drive(llm._settings(), "t", _replay([["a1"]]), poll_seconds=0)
    saved = json.loads((batch.STATE_DIR / "t.json").read_text(encoding="utf-8"))
    assert saved["job"] == "batches/1"
    monkeypatch.setattr(batch, "_wait", original)
    rows = batch.drive(llm._settings(), "t", _replay([["a1"]]), poll_seconds=0)
    assert rows == [["reply 0"]] and len(fake.jobs) == 1  # no second job was paid for


def test_network_loss_during_polling_is_survived(sandbox, monkeypatch):
    fake = FakeGemini(polls_until_done=0)
    drops = []

    def flaky(cfg, method, url, body=None):
        if method == "GET" and not drops:
            drops.append(1)
            raise urllib.error.URLError("network is asleep")
        return fake(cfg, method, url, body)

    monkeypatch.setattr(batch, "_http", flaky)
    assert batch.drive(llm._settings(), "t", _replay([["a1"]]), poll_seconds=0) == [["reply 0"]]


def test_a_request_that_keeps_failing_stops_being_retried(sandbox, monkeypatch):
    key = llm.cache_key("gemini", "gemini-3.8-flash", 1, "a1")
    fake = FakeGemini(polls_until_done=0, fail_keys=[key])
    monkeypatch.setattr(batch, "_http", fake)

    def replay(collector):
        with llm.use_batch(collector):
            try:
                return [llm.call_llm("role", "a1")]
            except llm.BatchPending:
                return [None]
            except llm.LLMError as e:
                return [str(e)]

    rows = batch.drive(llm._settings(), "t", replay, poll_seconds=0)
    assert rows == ["batch request failed repeatedly"] and len(fake.jobs) == batch.MAX_KEY_FAILURES


def test_an_expired_job_is_resubmitted(sandbox, monkeypatch):
    fake = FakeGemini(polls_until_done=0)

    def expiring(cfg, method, url, body=None):
        if method == "GET" and url.endswith("batches/1"):
            return {"metadata": {"state": "JOB_STATE_EXPIRED"}}
        return fake(cfg, method, url, body)

    monkeypatch.setattr(batch, "_http", expiring)
    assert batch.drive(llm._settings(), "t", _replay([["a1"]]), poll_seconds=0) == [["reply 0"]]
    assert len(fake.jobs) == 2


def test_oversized_rounds_are_split(monkeypatch):
    monkeypatch.setattr(batch, "MAX_REQUEST_BYTES", 1000)
    taken = batch._take({"a": "x" * 600, "b": "y" * 600, "c": "z"})
    assert list(taken) == ["a"]


def test_replies_from_batch_are_logged_once(sandbox, monkeypatch):
    monkeypatch.setattr(batch, "_http", FakeGemini(polls_until_done=0))
    batch.drive(llm._settings(), "t", _replay([["a1", "a2"]]), poll_seconds=0)
    lines = (sandbox / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2 and all(r["batch"] and r["total_tokens"] == 100 for r in records)
