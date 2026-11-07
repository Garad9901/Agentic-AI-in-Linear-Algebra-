"""RQ1, RQ4, McNemar, Holm, providers, truncation and batch mode.

Every expected number here is worked out by hand in the comment beside it.
Model replies are scripted stand-ins for control flow, and nothing here reaches results/.
"""
import io
import json
import sys
import urllib.error

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
from verifier import check_claim, step_statuses

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
    # Certified = 3 (weak counts) -> coverage 3/4. Refuted 0 -> decidable 3/4. Weak 1/4.
    trace = [{"matrix": [[1, 2], [0, -2]]}, {"value": -2}, {"text": "done"}]
    stats = score.claim_stats(DET, True, trace, {"determinant": -2}, [])
    assert (stats["n_claims"], stats["n_certified"], stats["n_weak"], stats["n_refuted"]) == (4, 3, 1, 0)
    frame = pd.DataFrame([{"problem_id": "d", **stats}])
    row = score.coverage_row(frame, bootstrap=100)
    assert (row["certified_coverage"], row["decidable_share"], row["weak_share"]) == ("0.750", "0.750", "0.250")


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


def test_error_messages_carry_no_organisation_id(tmp_path, monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError("u", 413, "big", {}, io.BytesIO(b"too large in organization `org_01abc9` on tokens"))

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "calls.jsonl")
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
    # One system, OURS, with 1 input and 10 output per problem on D1 and on D2.
    # D1 270 x 3 = 810. D2 210 x 3 = 630. Ablations 2 x 810 = 1620. Sum 3060 problem-runs.
    out = cost_probe.design_tokens({"OURS": (1, 10)}, {"OURS": (1, 10)})
    assert out == {"input": 3060, "output": 30600}


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
