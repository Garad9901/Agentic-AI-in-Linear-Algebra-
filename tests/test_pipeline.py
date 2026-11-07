"""Parser, LLM wrapper, agents, runner gate and scorer.

Model replies here are scripted stand-ins for testing control flow. They live
only in these tests and nothing they produce is written to results/.
"""
import http.client
import io
import json
import sys
import threading
import urllib.error

import pandas as pd
import pytest

import d3 as d3_data
import grader
import llm
import run
import score
import systems
from agents import base, diagnostic, hint, verification
from agents.parser import ReplyParseError, parse_json
from data import gen_d1
from systems import run_system

P = {"id": "t", "topic": "determinant", "size": 2, "problem": "Find det.\nA = [[1, 2], [3, 4]]", "A": [[1, 2], [3, 4]],
     "ground_truth": {"determinant": -2}, "split": "dev"}
GOOD = json.dumps({"steps": [{"text": "ad-bc", "expr": "1*4-2*3", "value": -2}], "answer": {"determinant": -2}})
BAD = json.dumps({"steps": [{"text": "ad-bc", "expr": "1*4+2*3", "value": 10}], "answer": {"determinant": 10}})
NO_LEAK = json.dumps({"text": "Which two products does a 2 by 2 determinant use?", "claims": []})
LEAK = json.dumps({"text": "The determinant is -2.", "claims": []})
FALSE = json.dumps({"text": "You should get 77.", "claims": []})


# ---- parser
@pytest.mark.parametrize("reply,want", [
    ('{"a": 1}', {"a": 1}),
    ('Sure!\n```json\n{"a": 1}\n```\nDone.', {"a": 1}),
    ('Here you go: {"a": [1, 2]} hope it helps', {"a": [1, 2]}),
    ('{"a": [[1/2, -3/4], [1, 2]]}', {"a": [["1/2", "-3/4"], [1, 2]]}),
    ('{"a": [1, 2,], }', {"a": [1, 2]}),
    ('<think>{"no": 1}</think>{"a": 1}', {"a": 1}),
    ("{'a': 1, 'b': None}", {"a": 1, "b": None}),
    ('﻿{"a": 1}', {"a": 1}),
])
def test_parser_repairs(reply, want):
    assert parse_json(reply) == want


def test_parser_keeps_latex_intact():
    assert parse_json(r'{"t": "use \frac{1}{2} and \times and \lambda"}')["t"] == r"use \frac{1}{2} and \times and \lambda"


def test_parser_keeps_real_newlines():
    assert parse_json(r'{"t": "line one\nNext line"}')["t"] == "line one\nNext line"


def test_parser_leaves_fractions_in_strings_alone():
    assert parse_json('{"t": "1/2 of it", "v": "3/4"}') == {"t": "1/2 of it", "v": "3/4"}


@pytest.mark.parametrize("reply", ["", "no json here", "{broken", None,
                                   '{"steps": [{"text": "a", "value": 1}, {"text": "b"'])
def test_parser_rejects(reply):
    with pytest.raises(ReplyParseError):
        parse_json(reply)


# ---- LLM wrapper
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def reply(text="hello"):
    return FakeResponse(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode())


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "logs" / "calls.jsonl")
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_RPM", "100000")
    sleeps.clear()
    return tmp_path


sleeps: list = []


def test_llm_sends_prereg_temperature_and_caches(sandbox, monkeypatch):
    bodies = []

    def urlopen(req, timeout):
        bodies.append(json.loads(req.data))
        return reply("hello")

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert llm.call_llm("solver", "p") == "hello"
    assert llm.call_llm("hint", "p") == "hello"  # cache is keyed on (model, prompt)
    assert len(bodies) == 1 and bodies[0]["temperature"] == 0.7
    assert llm.call_llm("solver", "other") == "hello"
    assert len(bodies) == 2
    monkeypatch.setenv("LLM_MODEL", "another-model")
    llm.call_llm("solver", "p")
    assert len(bodies) == 3


def test_llm_logs_role_tokens_latency_and_no_secrets(sandbox, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: reply())
    llm.call_llm("solver", "secret prompt")
    llm.call_llm("solver", "secret prompt")
    raw = (sandbox / "logs" / "calls.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[0]["role"] == "solver" and rows[0]["prompt_tokens"] == 7 and rows[0]["completion_tokens"] == 3
    assert "latency_s" in rows[0]
    assert "test-key" not in raw and "secret prompt" not in raw


def test_llm_backs_off_exponentially_on_rate_limits(sandbox, monkeypatch):
    attempts = []

    def urlopen(req, timeout):
        attempts.append(1)
        if len(attempts) < 4:
            raise urllib.error.HTTPError("u", 429, "slow", {}, io.BytesIO(b"rate limited"))
        return reply()

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert llm.call_llm("solver", "p") == "hello"
    waits = [s for s in sleeps if s >= 1]
    assert waits == [5.0, 10.0, 20.0]


def test_llm_separates_waiting_from_compute(sandbox, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: reply())
    with llm.track() as calls:
        llm.call_llm("solver", "p")
        llm.call_llm("solver", "p")
    assert calls[0]["request_s"] >= 0 and calls[1]["request_s"] == 0


def test_llm_caps_the_reply_to_the_tpm_limit(sandbox, monkeypatch):
    bodies = []

    def urlopen(req, timeout):
        bodies.append(json.loads(req.data))
        return reply()

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("LLM_TPM", "8000")
    llm.call_llm("solver", "x" * 2500)
    assert bodies[0]["max_tokens"] == 8000 - int(2500 / llm.CHARS_PER_TOKEN) - llm.TPM_MARGIN
    with pytest.raises(llm.LLMError, match="no reply budget"):
        llm.call_llm("solver", "x" * 20000)


def test_each_run_gets_a_fresh_response_and_reruns_are_cached(sandbox, monkeypatch):
    replies = iter(["first", "second", "third"])
    calls = []

    def urlopen(req, timeout):
        calls.append(1)
        return reply(next(replies))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    texts = []
    for run in (1, 2, 3):
        with llm.use_run(run):
            texts.append(llm.call_llm("solver", "same prompt"))
    assert texts == ["first", "second", "third"] and len(calls) == 3  # same prompt, three fresh responses
    for run, want in zip((1, 2, 3), texts):
        with llm.use_run(run):
            assert llm.call_llm("solver", "same prompt") == want
    assert len(calls) == 3  # rerunning any run costs nothing


def test_cache_key_depends_on_the_run():
    assert llm.cache_key("groq", "m", 1, "p") != llm.cache_key("groq", "m", 2, "p")
    assert llm.cache_key("groq", "m", 1, "p") == llm.cache_key("groq", "m", 1, "p")


def test_use_run_restores_the_previous_run():
    with llm.use_run(2):
        with llm.use_run(3):
            assert llm.current_run() == 3
        assert llm.current_run() == 2


def test_runs_and_tracking_do_not_leak_between_threads(sandbox, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: reply("x"))
    barrier = threading.Barrier(2)
    seen = {}

    def work(run_index):
        with llm.track() as calls, llm.use_run(run_index):
            barrier.wait(timeout=5)
            llm.call_llm("solver", f"prompt {run_index}")
            seen[run_index] = [c["run"] for c in calls]

    threads = [threading.Thread(target=work, args=(k,)) for k in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {1: [1], 2: [2]}


def test_llm_pacing_spaces_requests_across_threads(sandbox, monkeypatch):
    monkeypatch.setattr(llm, "_next_slot", 0.0)
    monkeypatch.setattr(llm.time, "time", lambda: 1000.0)
    sleeps.clear()
    for _ in range(3):
        llm._pace(60.0)
    assert sleeps == [1.0, 2.0]


def test_llm_counts_timeouts_and_retries_them(sandbox, monkeypatch):
    attempts = []

    def urlopen(req, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise TimeoutError("timed out")
        return reply()

    monkeypatch.setenv("LLM_TIMEOUT", "42")
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with llm.track() as calls:
        assert llm.call_llm("solver", "p") == "hello"
    assert attempts == [42, 42] and calls[0]["timeouts"] == 1


def test_llm_retries_a_dropped_connection(sandbox, monkeypatch):
    attempts = []

    def urlopen(req, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise http.client.RemoteDisconnected("closed")
        return reply()

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert llm.call_llm("solver", "p") == "hello"


def test_total_tokens_include_thinking(sandbox, monkeypatch):
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 90}}
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResponse(json.dumps(body).encode()))
    with llm.track() as calls:
        llm.call_llm("solver", "p")
    assert calls[0]["total_tokens"] == 90
    assert run._usage(calls)["total_tokens"] == 90 and run._usage(calls)["live_tokens"] == 90


def test_llm_truncated_empty_reply_is_not_an_outage(sandbox, monkeypatch):
    cut = FakeResponse(json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: cut)
    assert llm.call_llm("solver", "p") == ""


def test_llm_does_not_retry_a_client_error(sandbox, monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.LLMError):
        llm.call_llm("solver", "p")


def test_llm_gives_up_at_once_on_a_daily_quota(sandbox, monkeypatch):
    attempts = []

    def urlopen(req, timeout):
        attempts.append(1)
        body = b'{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}'
        raise urllib.error.HTTPError("u", 429, "quota", {}, io.BytesIO(b"x" * 900 + body))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.LLMError, match="daily quota"):
        llm.call_llm("solver", "p")
    assert len(attempts) == 1 and not sleeps


def test_llm_stops_when_told_to_wait_too_long(sandbox, monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError("u", 429, "slow", {"Retry-After": "9999"}, io.BytesIO(b"x"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.LLMError, match="quota"):
        llm.call_llm("solver", "p")


# ---- agents, with a scripted model
def script(monkeypatch, replies):
    """Replies are consumed in order, whatever the role."""
    queue = list(replies)
    roles = []

    def fake(role, prompt):
        roles.append(role)
        return queue.pop(0)

    monkeypatch.setattr(llm, "call_llm", fake)
    return roles


def test_verification_passes_first_time(monkeypatch):
    script(monkeypatch, [GOOD])
    result = verification.solve_verified(P)
    assert result.verified and result.retries == 0


def test_verification_retries_with_the_reason(monkeypatch):
    prompts = []

    def fake(role, prompt):
        prompts.append(prompt)
        return BAD if len(prompts) == 1 else GOOD

    monkeypatch.setattr(llm, "call_llm", fake)
    result = verification.solve_verified(P)
    assert result.verified and result.retries == 1
    assert "rejected it" in prompts[1] and "determinant is not exactly" in prompts[1]


def test_verification_abstains_after_two_retries(monkeypatch):
    roles = script(monkeypatch, [BAD, BAD, BAD])
    result = verification.solve_verified(P)
    assert not result.verified and result.solution is None and result.retries == 2
    assert len(roles) == 3


def test_verification_steps_mode_rejects_a_bad_trace(monkeypatch):
    right_answer_bad_trace = json.dumps({"steps": [{"expr": "1*4-2*3", "value": 5}], "answer": {"determinant": -2}})
    script(monkeypatch, [right_answer_bad_trace] * 3)
    assert not verification.solve_verified(P, "steps").verified
    script(monkeypatch, [right_answer_bad_trace])
    assert verification.solve_verified(P, "final").verified


def test_hint_regenerates_then_passes(monkeypatch):
    script(monkeypatch, [LEAK, FALSE, NO_LEAK, NO_LEAK, NO_LEAK])
    solution = {"answer": {"determinant": -2}, "steps": []}
    h = hint.make_hints(P, solution, reject=hint.certificate_rejection(P, solution))[0]
    assert not h["withheld"] and h["regenerations"] == 2


def test_hint_withheld_after_two_regenerations(monkeypatch):
    roles = script(monkeypatch, [LEAK, LEAK, LEAK, NO_LEAK, NO_LEAK])
    solution = {"answer": {"determinant": -2}, "steps": []}
    h = hint.make_hints(P, solution, reject=hint.certificate_rejection(P, solution))[0]
    assert h["withheld"] and h["text"] == "" and len(roles) == 5


def test_diagnostic_empty_taxonomy_gives_no_label(monkeypatch):
    p = {**P, "student_steps": [{"step_no": 1, "step_text": "x", "step_value": ""}]}
    script(monkeypatch, ['{"first_wrong_step": 1, "misconception": "MADE_UP"}'])
    assert diagnostic.diagnose(p, []) == {"first_wrong_step": 1, "misconception": None}


def test_diagnostic_rejects_labels_outside_the_taxonomy(monkeypatch):
    p = {**P, "student_steps": [{"step_no": 1, "step_text": "x", "step_value": ""}]}
    script(monkeypatch, ['{"first_wrong_step": 9, "misconception": "OTHER"}'])
    assert diagnostic.diagnose(p, ["SIGN_ERROR"]) == {"first_wrong_step": None, "misconception": None}
    script(monkeypatch, ['{"first_wrong_step": 1, "misconception": "SIGN_ERROR"}'])
    assert diagnostic.diagnose(p, ["SIGN_ERROR"])["misconception"] == "SIGN_ERROR"


def test_taxonomy_file_is_strict_json_with_an_empty_list():
    raw = json.loads(diagnostic.TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert raw["categories"] == [] and "_comment" in raw
    assert diagnostic.load_taxonomy() == []


# ---- systems
def single_reply(answer=-2):
    return json.dumps({"steps": [], "answer": {"determinant": answer},
                       "hints": [{"level": i, "text": "Think.", "claims": []} for i in (1, 2, 3)]})


def test_b1_is_one_call(monkeypatch):
    roles = script(monkeypatch, [single_reply()])
    out = run_system("B1", P, [])
    assert roles == ["single"] and out["answer"] == {"determinant": -2} and len(out["hints"]) == 3


def test_b3_adds_a_self_check_that_can_change_the_answer(monkeypatch):
    roles = script(monkeypatch, [single_reply(5), single_reply(-2)])
    out = run_system("B3", P, [])
    assert roles == ["single", "self_check"] and out["answer"] == {"determinant": -2}


JUDGE_OK = '{"verdict": "correct", "reason": "fine"}'
JUDGE_NO = '{"verdict": "incorrect", "reason": "recheck the sign"}'


def test_b2_llm_verifier_can_approve_a_wrong_answer(monkeypatch):
    # The point of B2: an LLM judge is no independent check, so this reaches the student.
    roles = script(monkeypatch, [BAD, JUDGE_OK, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK])
    out = run_system("B2", P, [])
    assert out["status"] == "answered" and out["answer"] == {"determinant": 10}
    assert roles[:2] == ["solver", "llm_verifier"] and roles.count("llm_verifier") == 4


def test_b2_rejection_retries_then_abstains(monkeypatch):
    prompts = []
    replies = iter([BAD, JUDGE_NO, BAD, JUDGE_NO, BAD, JUDGE_NO])

    def fake(role, prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(llm, "call_llm", fake)
    out = run_system("B2", P, [])
    assert out["status"] == "abstained" and out["answer"] is None and out["retries"] == 2
    assert "recheck the sign" in prompts[2]


def test_b2_regenerates_a_rejected_hint(monkeypatch):
    script(monkeypatch, [GOOD, JUDGE_OK, NO_LEAK, JUDGE_NO, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK])
    out = run_system("B2", P, [])
    assert out["hints"][0]["regenerations"] == 1 and len(out["hints"]) == 3


def test_b2_never_uses_the_certificates(monkeypatch):
    monkeypatch.setattr(verification, "verify", lambda *a: pytest.fail("B2 must not call the SymPy verifier"))
    monkeypatch.setattr(verification, "check_steps", lambda *a: pytest.fail("B2 must not call the step checker"))
    script(monkeypatch, [GOOD, JUDGE_OK, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK, NO_LEAK, JUDGE_OK])
    assert run_system("B2", P, [])["status"] == "answered"


def test_b4_never_filters(monkeypatch):
    roles = script(monkeypatch, [BAD, LEAK, LEAK, LEAK])
    out = run_system("B4", P, [])
    assert roles == ["solver", "hint", "hint", "hint"]
    assert out["answer"] == {"determinant": 10} and out["hints"][0]["text"] == "The determinant is -2."


def test_ours_abstention_reaches_the_student_with_nothing(monkeypatch):
    roles = script(monkeypatch, [BAD, BAD, BAD])
    out = run_system("OURS", P, [])
    assert out["status"] == "abstained" and out["answer"] is None and out["hints"] == []
    assert roles == ["solver"] * 3


def test_truncated_replies_count_as_no_answer_for_every_system(monkeypatch):
    cut = '{"steps": [{"text": "a", "value": 1}, {"text": "b"'
    for name in ("B1", "B3", "OURS"):
        script(monkeypatch, [cut] * 6)
        out = run_system(name, P, [])
        assert out["status"] == "parse_error" and out["answer"] is None and out["hints"] == [], name


def test_ours_full_path(monkeypatch):
    roles = script(monkeypatch, [GOOD, NO_LEAK, NO_LEAK, NO_LEAK])
    out = run_system("OURS", P, [])
    assert out["status"] == "answered" and roles == ["solver", "hint", "hint", "hint"]


def test_a1_removes_verification(monkeypatch):
    script(monkeypatch, [BAD, LEAK, LEAK, LEAK])
    out = run_system("A1", P, [])
    assert out["status"] == "answered" and out["answer"] == {"determinant": 10}


def test_a2_verifies_the_final_answer_only(monkeypatch):
    right_answer_bad_trace = json.dumps({"steps": [{"expr": "1*4-2*3", "value": 5}], "answer": {"determinant": -2}})
    script(monkeypatch, [right_answer_bad_trace, NO_LEAK, NO_LEAK, NO_LEAK])
    assert run_system("A2", P, [])["status"] == "answered"
    script(monkeypatch, [right_answer_bad_trace] * 3)
    assert run_system("OURS", P, [])["status"] == "abstained"


def test_a4_gives_a_single_hint(monkeypatch):
    roles = script(monkeypatch, [GOOD, NO_LEAK])
    out = run_system("A4", P, [])
    assert len(out["hints"]) == 1 and roles == ["solver", "hint"]


STUDENT = {**P, "id": "S1-P1", "student_steps": [
    {"step_no": 1, "step_text": "R2 <- R2 - 3R1", "step_value": "[[1,2],[0,-3]]", "check": [[1, 2], [0, -3]]},
    {"step_no": 2, "step_text": "det is -3", "step_value": "-3", "check": -3}]}


def test_ours_on_student_work_uses_the_step_checker(monkeypatch):
    roles = script(monkeypatch, [GOOD, '{"first_wrong_step": 2, "misconception": "SIGN"}', NO_LEAK, NO_LEAK, NO_LEAK])
    out = run_system("OURS", STUDENT, ["SIGN"])
    assert out["pred_first_wrong_step"] == 1  # the certificate wins over the model's guess
    assert out["pred_misconception"] == "SIGN" and "diagnostic" in roles


def test_det_only_locates_the_first_error(monkeypatch):
    roles = script(monkeypatch, ['{"first_wrong_step": 2}'])
    out = run_system("DET", STUDENT, [])
    assert roles == ["detector"] and out["pred_first_wrong_step"] == 2 and out["hints"] == []


def test_a3_removes_the_diagnostic_agent(monkeypatch):
    roles = script(monkeypatch, [GOOD, NO_LEAK, NO_LEAK, NO_LEAK])
    out = run_system("A3", STUDENT, ["SIGN"])
    assert out["pred_first_wrong_step"] == 1 and out["pred_misconception"] is None
    assert "diagnostic" not in roles


# ---- runner
def test_test_split_needs_frozen(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(run, "FROZEN", tmp_path / "FROZEN")
    monkeypatch.setattr(sys, "argv", ["run.py", "--provider", "groq", "--system", "B1", "--dataset", "d1", "--split", "test"])
    with pytest.raises(SystemExit, match="FROZEN"):
        run.main()


def test_det_refuses_datasets_without_student_work(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--provider", "groq", "--system", "DET", "--dataset", "d1", "--split", "dev"])
    with pytest.raises(SystemExit, match="student work"):
        run.main()


def test_run_info_records_model_and_date(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(run, "RUN_INFO", tmp_path / "RUN_INFO.md")
    run.log_run("B1", "d1", "dev", 2, 20, "complete")
    run.log_run("B2", "d1", "dev", 2, 20, "stopped: API errors")
    text = (tmp_path / "RUN_INFO.md").read_text(encoding="utf-8")
    assert text.count("gemini-3.8-flash") == 2 and "| 2 | 20 |" in text and "0.7" in text
    assert "stopped: API errors" in text


def test_d3_refuses_without_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr(d3_data, "D3_PATH", tmp_path / "missing.csv")
    with pytest.raises(SystemExit, match="cannot run D3"):
        run.load_dataset("d3", "dev")


def test_round_robin_covers_topics_first():
    problems = [{"id": f"{t}-{s}-{k}", "topic": t, "size": s} for t in gen_d1.TOPICS for s in gen_d1.SIZES for k in range(3)]
    first = run.round_robin(problems)[:5]
    assert len({p["topic"] for p in first}) == 5
    assert {p["size"] for p in first} == set(gen_d1.SIZES)
    assert len({(p["topic"], p["size"]) for p in run.round_robin(problems)[:18]}) == 18
    assert sorted(p["id"] for p in run.round_robin(problems)) == sorted(p["id"] for p in problems)


def test_d3_template_has_the_required_columns_and_no_rows():
    frame = pd.read_csv(d3_data.TEMPLATE_PATH)
    assert list(frame.columns) == d3_data.COLUMNS and frame.empty


def test_student_data_stays_out_of_git():
    gitignore = (run.ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/private/" in gitignore and d3_data.D3_PATH.parent.name == "private"


# ---- scoring
def test_cohen_kappa_known_values():
    assert score.cohen_kappa(["a", "b"], ["a", "b"]) == 1
    # Worked example: observed 0.7, chance 0.5, so kappa = 0.4.
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert score.cohen_kappa(a, b) == pytest.approx(0.4)
    assert score.cohen_kappa([], []) is None
    assert score.cohen_kappa(["a", "a"], ["a", "a"]) is None


def test_macro_f1_and_step_counts():
    assert score.macro_f1(["a", "b"], ["a", "b"]) == 1
    assert score.macro_f1(["a", "a"], ["b", "b"]) == 0
    assert score.macro_f1([], []) is None
    labels = {1: "correct", 2: "wrong", 3: "wrong"}
    assert score.step_counts(labels, 2) == (1, 0, 0)  # step 3 lies after the prediction, so it is not scored
    assert score.step_counts(labels, None) == (0, 0, 2)
    assert score.step_counts(labels, 1) == (0, 1, 0)


def _row(pid, status, answer, trace=None, hints=(), topic="determinant"):
    return {"problem_id": pid, "topic": topic, "size": 3, "status": status, "answer_json": json.dumps(answer),
            "trace_json": json.dumps(trace), "hints_json": json.dumps(list(hints))}


EIG = {"topic": "eigenpairs", "size": 2, "A": [[2, 1], [1, 2]], "ground_truth": {  # eigenvalues 1, 3; trace 4
    "eigenvalues": [{"value": 1, "multiplicity": 1}, {"value": 3, "multiplicity": 1}],
    "eigenspaces": [{"eigenvalue": 1, "basis": [[-1, 1]]}, {"eigenvalue": 3, "basis": [[1, 1]]}]}}
DIAG = {**EIG, "topic": "diagonalisation"}
# Wrong, but 0 + 4 = trace(A), so it passes the surface check.
PLAUSIBLE = {"eigenpairs": [{"eigenvalue": 0, "eigenvector": [1, 0]}, {"eigenvalue": 4, "eigenvector": [0, 1]}]}
SUM_OFF = {"eigenpairs": [{"eigenvalue": 0, "eigenvector": [1, 0]}, {"eigenvalue": 9, "eigenvector": [0, 1]}]}
RIGHT = {"eigenpairs": [{"eigenvalue": 1, "eigenvector": [1, -1]}, {"eigenvalue": 3, "eigenvector": [1, 1]}]}
SHOWN = [{"eigenvalue": 0, "vector": [1, 0]}, {"eigenvalue": 4, "vector": [0, 1]}]


def test_fabricated_needs_wrong_underived_and_plausible():
    assert grader.fabricated(EIG, PLAUSIBLE, None) is True            # no working shown
    assert grader.fabricated(EIG, PLAUSIBLE, []) is True              # empty working
    assert grader.fabricated(EIG, PLAUSIBLE, [{"text": "so"}]) is True  # working computes nothing
    other = [{"eigenvalue": 2, "vector": [1, 1]}]
    assert grader.fabricated(EIG, PLAUSIBLE, other) is True           # differs from the working


def test_not_fabricated_when_the_working_arrives_at_it():
    assert grader.fabricated(EIG, PLAUSIBLE, SHOWN) is False
    assert grader.consistent_computation_error(EIG, PLAUSIBLE, SHOWN) is True


def test_not_fabricated_when_the_surface_check_fails():
    assert grader.fabricated(EIG, SUM_OFF, None) is False
    assert grader.fabricated(EIG, {"eigenpairs": []}, None) is False


def test_not_fabricated_or_consistent_when_correct():
    assert grader.fabricated(EIG, RIGHT, None) is False
    assert grader.consistent_computation_error(EIG, RIGHT, None) is False


def test_fabricated_diagonalisation_uses_trace_of_d():
    wrong = {"P": [[1, 0], [0, 1]], "D": [[0, 0], [0, 4]]}
    assert grader.passes_surface_check(DIAG, wrong) is True
    assert grader.fabricated(DIAG, wrong, None) is True
    off = {"P": [[1, 0], [0, 1]], "D": [[0, 0], [0, 9]]}
    assert grader.passes_surface_check(DIAG, off) is False
    assert grader.fabricated(DIAG, off, None) is False
    assert grader.passes_surface_check(DIAG, {"P": "x", "D": "x"}) is False


def test_fabricated_is_na_without_a_surface_check():
    for topic in ("determinant", "inverse", "rref_rank", "least_squares"):
        p = {**P, "topic": topic}
        assert grader.fabricated(p, {"determinant": 9}, None) is None
        assert grader.passes_surface_check(p, {"determinant": 9}) is None


def test_consistent_computation_error_uses_the_last_value():
    assert grader.consistent_computation_error(P, {"determinant": 9}, [{"value": 9}]) is True
    assert grader.consistent_computation_error(P, {"determinant": 9}, [{"value": 9}, {"value": 8}]) is False
    assert grader.consistent_computation_error(P, {"determinant": 9}, None) is False
    assert grader.consistent_computation_error(P, {"determinant": -2}, [{"value": -2}]) is False


def test_consistent_and_underived_are_exact_complements_among_wrong_answers():
    for trace in (None, [], [{"value": 9}], [{"value": 8}]):
        consistent = grader.consistent_computation_error(P, {"determinant": 9}, trace)
        assert consistent == grader.follows_from_working("determinant", trace, {"determinant": 9})


def test_grading_separates_the_two_metrics_and_excludes_api_errors():
    problems = {"p0": P, "p1": P, "p2": P, "p3": P, "p4": P, "e": EIG}
    frame = pd.DataFrame([
        _row("p0", "answered", {"determinant": -2}, [{"value": -2}]),
        _row("p1", "answered", {"determinant": 9}, [{"value": 9}]),      # wrong, working agrees
        _row("p2", "answered", {"determinant": 9}, [{"value": 8}]),      # wrong, working disagrees
        _row("p3", "abstained", None),
        _row("p4", "api_error", None),
        _row("e", "answered", PLAUSIBLE, None, topic="eigenpairs"),      # fabricated
    ])
    graded = score.grade_d1(frame, problems)
    assert len(graded) == 5
    assert graded["correct"].tolist() == [True, False, False, False, False]
    assert graded["consistent_error"].tolist() == [False, True, False, False, False]
    assert graded["fabricated"].tolist() == [None, None, None, None, True]
    assert graded["delivered"].tolist() == [True, True, True, False, True]
    assert score._fabricated_rate(graded) == "1.000 (1/1)"
    assert score._fabricated_rate(graded[graded["topic"] == "determinant"]) == "n/a"


def test_empty_answer_is_wrong_but_not_fabricated():
    row = _row("e", "answered", {"eigenpairs": []}, [{"text": "x"}], topic="eigenpairs")
    graded = score.grade_d1(pd.DataFrame([row]), {"e": EIG})
    assert graded["wrong"].tolist() == [True]
    assert graded["fabricated"].tolist() == [False] and graded["consistent_error"].tolist() == [False]


def test_kappa_table_from_two_annotators():
    rows = []
    for sid, a, b in (("S1-P1", "correct", "correct"), ("S2-P1", "wrong", "wrong"), ("S3-P1", "wrong", "correct")):
        rows.append({"solution_id": sid, "step_no": 1, "annotator": "ann1", "step_label": a,
                     "first_error_step": "1" if a == "wrong" else "", "misconception": ""})
        rows.append({"solution_id": sid, "step_no": 1, "annotator": "ann2", "step_label": b,
                     "first_error_step": "1" if b == "wrong" else "", "misconception": ""})
    table, notes = score.t4b_kappa(pd.DataFrame(rows))
    assert not notes
    assert table.iloc[0]["items"] == 3 and table.iloc[0]["kappa"] == "0.400"


def test_gold_needs_agreement_or_a_gold_row():
    base_row = {"step_no": 1, "step_label": "wrong", "misconception": ""}
    rows = [{**base_row, "solution_id": "S1-P1", "annotator": "a", "first_error_step": "1"},
            {**base_row, "solution_id": "S1-P1", "annotator": "b", "first_error_step": "1"},
            {**base_row, "solution_id": "S2-P1", "annotator": "a", "first_error_step": "1"},
            {**base_row, "solution_id": "S2-P1", "annotator": "b", "first_error_step": "2"}]
    gold, left_out = d3_data.gold_labels(pd.DataFrame(rows))
    assert list(gold) == ["S1-P1"] and left_out == 1


def test_tables_leave_d3_blank_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(d3_data, "read_rows", lambda: (_ for _ in ()).throw(d3_data.D3Unavailable("no data yet")))
    tables = {t[0]: t for t in score.build("dev", None, tmp_path, bootstrap=100)}
    t4 = tables["T4"][2]
    assert t4["first_wrong_step_accuracy"].eq("").all()
    assert any("no data yet" in n for n in tables["T4"][3])
    assert tables["T4b"][2]["kappa"].eq("").all()


def test_deviations_state_they_precede_the_test_split():
    rows = [line for line in (run.ROOT / "docs" / "DEVIATIONS.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| 2026-11-07")]
    assert rows and all("Made before any test-split model call." in r for r in rows)


def test_no_test_split_results_exist():
    assert not list(run.RESULTS.glob("*_test_run*.csv"))
    assert not list(run.RESULTS.rglob("*_test_run*.csv"))
    assert not run.FROZEN.exists()


def test_system_names():
    assert systems.SYSTEMS == ["B1", "B2", "B3", "B4", "OURS"]
    assert systems.ABLATIONS == ["A1", "A2", "A3", "A4"]
