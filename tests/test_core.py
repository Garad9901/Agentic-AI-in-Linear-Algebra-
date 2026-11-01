"""Unit tests: python -m pytest -q"""
import json
import sys
from pathlib import Path

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latutor import generator  # noqa: E402
from latutor.exact import ParseError, is_rref, num, same_rowspace  # noqa: E402
from latutor.grader import Grader, gold_final_claims  # noqa: E402
from latutor.leakage import detect  # noqa: E402
from latutor.llm import extract_json  # noqa: E402
from latutor.stats import cohen_kappa, holm, matched_or, mcnemar  # noqa: E402
from latutor.verifier import Verifier  # noqa: E402

P = generator.load(ROOT / "data/d1/problems.jsonl")


def test_bank_shape_and_split():
    assert len(P) == 270
    assert sum(p["split"] == "test" for p in P) == 216
    cells = {}
    for p in P:
        cells.setdefault((p["topic"], p["size"]), []).append(p["split"])
    assert all(len(v) == 15 and v.count("dev") == 3 for v in cells.values())
    for p in P:
        assert max(abs(int(x)) for row in p["A"] for x in row) <= 9


def test_bank_is_reproducible():
    again = generator.generate()
    assert [json.dumps(x, sort_keys=True) for x in again[:20]] == [json.dumps(x, sort_keys=True) for x in P[:20]]


def test_eigen_integer_and_irreducible():
    for p in P:
        if p["topic"] in ("eigen", "diag"):
            for e in p["solution"]["eigenvalues"]:
                assert num(e["value"]).q == 1


def test_gold_claims_certified_and_correct():
    for p in P:
        v, g = Verifier(p), Grader(p)
        for c in gold_final_claims(p):
            assert v.check(c).status == "certified", (p["id"], c["type"])
            assert g.grade(c) == "correct", (p["id"], c["type"])


def test_parse_rejects_unsafe():
    for bad in ["__import__('os')", "sqrt(2)", "1e5", "", "1/0"]:
        with pytest.raises(ParseError):
            num(bad)
    assert num("0.25") == sp.Rational(1, 4) and num("-7/4") == sp.Rational(-7, 4)


def test_rref_helpers():
    assert is_rref(sp.Matrix([[1, 0, 2], [0, 1, 3], [0, 0, 0]]))
    assert not is_rref(sp.Matrix([[1, 2, 0], [0, 2, 0]]))
    assert same_rowspace(sp.Matrix([[1, 2], [3, 4]]), sp.eye(2))


def test_uncheckable_and_malformed():
    p = P[0]
    v = Verifier(p)
    assert v.check({"type": "statement", "text": "hi"}).status == "uncheckable"
    assert v.check({"type": "determinant", "value": "pi"}).status == "uncheckable"
    assert v.check({"foo": 1}).status == "uncheckable"


def test_leakage():
    p = next(q for q in P if q["topic"] == "det" and abs(num(q["solution"]["determinant"])) >= 10)
    d = p["solution"]["determinant"]
    g = Grader(p)
    judge = lambda c: g.grade(c) == "correct"  # noqa: E731
    gold = gold_final_claims(p)
    assert detect([{"text": f"so det(A) = {d}", "claims": []}], "det", gold, judge)["leak"]
    assert not detect([{"text": "check the sign in row 2", "claims": []}], "det", gold, judge)["leak"]
    assert detect([{"text": "x", "claims": [{"type": "determinant", "value": d}]}], "det", gold, judge)["leak"]


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('text {"a": {"b": "}"}} more') == {"a": {"b": "}"}}


def test_stats():
    r = mcnemar([1] * 10 + [0] * 30, [0] * 10 + [0] * 30)
    assert r["b"] == 10 and r["c"] == 0 and r["method"] == "exact binomial" and r["p"] < 0.01
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.06)
    orr, lo, hi = matched_or(8, 2)
    assert orr == 4 and lo < 4 < hi
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)


def test_bank_checksum():
    import hashlib
    want = (ROOT / "data/d1/problems.md5").read_text().split()[0]
    got = hashlib.md5((ROOT / "data/d1/problems.jsonl").read_bytes()).hexdigest()
    assert got == want, "problem bank differs from the frozen checksum"
