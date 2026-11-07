"""Leak checker: does a hint contain the final answer?

`answer` is in the candidate schema of certificates.py. Inside the tutor it is
the Solver's verified answer; in score.py it is the stored ground truth.

Small integers turn up in hints for innocent reasons ("row 2", "add 3 times"),
so a small scalar answer only counts when it is stated as the answer. Matrices
and vectors count when all their entries appear in order, and answers with
fractions count when every entry appears at all. The "reveals it in one easy
step" sense of leaking is not detected: that needs human judgement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import sympy as sp

from .exact import ParseError, mat, num, unify_minus, vec

_STRUCTURAL = [
    r"[Rr]_?\{?\d+\}?",                                    # R2
    r"\b(?:rows?|columns?|cols?|steps?|levels?|entry|entries|position|pivot)\s*\(?\d+(?:\s*(?:,|and|&)\s*\d+)*\)?",
    r"\b\d+\s*(?:x|×|\\times|by)\s*\d+\b",            # 3x3
    r"\b\d+(?:st|nd|rd|th)\b",                             # 2nd
    r"[A-Za-z]\^\{?-?\w+\}?",                              # A^T, P^{-1}, x^2
    r"[A-Za-z]_\{?\w+\}?",                                 # a_12
]
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?![\w])")
_EIGEN_STATED = re.compile(r"(?:eigenvalues?|roots?)[^.\n]{0,30}?(?:=|\bare\b|\bis\b|:)", re.I)
_LAMBDA_VALUE = re.compile(r"(?:\\lambda|λ|\blambda\b)(?:_\{?\d\}?|[₁-₉])?\s*=\s*(-?\d+(?:\s*/\s*\d+)?)")
_NEAR = 15  # a stated value starts this close after "eigenvalues are"
_LIST_SPAN = 50
_STATED = r"[^.\n]{0,40}?(?:=|\bis\b|\bequals\b|\bare\b|:|\bof\b)\s*"


def normalise(text: str) -> str:
    s = unify_minus(str(text or ""))
    s = re.sub(r"\\[dt]?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", r"\1/\2", s)
    s = re.sub(r"\b(?:minus|negative)\s+(?=\d)", "-", s, flags=re.I)
    s = s.replace("&", " ").replace("\\\\", " ").replace("$", " ")
    return s


def numbers_in(text: str) -> list:
    """Exact numbers in reading order, ignoring row labels, shapes and the like."""
    s = normalise(text)
    for pattern in _STRUCTURAL:
        s = re.sub(pattern, " ", s, flags=re.I)
    out = []
    for m in _NUMBER.finditer(s):
        try:
            out.append(num(m.group(0)))
        except ParseError:
            pass
    return out


def _contains_run(found: list, run: list) -> bool:
    k = len(run)
    return any(found[i:i + k] == run for i in range(len(found) - k + 1))


def _sequence_leak(found: list, entries: list) -> bool:
    if not entries:
        return False
    if _contains_run(found, entries):
        return True
    fractional = any(e.q != 1 or abs(e) > 9 for e in entries)
    if not fractional:
        return False
    pool = list(found)
    for e in entries:
        if e not in pool:
            return False
        pool.remove(e)
    return True


def _scalar_leak(text: str, found: list, value, words: str) -> bool:
    forms = {str(value), str(float(value)) if value.q != 1 else str(value)}
    pattern = "|".join(re.escape(f) for f in forms)
    if re.search(rf"(?:{words}){_STATED}(?:{pattern})(?![\d/])", normalise(text), re.I):
        return True
    return (value.q != 1 or abs(value) > 9) and value in found


def _eigen_leak(text: str, values: list) -> bool:
    """Every eigenvalue is stated as one. Numbers merely near the word (matrix entries, "= 0") do not count."""
    s = normalise(text)
    found = {num(m.group(1)) for m in _LAMBDA_VALUE.finditer(s)}
    for m in _EIGEN_STATED.finditer(s):
        window = s[m.end():m.end() + _LIST_SPAN].split("\n")[0]
        first = _NUMBER.search(window)
        if first and first.start() <= _NEAR:
            found.update(numbers_in(window))
    return bool(values) and set(values) <= found


def claims_text(claims) -> str:
    """Render structured claims as sentences so the text rules apply to them too."""
    lines = []
    for c in claims or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("type", "claim")).replace("_", " ")
        if c.get("type") == "eigenvalue":
            kind = "eigenvalue"
        body = {k: v for k, v in c.items() if k not in ("type", "op", "expr")}
        lines.append(f"{kind} = {json.dumps(body)}.".replace('"', " "))
    return "\n".join(lines)


@dataclass
class LeakResult:
    leaked: bool
    reason: str = ""


def _eigenvalues(topic, answer):
    if topic == "eigenpairs":
        return [num(p["eigenvalue"]) for p in answer["eigenpairs"]]
    D = mat(answer["D"])
    return [D[i, i] for i in range(D.rows)]


def leak_check(text: str, topic: str, answer: dict, claims=None) -> LeakResult:
    """Check hint text and its structured claims against the final answer."""
    if not isinstance(answer, dict):
        return LeakResult(False, "no answer to compare against")
    full = str(text or "")
    extra = claims_text(claims)
    if extra:
        full = f"{full}\n{extra}"
    found = numbers_in(full)
    try:
        if topic == "determinant":
            if _scalar_leak(full, found, num(answer["determinant"]), r"det\w*|\|A\||answer|result"):
                return LeakResult(True, "states the determinant")
        elif topic == "rref_rank":
            R = mat(answer["rref"])
            if _sequence_leak(found, list(R)):
                return LeakResult(True, "states the RREF")
            if R == sp.eye(R.rows) and re.search(r"(?:rref|reduces?\s+to|reduced\s+form)[^.\n]{0,40}identity", full, re.I):
                return LeakResult(True, "says the RREF is the identity")
            if _scalar_leak(full, found, num(answer["rank"]), r"\brank"):
                return LeakResult(True, "states the rank")
        elif topic == "inverse":
            if _sequence_leak(found, list(mat(answer["inverse"]))):
                return LeakResult(True, "states the inverse")
        elif topic == "least_squares":
            if _sequence_leak(found, list(vec(answer["x"]))):
                return LeakResult(True, "states the solution x")
        elif topic in ("eigenpairs", "diagonalisation"):
            if _eigen_leak(full, _eigenvalues(topic, answer)):
                return LeakResult(True, "states every eigenvalue")
            if topic == "diagonalisation" and _sequence_leak(found, list(mat(answer["P"]))):
                return LeakResult(True, "states P")
    except (ParseError, KeyError, TypeError):
        return LeakResult(False, "answer could not be read")
    return LeakResult(False)
