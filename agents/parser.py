"""Robust JSON parsing for model replies.

Models wrap JSON in prose or code fences, write bare fractions (1/2), leave
trailing commas, and put raw LaTeX in strings. "\\frac" is the nasty case: it is
valid JSON for a form feed followed by "rac", so it would parse silently into
the wrong text. LaTeX commands are therefore escaped before the first parse.
"""
from __future__ import annotations

import ast
import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_BARE_FRACTION = re.compile(r"(?<![\w.\"])(-?\d+\s*/\s*\d+)(?![\w.\"])")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# LaTeX commands whose first letter is also a JSON escape (b, f, n, r, t, u).
_LATEX = ("frac", "begin", "binom", "bar", "bmatrix", "neq", "ne", "nu", "not", "nabla", "right", "rightarrow",
          "rho", "times", "to", "text", "tau", "theta", "top", "tfrac", "tilde", "underline", "bf", "rm")
_LATEX_RE = re.compile(r"(?:" + "|".join(sorted(_LATEX, key=len, reverse=True)) + r")(?![a-z])")


class ReplyParseError(ValueError):
    pass


def _fix_strings(text: str) -> str:
    """Inside strings, double the backslash of LaTeX commands and of invalid escapes."""
    out, i, in_str = [], 0, False
    while i < len(text):
        ch = text[i]
        if not in_str:
            in_str = ch == '"'
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            rest = text[i + 1:i + 12]
            if nxt in '"\\/':
                out.append(text[i:i + 2])
                i += 2
                continue
            if _LATEX_RE.match(rest) or nxt not in "bfnrtu" or (nxt == "u" and not re.match(r"u[0-9a-fA-F]{4}", rest)):
                out.append("\\\\")
                i += 1
                continue
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _outside_strings(text: str, fn) -> str:
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    return "".join(p if k % 2 else fn(p) for k, p in enumerate(parts))


def _candidates(text: str):
    """Top-level balanced {...} spans in order.

    An unbalanced object means the reply was cut off. Its inner objects are not the reply,
    so scanning stops there rather than returning one step as if it were the whole answer.
    """
    start = text.find("{")
    while start != -1:
        end = None
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    yield text[start:i + 1]
                    break
        if end is None:
            return
        start = text.find("{", end + 1)


def _loads(span: str):
    fixed = _fix_strings(span)
    repaired = _outside_strings(fixed, lambda p: _TRAILING_COMMA.sub(r"\1", _BARE_FRACTION.sub(r'"\1"', p)))
    for attempt in (fixed, repaired):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    try:
        return ast.literal_eval(_BARE_FRACTION.sub(r'"\1"', span))
    except (ValueError, SyntaxError, RecursionError, MemoryError):
        return None


def parse_json(text: str) -> dict:
    """The first JSON object in a reply. Raises ReplyParseError if there is none."""
    if not isinstance(text, str) or not text.strip():
        raise ReplyParseError("empty reply")
    body = _THINK.sub("", text.lstrip("﻿")).strip()
    sources = [m.group(1) for m in _FENCE.finditer(body)] + [body]
    for source in sources:
        for span in _candidates(source):
            obj = _loads(span)
            if isinstance(obj, dict):
                return obj
    raise ReplyParseError("no JSON object found in reply")
