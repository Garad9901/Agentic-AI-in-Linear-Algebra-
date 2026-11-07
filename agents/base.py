"""Shared by all agents: ask the model for strict JSON, with one re-ask."""
from __future__ import annotations

import llm

from . import prompts as P
from .parser import ReplyParseError, parse_json


def ask_json(role: str, prompt: str) -> dict | None:
    """Parsed reply, or None if two replies in a row are not JSON."""
    for attempt in (prompt, P.reask(prompt)):
        try:
            return parse_json(llm.call_llm(role, attempt))
        except ReplyParseError:
            llm.flag_last_call("parse_failed")
    return None
