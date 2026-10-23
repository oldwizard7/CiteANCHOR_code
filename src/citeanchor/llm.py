"""OpenAI-compatible chat wrapper with token accounting, plus LLM citation-field extraction."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from .runlog import get as _get_logger

_log = _get_logger("llm")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))

__all__ = ["chat", "usage", "usage_snapshot", "model_name", "OpenAIFatalError",
           "FatalAPIError", "extract_metadata", "parse_json_from_text",
           "OPENAI_MODEL", "OPENAI_MAX_TOKENS"]


class FatalAPIError(Exception):
    """Unrecoverable API error: the caller should abort the whole run."""


class OpenAIFatalError(FatalAPIError):
    pass


_OPENAI_FATAL_PATTERNS = (
    "invalid_api_key", "incorrect api key", "insufficient_quota",
    "exceeded your current quota", "authenticationerror", "permissiondeniederror",
    "unauthorized", " 401", " 403",
)

# --------------------------------------------------------------- accounting ---
# Token counters shared by worker threads.
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
_usage_lock = threading.Lock()


def _record_usage(response) -> None:
    u = getattr(response, "usage", None)
    if u is None:
        return
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    with _usage_lock:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        USAGE["reasoning_tokens"] += reasoning


def usage_snapshot() -> Dict:
    with _usage_lock:
        return dict(USAGE)


def usage() -> dict:
    return usage_snapshot()


def model_name() -> str:
    return OPENAI_MODEL


# -------------------------------------------------------------------- calls ---
_openai_token_param: Optional[str] = None


def _openai_chat(client, prompt: str, max_retries: int = 3, reasoning_effort: Optional[str] = None):
    """One chat completion, tolerant of both token-parameter spellings."""
    global _openai_token_param
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        candidates = ([_openai_token_param] if _openai_token_param
                      else ["max_completion_tokens", "max_tokens"])
        unsupported = 0
        for param in candidates:
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    **{param: OPENAI_MAX_TOKENS},
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                )
                _openai_token_param = param
                _record_usage(response)
                return response
            except Exception as e:
                last_err = e
                err = str(e).lower()
                if any(p in err for p in _OPENAI_FATAL_PATTERNS):
                    raise OpenAIFatalError(f"OpenAI auth/quota error: {e}")
                if "unsupported" in err or "unrecognized" in err or "not supported" in err:
                    unsupported += 1
                    continue  # wrong spelling for this model: try the other
                if "429" in err or "rate limit" in err:
                    break     # back off below, then retry
                raise

        if unsupported == len(candidates):
            raise OpenAIFatalError(
                f"model {OPENAI_MODEL!r} rejected both max_tokens and "
                f"max_completion_tokens: {last_err}")
        if attempt < max_retries - 1:
            wait = (2 ** attempt) * 3
            _log.warning("rate limit, waiting %ss before retry %d/%d", wait, attempt + 1, max_retries)
            time.sleep(wait)

    raise OpenAIFatalError(f"OpenAI failed after {max_retries} retries: {last_err}")


def chat(client, prompt: str, *, max_retries: int = 3, reasoning_effort=None) -> Any:
    """One chat completion. Raises OpenAIFatalError on auth/quota/model errors."""
    return _openai_chat(client, prompt, max_retries=max_retries, reasoning_effort=reasoning_effort)


def parse_json_from_text(text: str) -> Dict:
    """The first JSON object in a model response, or {}."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _generate_json(prompt: str, client) -> Dict:
    if client is None:
        return {}
    response = _openai_chat(client, prompt)
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        _log.warning("empty model reply (finish_reason=%s); raise OPENAI_MAX_TOKENS above %d if this repeats",
                     choice.finish_reason, OPENAI_MAX_TOKENS)
        return {}
    _log.debug("model response: %s", content)
    return parse_json_from_text(content)


# ------------------------------------------------------------- extraction ---
EXTRACT_PROMPT = """\
Extract structured metadata from this academic citation.
Return ONLY a JSON object with these fields:
{{"authors": "author names as they appear", "title": "paper/work title", "venue": "journal, conference, publisher, or N/A", "year": "publication year or N/A"}}

Use the provided structured fields when they are already reliable, but fix obvious OCR or formatting issues.

Citation:
{raw}"""


def extract_metadata(raw: str, client) -> Dict:
    """Title, authors, venue and year of a citation string via one LLM call; {} on failure."""
    if not (raw or "").strip() or client is None:
        return {}
    extracted = _generate_json(EXTRACT_PROMPT.format(raw=raw), client)
    if not extracted:
        return {}
    out: Dict = {"raw": raw}
    for key in ("title", "authors", "venue", "year"):
        value = extracted.get(key)
        out[key] = value if (value and value != "N/A") else "N/A"
    out["metadata_method"] = "llm_extract"
    return out
