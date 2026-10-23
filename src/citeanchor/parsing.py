"""Parsing a raw citation string into a `Query`."""
from __future__ import annotations

import re

import os
import sys
from typing import Optional

from .runlog import get as _get_logger
from .records import (Query, extract_arxiv_id, extract_doi, extract_url,
                      extract_volume_pages)
from .source_type import UNKNOWN, classify as classify_source, classify_llm

log = _get_logger("parsing")


def llm_query(raw: str, client=None) -> Optional[Query]:
    """Parse with one LLM call; None when no API key is set or the call fails."""
    from .llm import extract_metadata
    if not os.getenv("OPENAI_API_KEY"):
        log.debug("no LLM key configured; falling back to the regex parser")
        return None

    if client is None:
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except Exception:
            client = None

    try:
        meta = extract_metadata(raw, client)
    except Exception as exc:
        log.warning("LLM parse failed (%s: %s); falling back to the regex parser",
                    type(exc).__name__, exc)
        return None
    if not meta or not meta.get("title"):
        log.warning("LLM parse returned no title (%r); falling back to regex", meta)
        return None

    year = meta.get("year")
    try:
        year = int(str(year)[:4])
    except (TypeError, ValueError):
        year = None

    authors_raw = meta.get("authors") or ""
    authors = [a.strip() for a in authors_raw.replace(" and ", ",").split(",")
               if a.strip() and a.strip().lower() not in ("et al", "et al.")]
    authors = _rejoin_inverted(authors)

    venue = meta.get("venue") or ""
    return Query(
        raw=raw,
        title=str(meta["title"]).strip().rstrip("."),
        authors=authors,
        year=year,
        venue="" if venue == "N/A" else venue,
        doi=extract_doi(raw) or "",
        arxiv_id=extract_arxiv_id(raw) if "arxiv" in raw.lower() else "",
        parser="llm",
    )


_INITIALS = re.compile(r"^(?:[A-Z]\.?[\s-]*){1,4}$")


def _rejoin_inverted(parts: list) -> list:
    """Rejoin surname/initials pairs split on commas: ["Smith", "J."] -> ["J. Smith"]."""
    if len(parts) < 2 or not any(_INITIALS.match(p) for p in parts):
        return parts
    out, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and not _INITIALS.match(parts[i]) and _INITIALS.match(parts[i + 1]):
            out.append(f"{parts[i + 1]} {parts[i]}")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def build_query(raw: str, client=None, prefer_llm: bool = True) -> Query:
    """LLM parse when possible, regex otherwise."""
    q = llm_query(raw, client) if prefer_llm else None
    if q is None:
        from .cascade import parse_citation
        q = parse_citation(raw)
        q.parser = "regex"
    q.source_type = classify_source(raw, q.venue, q.doi, q.arxiv_id)
    q.type_via = "rules"
    if q.source_type == UNKNOWN and client is not None:
        q.source_type = classify_llm(raw, client)
        q.type_via = "llm"
    q.url = extract_url(raw) or ""
    q.volume, q.pages = extract_volume_pages(raw)
    log.debug("source_type=%s volume=%r pages=%r", q.source_type, q.volume, q.pages)
    return q
