"""Deterministic title and author matching that ignores case, punctuation, accents and initials."""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from citeanchor.records import names_match  # noqa: E402

# list-truncation markers such as "et al." and "and others"
_PLACEHOLDER = re.compile(r"^\s*(others|and others|et\.? ?al\.?|\.\.\.|…)\s*$", re.I)


def norm(s: str) -> str:
    """Lowercase, strip accents, and collapse non-alphanumeric runs to single spaces."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def real_authors(authors: list) -> list:
    return [a for a in (authors or []) if a and not _PLACEHOLDER.match(a)]


def strict_title(pred: str, gold: str, *, gold_is_reference: bool = False) -> bool:
    """Equal once normalised; if `gold_is_reference`, `pred` (12+ chars) must occur inside `gold`."""
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    if gold_is_reference:
        return len(p) >= 12 and p in g
    return p == g


def strict_authors(pred: list, gold: list) -> Optional[bool]:
    """True if both lists name the same people; None when gold names no authors."""
    g, p = real_authors(gold), real_authors(pred)
    if not g:
        return None
    if not p:
        return False
    return (all(any(names_match(a, b) for b in p) for a in g)
            and all(any(names_match(a, b) for a in g) for b in p))
