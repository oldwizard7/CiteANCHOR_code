"""Corrected reference for a citation judged wrong, built from the matched record."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field, replace
from typing import Optional

from .cascade import Candidate
from .records import FieldMatch, Work, norm_title, same_surname

# Minimum title similarity between citation and record for a correction.
MIN_TITLE_SIMILARITY = 0.70


@dataclass
class Correction:
    """Corrected reference fields and how the record was identified."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source: str = ""                 # which connector produced it
    fields_to_fix: list[str] = field(default_factory=list)
    confidence: float = 0.0          # title similarity against the citation
    identified_by: str = ""          # identifier | title+authors | title

    def to_dict(self) -> dict:
        return asdict(self)


def _citation_entry(w: Work) -> bool:
    """Google Scholar entry of type "citation" or without a link."""
    return w.source == "serpapi_scholar" and (
        str((w.extra or {}).get("type") or "").lower() == "citation" or not (w.url or "").strip())


def _anchored(w: Work, m: FieldMatch) -> Optional[str]:
    """How the record is identified: "identifier", "title+authors", "title", or None."""
    if w.doi or w.arxiv_id:
        return "identifier"
    if len(w.authors) >= 2 and m.author_overlap > 0:
        return "title+authors"
    if len(w.authors) >= 2 or w.venue:
        return "title"
    return None


def _same_work(a: Work, b: Work) -> bool:
    """Same normalized title plus a shared DOI, arXiv id or first-author surname."""
    if norm_title(a.title) != norm_title(b.title):
        return False
    if a.doi and b.doi and a.doi.lower() == b.doi.lower():
        return True
    if a.arxiv_id and b.arxiv_id and \
            re.sub(r"v\d+$", "", a.arxiv_id) == re.sub(r"v\d+$", "", b.arxiv_id):
        return True
    return bool(a.authors and b.authors) and same_surname(a.authors[0], b.authors[0])


def _cut_short(short: list, long: list) -> bool:
    """`long` extends `short` with the same leading surnames; all its names are multi-word."""
    return (0 < len(short) < len(long)
            and all(same_surname(x, y) for x, y in zip(short, long))
            and all(len(n.split()) >= 2 for n in long))


def _consolidated(w: Work, pool: Optional[list]) -> Work:
    """Fill a missing venue/year and a truncated author list from same-work records."""
    same = [c.work for c in pool or [] if c.work is not w and _same_work(w, c.work)]
    if not same:
        return w
    venue, year, authors = w.venue, w.year, list(w.authors)
    if not (venue or "").strip():
        donor = next((x for x in same if (x.venue or "").strip()), None)
        if donor is not None:
            venue, year = donor.venue, donor.year or year
    if not year:
        year = next((x.year for x in same if x.year), None)
    longer = [x.authors for x in same if _cut_short(authors, x.authors)]
    if longer:
        authors = list(max(longer, key=len))
    return replace(w, venue=venue, year=year, authors=authors)


def propose(candidate: Optional[Candidate], conflicts: list[str],
            pool: Optional[list] = None) -> Optional[Correction]:
    """The corrected reference, or None when the evidence cannot support one."""
    if candidate is None:
        return None
    w, m = candidate.work, candidate.match
    if not w.title or m.title_similarity < MIN_TITLE_SIMILARITY:
        return None
    if _citation_entry(w):
        return None
    how = _anchored(w, m)
    if how is None:
        return None
    # Require an exact title or author match, not only title similarity.
    if not (m.title is True or m.authors is True):
        return None
    w = _consolidated(w, pool)
    return Correction(
        title=w.title, authors=list(w.authors), year=w.year, venue=w.venue,
        doi=w.doi, arxiv_id=w.arxiv_id, url=w.url, source=w.source,
        fields_to_fix=list(conflicts),
        confidence=round(m.title_similarity, 4),
        identified_by=how,
    )
