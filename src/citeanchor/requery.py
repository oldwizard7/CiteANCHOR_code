"""Follow-up query ladder for citations whose first retrieval was empty or weak."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import urlparse, urlunparse

from .runlog import get as _get_logger
from .records import (Query, Work, bare_record, norm_title, same_resource, surnames,
                      title_tokens)

log = _get_logger("requery")


@dataclass
class Budget:
    """Per-citation call limits for the ladder."""
    paid: int = 2          # SerpApi calls
    free: int = 3          # free connector fan-outs
    fetches: int = 1       # direct page fetches

    @classmethod
    def for_correction(cls) -> "Budget":
        """Smaller budget used for the "correction_only" trigger."""
        return cls(paid=1, free=3, fetches=0)


@dataclass
class RecoveryTrace:
    trigger: str = ""
    rungs: list = field(default_factory=list)   # {name, n, n_new, paid, usable, query} or {name, error}
    paid_calls: int = 0
    found: int = 0
    settled_rung: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Value of `extra["via"]` on records added by the ladder.
VIA = "recover"


def mark(w: Work, rung: str) -> Work:
    w.extra = {**(w.extra or {}), "via": VIA, "rung": rung}
    return w


def is_recovered(w: Work) -> bool:
    return (w.extra or {}).get("via") == VIA


# ---------------------------------------------------------------- trigger ---

# Verdict rules meaning that no matching record was found.
_ABSENCE_RULES = {"no_candidates", "below_title_threshold", "url_gone",
                  "judge_different_work", "paper_web_only"}


def trigger(v, res) -> Optional[str]:
    """Reason to run the ladder ("absence", "correction_only" or "weak_real"), or None."""
    rule = (v.rule or "").split("/")[0]
    if rule in _ABSENCE_RULES:
        return "absence"
    if not v.is_real:
        # Not real: run only if no correction was proposed.
        return None if v.correction else "correction_only"
    # The record the verdict rests on, which may differ from `res.best`.
    ev = v.evidence or {}
    if not ev:
        return None
    # Real, but resting on a SerpApi hit without authors.
    if not ev.get("authors") and str(ev.get("source") or "").startswith("serpapi"):
        return "weak_real"
    return None


# ----------------------------------------------------------------- ladder ---

_SUBTITLE = re.compile(r"\s*[:：]\s+")


def _probe(q: Query, title: str, *, authors: Optional[list] = None) -> Query:
    """Copy of `q` with a new title and authors, and no venue."""
    return Query(raw=q.raw, title=title, authors=authors if authors is not None else [],
                 year=q.year, venue="", doi=q.doi, arxiv_id=q.arxiv_id,
                 parser=q.parser, source_type=q.source_type, url=q.url)


# A one-word main title must contain two capitals, a digit or a hyphen.
_DISTINCTIVE = re.compile(r"[A-Z].*[A-Z]|\d|-")


def _main_title(title: str) -> Optional[str]:
    """Text before the subtitle colon if it has 2+ tokens or one distinctive word, else None."""
    head = _SUBTITLE.split(title or "", maxsplit=1)[0].strip()
    if not head or head == (title or "").strip():
        return None
    n = len(title_tokens(head))
    if n >= 2 or (n == 1 and _DISTINCTIVE.search(head)):
        return head
    return None


def _parent_url(url: str) -> Optional[str]:
    """`url` with its last path segment removed, or None if the path has fewer than two."""
    try:
        p = urlparse(url)
    except Exception:
        return None
    parts = [x for x in (p.path or "").split("/") if x]
    if len(parts) < 2:
        return None
    return urlunparse((p.scheme, p.netloc, "/" + "/".join(parts[:-1]) + "/", "", "", ""))


def recovery_eligible(q: Query, w: Work) -> bool:
    """Looser eligibility: same URL/DOI/arXiv id, half the cited title's tokens, or two shared surnames."""
    if q.url and w.url and same_resource(q.url, w.url):
        return True
    if q.doi and w.doi and q.doi.lower() == w.doi.lower():
        return True
    if q.arxiv_id and w.arxiv_id and q.arxiv_id.split("v")[0] == w.arxiv_id.split("v")[0]:
        return True
    a, b = set(title_tokens(q.title)), set(title_tokens(w.title))
    if a and b and len(a & b) / len(a) >= 0.5:
        return True
    return len(surnames(q.authors) & surnames(w.authors)) >= 2


def take_into(res, q: Query, casc, tr: RecoveryTrace, seen: set, works: list, rung: str) -> bool:
    """Add unseen `works` to `res` as candidates marked with `rung`; True if any is usable."""
    fresh = []
    for w in works:
        k = (w.source, norm_title(w.title))
        if k in seen or not w.title:
            continue
        seen.add(k)
        fresh.append(mark(w, rung))
    if not fresh:
        return False
    # Scored directly, without the cascade's title filter.
    from .cascade import Candidate
    from .records import compare
    cands = [Candidate(w, compare(q, w)) for w in fresh]
    res.candidates = list(res.candidates) + cands
    tr.found += len(cands)
    # For a paper, `bare_record` hits do not count as usable.
    if q.source_type == "paper":
        cands = [c for c in cands if not bare_record(c.work)]
    return casc._usable(cands)


def ladder(res, casc, trig: str, *, budget: Budget = Budget()) -> RecoveryTrace:
    """Run the rungs cheapest first until one yields usable candidates; return the trace."""
    q: Query = res.query
    tr = RecoveryTrace(trigger=trig)
    paid = free = fetches = 0
    seen = {(w.work.source, norm_title(w.work.title)) for w in res.candidates}

    def take(works: list, rung: str) -> bool:
        return take_into(res, q, casc, tr, seen, works, rung)

    def rung(name: str, fn) -> bool:
        nonlocal paid, free, fetches
        before = len(res.candidates)
        try:
            works, cost = fn()
        except Exception as exc:
            log.debug("requery: rung %s failed (%s)", name, type(exc).__name__)
            tr.rungs.append({"name": name, "error": type(exc).__name__})
            return False
        paid += cost.get("paid", 0)
        free += cost.get("free", 0)
        fetches += cost.get("fetch", 0)
        tr.paid_calls += cost.get("paid", 0)
        ok = take(works, name)
        tr.rungs.append({"name": name, "n": len(works),
                         "n_new": len(res.candidates) - before,
                         "paid": cost.get("paid", 0), "usable": ok,
                         "query": (res.calls[-1].query if res.calls else "")[:120]})
        if ok:
            tr.settled_rung = name
        return ok

    title = q.title or ""
    if not title and not q.url:
        return tr
    # Paid rungs only when the cascade allows paid calls.
    can_pay = getattr(casc, "use_paid", True)

    # Title only, free connectors.
    if title and free < budget.free:
        if rung("title_only", lambda: (
                casc.free_probe(res, q, _probe(q, title), "recover:title_only",
                                eligible=recovery_eligible), {"free": 1})):
            return tr

    # Main title without the subtitle, free connectors.
    head = _main_title(title)
    if head and free < budget.free:
        if rung("main_title", lambda: (
                casc.free_probe(res, q, _probe(q, head), "recover:main_title",
                                eligible=recovery_eligible), {"free": 1})):
            return tr

    # Parent URL path, for the absence trigger only.
    if q.url and fetches < budget.fetches and trig == "absence":
        parent = _parent_url(q.url)
        if parent:
            def _parent():
                w = casc._fetch_url(res, parent)
                return ([] if w is None or w is casc.GONE else [w]), {"fetch": 1}
            if rung("parent_path", _parent):
                return tr

    # Google Scholar: first-author surname plus main title, filtered by the cited year.
    if title and can_pay and paid < budget.paid:
        first = (q.authors or [""])[0]
        # A corporate author can have no surname key.
        keys = sorted(surnames([first])) if first else []
        last = keys[0] if keys else ""
        qs = (f"author:{last} " if last else "") + (head or title)
        if rung("scholar_author_year", lambda: (
                casc.paid_probe(res, q, "scholar", "recover:scholar_author_year",
                                raw_q=qs, year=q.year, eligible=recovery_eligible),
                {"paid": 1})):
            return tr

    # Google web search, limited to the cited host if any; not used for papers.
    if title and can_pay and paid < budget.paid and q.source_type != "paper":
        site = ""
        if q.url:
            try:
                host = urlparse(q.url).netloc
                site = f" site:{host}" if host else ""
            except Exception:
                site = ""
        if rung("google_title_only", lambda: (
                casc.paid_probe(res, q, "google", "recover:google_title_only",
                                raw_q=f"{title}{site}", eligible=recovery_eligible),
                {"paid": 1})):
            return tr

    return tr

