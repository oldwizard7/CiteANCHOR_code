"""Real/fake verdicts over cascade evidence, with their basis and deciding rule."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from .runlog import get as _get_logger
from .records import bare_record, same_resource, surnames, given_name, names_match, _corporate
from .source_type import PAPER
from .cascade import Candidate, SearchResult, _select_key
from .judge import adjudicate_ambiguous
from .correct import propose

MAX_SET_ASIDE = 3      # max candidates tried when the judge calls one a different work

log = _get_logger("verdict")

# Minimum title similarity for a near-miss title to count as a conflict.
CONFLICT_TITLE_FLOOR = 0.85


@dataclass
class Verdict:
    is_real: bool
    basis: str                                   # evidence_match | evidence_conflict | no_evidence
    rule: str = ""                               # which branch below decided
    conflicts: list[str] = field(default_factory=list)
    agreements: list[str] = field(default_factory=list)
    confidence: float = 0.0
    tier: str = "none"
    evidence: Optional[dict] = None              # the record the call was made on
    judge: Optional[dict] = None                 # the full judge exchange, if any
    # Proposed corrected metadata; only set when is_real is False.
    correction: Optional[dict] = None
    # Requery record not allowed to decide the verdict, and the rule it reached.
    recovered_evidence: Optional[dict] = None
    recovery_gated: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _absence_verdict(res: SearchResult, rule: str, reason: str,
                     *, confidence: float = 0.0, evidence=None,
                     judge: Optional[dict] = None,
                     may_gate: bool = True) -> Verdict:
    """Fake verdict with basis `no_evidence`; `may_gate` is ignored."""
    return Verdict(is_real=False, basis="no_evidence", rule=rule,
                   tier=res.tier, confidence=confidence,
                   evidence=evidence, judge=judge, reason=reason)


def _decision_pool(res: SearchResult) -> list:
    """Candidates a verdict may rest on: all of them, minus bare records for papers."""
    if getattr(res.query, "source_type", "") != PAPER:
        return list(res.candidates)
    return [c for c in res.candidates if not bare_record(c.work)]


def _best(pool: list) -> Optional[Candidate]:
    """`SearchResult.best`, over a pool rather than every candidate."""
    return max(pool, key=_select_key) if pool else None


def _spelled_out(authors: list[str]) -> int:
    """Count authors whose given name is written out, not initials such as "EG"."""
    return sum(1 for a in authors
               if any(len(t) > 1 and any(ch.islower() for ch in t)
                      for t in (given_name(a) or "").split()))


def _most_informative_witness(cands: list, best, cited_authors: list[str]):
    """Prefer an at-least-as-full exact-title record that confirms the cited authors."""
    if best is None or best.match.title is not True or not cited_authors:
        return best
    cited = surnames(cited_authors)
    pool = [c for c in cands if c.match.title is True
            and (surnames(c.work.authors) & cited)]
    if len(pool) < 2:
        return best
    key = lambda c: (_spelled_out(c.work.authors), len(c.work.authors))
    top = max(pool, key=key)
    # Switch only to a record that confirms the authors when `best` does not.
    if key(top) >= key(best) and top is not best and top.match.authors is True \
            and best.match.authors is not True:
        log.debug("verdict: %s record (%d authors, %d spelled out) confirms the authors; "
                  "it testifies instead of %s", top.work.source, key(top)[1], key(top)[0],
                  best.work.source)
        return top
    return best


def _cross_clear(res: SearchResult) -> None:
    """Drop author mismatches that another exact-title record lists (in place)."""
    exact = [c for c in res.candidates if c.match.title is True and c.work.authors]
    if len(exact) < 2:
        return
    for c in exact:
        m, changed = c.match, False
        for attr in ("foreign", "initial_conflict", "foreign_longer"):
            names = getattr(m, attr, None) or []
            left = [a for a in names
                    if not any(o is not c and any(names_match(a, r) for r in o.work.authors)
                               for o in exact)]
            if len(left) != len(names):
                setattr(m, attr, left)
                changed = True
        base = getattr(m, "authors_base", None)
        if changed and not m.foreign and not m.initial_conflict and base:
            m.authors = base[0]
            if base[1] and "authors" not in m.ambiguous:
                m.ambiguous.append("authors")
            m.notes.append("accusation withdrawn: another record of this work lists the author")
            log.debug("verdict: [%s] accusation withdrawn — another record of the work "
                      "lists the cited author", c.work.source)


def _foreign_consensus(res: SearchResult) -> list[str]:
    """Cited authors missing from two exact-title records of one work from different sources."""
    # Same work: surname sets overlap by Jaccard >= 0.6.
    def same_work(x: Candidate, y: Candidate) -> bool:
        sx, sy = surnames(x.work.authors), surnames(y.work.authors)
        return bool(sx and sy) and len(sx & sy) / len(sx | sy) >= 0.6

    def names(c: Candidate) -> set:
        return set(getattr(c.match, "foreign", None) or []) | set(getattr(c.match, "foreign_longer", None) or [])
    accusers = [c for c in res.candidates if c.match.title is True and names(c)]
    agreed = set()
    for i, x in enumerate(accusers):
        for y in accusers[i + 1:]:
            if x.work.source != y.work.source and same_work(x, y):
                # foreign_longer names count only between records of equal author count.
                if (getattr(x.match, "foreign_longer", None) or getattr(y.match, "foreign_longer", None)) \
                        and len(x.work.authors) != len(y.work.authors):
                    continue
                agreed |= names(x) & names(y)
    return sorted(agreed)


def _conflict_verdict(c: Candidate, res: SearchResult, rule: str,
                      judge: Optional[dict] = None) -> Verdict:
    m = c.match
    return Verdict(
        is_real=False,
        basis="evidence_conflict",
        rule=rule,
        conflicts=m.conflicts,
        agreements=m.agreements,
        confidence=m.title_similarity,
        tier=res.tier,
        evidence=c.work.to_dict(),
        judge=judge,
        reason=("retrieved a record for this work, but "
                + ", ".join(m.conflicts) + " disagree with the citation"),
    )


def _from_judge(res: SearchResult, c: Candidate, out: dict) -> Verdict:
    """Turn a judge answer into a verdict."""
    verdict = out["verdict"]
    if (verdict == "different_work"
            and c.match.title is True and c.match.authors is True):
        # Same title and authors: another version of the same work, so real.
        return Verdict(
            is_real=True, basis="evidence_match", rule="same_work_other_version",
            agreements=c.match.agreements, confidence=c.match.title_similarity,
            tier=res.tier, evidence=c.work.to_dict(), judge=out,
            reason="record has the cited title and authors; the judge's 'different work' "
                   "is another publication of the same work")
    if verdict == "different_work":
        # The record is not evidence about this citation: treat as nothing found.
        return _absence_verdict(
            res, "judge_different_work",
            "judge: retrieved record is a different work",
            confidence=c.match.title_similarity, evidence=c.work.to_dict(),
            judge=out)
    is_error = verdict == "error"
    # A judge "error" on venue, year or identifier alone does not make the citation fake.
    if is_error:
        off = {str(f).lower() for f in (out.get("offending_fields") or [])}
        soft = {"venue", "year", "identifier"}
        # Corporate or anonymous authors are not compared, so authors is soft too.
        if _corporate(res.query.authors or []):
            soft = soft | {"authors"}
        if off and off <= soft:
            return Verdict(
                is_real=True, basis="evidence_match", rule="judge_soft_conflict_only",
                agreements=c.match.agreements, confidence=c.match.title_similarity,
                tier=res.tier, evidence=c.work.to_dict(), judge=out,
                reason="judge objected only to " + ", ".join(sorted(off))
                       + ", which cannot decide on their own")
    return Verdict(
        is_real=not is_error,
        basis="evidence_conflict" if is_error else "evidence_match",
        rule="judge_error" if is_error else "judge_consistent",
        conflicts=out.get("offending_fields", []) if is_error else [],
        agreements=c.match.agreements,
        confidence=c.match.title_similarity,
        tier=res.tier, evidence=c.work.to_dict(), judge=out,
        reason=f"judge: {out.get('reason', '')}".strip(),
    )


def _escalate(res: SearchResult, c: Candidate, client, model, *,
              fields: list[str], trigger: str) -> Optional[Verdict]:
    """Ask the judge; None when it is unavailable or gave nothing usable."""
    out = adjudicate_ambiguous(res.query, c.work, c.match, client, model,
                               fields=fields, trigger=trigger)
    return _from_judge(res, c, out) if out else None


# Deterministic rules allowed to decide on a record found by requery.
_GATED_RULES = frozenset({"exact_title_clean", "url_resolved",
                          "identifier_field_conflict", "identifier_mismatch",
                          "truncated_record_clean", "near_title_conflict"})


def _gate(v: Verdict, v0: Verdict, res: SearchResult) -> Verdict:
    """Return `v` if requery evidence may decide it, else `v0` with that record attached."""
    # A real verdict can be overturned only by a gated rule.
    if (v0.is_real and not v.is_real
            and v.rule not in _GATED_RULES):
        log.debug("requery: second pass ended in %s, which cannot overturn a real verdict; "
                  "keeping it", v.rule)
        v0.recovery_gated = v.rule
        return v0
    ev = v.evidence or {}
    if not ev or (ev.get("extra") or {}).get("via") != "recover":
        return v
    if v.rule in _GATED_RULES:
        log.debug("requery: %r reached a deterministic rule (%s); verdict stands",
                  (ev.get("title") or "")[:60], v.rule)
        return v
    # Records from an agent query must reach a gated rule, whatever the source type.
    if str((ev.get("extra") or {}).get("rung", "")).startswith("agent"):
        log.debug("requery: agent-found record decided by %s, not a gated rule; "
                  "keeping the pre-recovery verdict", v.rule)
        v0.recovered_evidence = ev
        v0.recovery_gated = v.rule
        return v0
    if (v.rule == "judge_consistent"
            and getattr(res.query, "source_type", "") != PAPER):
        log.debug("requery: non-paper citation, judge found the requeried page consistent")
        return v
    log.debug("requery: %r decided by %s, not a gated rule; keeping the "
              "pre-recovery verdict and reporting it as the referent",
              (ev.get("title") or "")[:60], v.rule)
    v0.recovered_evidence = ev
    v0.recovery_gated = v.rule
    return v0


def adjudicate(res: SearchResult, *,
               conflict_floor: float = CONFLICT_TITLE_FLOOR,
               client=None, model: Optional[str] = None,
               use_judge: bool = True,
               casc=None, budget=None) -> Verdict:
    """Verdict for one search result, with a proposed correction and an optional requery pass."""
    def name_it(v: Verdict) -> Verdict:
        if not v.is_real and not v.correction:
            pool = _decision_pool(res)
            c = propose(_best(pool), v.conflicts, pool)
            if c:
                v.correction = c.to_dict()
                log.debug("correction: %r (%s, conf %.2f)",
                          c.title[:70], c.identified_by, c.confidence)
        return v

    v = name_it(_decide(res, conflict_floor=conflict_floor, client=client,
                        model=model, use_judge=use_judge))
    if casc is None:
        return v

    if _primary_witness(res, casc, v):
        v = name_it(_decide(res, conflict_floor=conflict_floor, client=client,
                            model=model, use_judge=use_judge))
    if _version_witness(res, casc, v):
        v = name_it(_decide(res, conflict_floor=conflict_floor, client=client,
                            model=model, use_judge=use_judge))

    from .requery import Budget, ladder, trigger
    t = trigger(v, res)
    if not t:
        return v

    before = len(res.candidates)
    b = budget or (Budget.for_correction() if t == "correction_only" else Budget())
    tr = ladder(res, casc, t, budget=b)
    res.recovery = tr.to_dict()
    res.recovery["verdict_before"] = "real" if v.is_real else "fake"

    if t == "correction_only":
        # The verdict is not re-decided; requery only looks for a correction.
        res.recovery["verdict_after"] = res.recovery["verdict_before"]
    elif len(res.candidates) > before:
        v0 = v
        v2 = name_it(_decide(res, conflict_floor=conflict_floor, client=client,
                             model=model, use_judge=use_judge))
        v = _gate(v2, v0, res)
        res.recovery["verdict_after"] = "real" if v.is_real else "fake"
        res.recovery["gated"] = getattr(v, "recovery_gated", None)
        if v.is_real != v0.is_real:
            log.info("  requery: %s -> %s via %s", res.recovery["verdict_before"],
                     res.recovery["verdict_after"], v.rule)
    else:
        res.recovery["verdict_after"] = res.recovery["verdict_before"]

    # Requery may have added candidates, so try the correction again.
    return name_it(v)


# Sources whose author names come from author profiles, not the byline.
_AGGREGATORS = frozenset({"semanticscholar", "openalex"})
_NAME_RULES = frozenset({"initial_conflict", "foreign_author_consensus"})


def _clears(fresh: list) -> bool:
    """True if any of the fetched records confirms the cited authors."""
    return any(c.match.authors is True for c in fresh)


def _primary_witness(res: SearchResult, casc, v: Verdict) -> bool:
    """Add the arXiv/Crossref record when an author conflict rests on aggregators alone; True if added."""
    if v.is_real:
        return False
    if not (v.rule in _NAME_RULES or (v.rule == "judge_error" and "authors" in (v.conflicts or []))):
        return False
    exact = [c for c in res.candidates if c.match.title is True and c.work.authors]
    if not exact or any(c.work.source not in _AGGREGATORS for c in exact):
        return False
    arx = next((c.work.arxiv_id for c in exact if c.work.arxiv_id), "")
    doi = next((c.work.doi for c in exact if c.work.doi
                and not c.work.doi.lower().startswith("10.48550/")), "")
    works = []
    try:
        if arx:
            works = casc._fetch(res, "arxiv/id", "witness", casc.arxiv.fetch_id, arx, keep_all=True)
        if not works and doi:
            works = casc._fetch(res, "crossref/doi", "witness", casc.crossref.fetch_doi, doi,
                                keep_all=True)
    except Exception as exc:
        log.debug("verdict: primary witness unavailable (%s)", type(exc).__name__)
        return False
    from .records import compare
    fresh = [Candidate(w, compare(res.query, w)) for w in works if w.title and w.authors]
    fresh = [c for c in fresh if c.match.title is True]
    if not _clears(fresh):
        return False
    for c in fresh:
        c.work.extra = {**(c.work.extra or {}), "via_witness": True}
    res.candidates = list(res.candidates) + fresh
    res.notes.append("author conviction rested on aggregator records; fetched the registry "
                     f"record ({fresh[0].work.source}) as a witness")
    log.debug("verdict: fetched [%s] as primary witness before convicting on an author name",
              fresh[0].work.source)
    return True


_AUTHOR_CONVICTIONS = frozenset({"initial_conflict", "foreign_author_consensus",
                                 "exact_title_hard_conflict", "identifier_field_conflict",
                                 "judge_error"})


def _version_witness(res: SearchResult, casc, v: Verdict) -> bool:
    """Add earlier arXiv versions when an author conflict is against an arXiv paper; True if added."""
    if v.is_real:
        return False
    if v.rule not in _AUTHOR_CONVICTIONS or "authors" not in (v.conflicts or []):
        return False
    exact = [c for c in res.candidates if c.match.title is True]
    if any((c.work.extra or {}).get("version") for c in exact):
        return False                                   # versions already in the pool
    arx = res.query.arxiv_id or next((c.work.arxiv_id for c in exact if c.work.arxiv_id), "")
    if not arx:
        return False
    try:
        works = casc._fetch(res, "arxiv/versions", "witness", casc.arxiv.fetch_versions, arx,
                            keep_all=True)
    except Exception as exc:
        log.debug("verdict: arXiv versions unavailable (%s)", type(exc).__name__)
        return False
    from .records import compare
    fresh = [Candidate(w, compare(res.query, w)) for w in works if w.title and w.authors]
    fresh = [c for c in fresh if c.match.title is True]
    if not _clears(fresh):
        return False
    res.candidates = list(res.candidates) + fresh
    res.notes.append(f"author conviction against arXiv:{arx}; earlier versions "
                     f"({', '.join((c.work.extra or {}).get('version', '?') for c in fresh)}) "
                     "fetched as witnesses")
    log.debug("verdict: fetched %d earlier version(s) of arXiv:%s before convicting on an "
              "author name", len(fresh), arx)
    return True


def _neutralise_truncated_doi(res: SearchResult) -> None:
    """If the cited DOI is a proper prefix of a retrieved DOI, clear identifier conflicts."""
    d = (res.query.doi or "").lower().rstrip("/")
    if not d or "/" not in d:
        return
    if not any((c.work.doi or "").lower().startswith(d) and (c.work.doi or "").lower().rstrip("/") != d
               for c in res.candidates):
        return
    for c in res.candidates:
        if c.match.identifier is False:
            c.match.identifier = None
            c.match.ambiguous = [f for f in c.match.ambiguous if f != "identifier"]
    log.debug("verdict: cited DOI %r is a truncated prefix of a retrieved DOI; not compared", d)


def _decide(res: SearchResult, *,
            conflict_floor: float = CONFLICT_TITLE_FLOOR,
            client=None, model: Optional[str] = None,
            use_judge: bool = True) -> Verdict:
    """Decide on the best candidate; if the judge calls it a different work, try the next."""
    import copy
    _neutralise_truncated_doi(res)
    _cross_clear(res)
    pool = _decision_pool(res)
    if res.candidates and not pool:
        page = _best(res.candidates)
        log.debug("verdict: a paper found only as web page titles -> not found")
        return _absence_verdict(
            res, "paper_web_only",
            "only web pages matched, none with an author list; a paper needs a "
            "bibliographic record",
            confidence=page.match.title_similarity, evidence=page.work.to_dict())
    excluded: list = []
    last_aside: Optional[Verdict] = None
    namesake_by_authors = False
    for _ in range(MAX_SET_ASIDE):
        if not excluded and len(pool) == len(res.candidates):
            scope = res
        else:
            scope = copy.copy(res)
            scope.candidates = [c for c in pool if c not in excluded]
        v = _decide_once(scope, conflict_floor=conflict_floor, client=client,
                         model=model, use_judge=use_judge)
        if v.rule != "judge_different_work":
            # Papers: once a same-title record with other authors was set aside,
            # a real verdict needs a record that confirms the authors.
            if (namesake_by_authors and v.is_real and last_aside is not None
                    and getattr(res.query, "source_type", "") == PAPER):
                ev = next((c for c in scope.candidates
                           if c.work.to_dict() == v.evidence), None)
                if ev is not None and ev.match.authors is not True:
                    log.debug("verdict: %r cannot confirm authors that an exact-title record "
                              "contradicted; the row stays unfound", (ev.work.title or "")[:60])
                    return last_aside
            return v
        rejected = next((c for c in scope.candidates
                         if c.work.to_dict() == v.evidence), None)
        if rejected is None or len(scope.candidates) <= 1:
            return v
        if rejected.match.title is True and rejected.match.authors is False:
            namesake_by_authors = True
        last_aside = v
        excluded.append(rejected)
        log.debug("verdict: judge set aside %r as a different work; trying the next candidate",
                  (rejected.work.title or "")[:60])
    return v


def _decide_once(res: SearchResult, *,
                 conflict_floor: float = CONFLICT_TITLE_FLOOR,
                 client=None, model: Optional[str] = None,
                 use_judge: bool = True) -> Verdict:
    """Reduce cascade evidence to a binary real/fake call."""
    # The cited URL does not exist and nothing was retrieved.
    if getattr(res, "url_gone", False) and not res.candidates:
        return Verdict(is_real=False, basis="no_evidence", rule="url_gone",
                       conflicts=["url"], tier=res.tier, confidence=0.9,
                       reason=f"the cited address {res.query.url} does not exist "
                              "(HTTP 404/410 or the host does not resolve) and the "
                              "web index returns nothing for the title either")

    if not res.candidates:
        log.debug("verdict: no candidates at all -> fake (no_evidence)")
        return _absence_verdict(res, "no_candidates",
                                "no bibliographic record found in any tier")

    def ask(c: Candidate, fields: list[str], trigger: str) -> Optional[Verdict]:
        return _escalate(res, c, client, model, fields=fields,
                         trigger=trigger) if use_judge else None

    # Identifier-resolved records: a title or author conflict decides fake.
    # If one has the exact title and no author conflict, non-exact titles are skipped.
    versioned = any(c.match.identifier is True and c.match.title is True
                    and c.match.authors is not False for c in res.candidates)
    for c in res.candidates:
        # The identifier points at an unrelated work.
        if c.match.identifier is True and c.match.title is False and c.match.authors is False:
            continue
        if versioned and c.match.title is not True:
            continue
        if c.match.identifier is True and (c.match.title is False
                                           or c.match.authors is False):
            # The judge is asked only when the title matches exactly.
            if c.match.title is True:
                # Same given-name rule as the exact-title branch.
                if getattr(c.match, "initial_conflict", None):
                    return _conflict_verdict(c, res, "initial_conflict")
                v = ask(c, c.match.conflicts, "identifier_resolved_field_conflict")
                if v:
                    return v
            return _conflict_verdict(c, res, "identifier_field_conflict")

    # Identifier and title both disagree: ask the judge, else fake.
    for c in res.candidates:
        if c.match.identifier is False and c.match.title is False:
            v = ask(c, ["identifier", "title"], "identifier_mismatch")
            return v or _conflict_verdict(c, res, "identifier_mismatch")

    # Non-papers: a record at the cited URL with a compatible title confirms the citation.
    if res.query.url and getattr(res.query, "source_type", "") != PAPER:
        for c in res.candidates:
            if not same_resource(res.query.url, c.work.url):
                continue
            m = c.match
            # Loose title check; year or author conflicts go to the judge.
            compatible = (m.title is True or m.title_containment >= 0.3
                          or m.title_similarity >= 0.3)
            if not compatible:
                log.debug("verdict: page at the cited URL does not match the title "
                          "(sim %.2f) — not treated as resolved", m.title_similarity)
                break
            soft = [f for f in ("year", "authors") if getattr(m, f) is False]
            if soft:
                v = ask(c, soft, "url_resolved_field_conflict")
                if v:
                    return v
            log.debug("verdict: candidate sits at the cited URL %s -> real", res.query.url)
            return Verdict(
                is_real=True, basis="evidence_match", rule="url_resolved",
                agreements=m.agreements, conflicts=m.conflicts,
                confidence=1.0, tier=res.tier, evidence=c.work.to_dict(),
                reason=f"the page at the cited address {res.query.url} matches the citation")

    best = _most_informative_witness(res.candidates, res.best, res.query.authors)
    m = best.match

    if m.title is True:
        # Exact title: fake only if another field contradicts the record.
        # `locator` (volume / first page) does not decide.
        hard = [f for f in ("authors", "identifier") if getattr(m, f) is False]
        if getattr(m, "initial_conflict", None):
            return _conflict_verdict(best, res, "initial_conflict")
        if m.authors is False and _foreign_consensus(res):
            return _conflict_verdict(best, res, "foreign_author_consensus")
        # Extra cited authors decide only when two sources agree on them.
        if (any(getattr(c.match, "foreign_longer", None) for c in res.candidates)
                and _foreign_consensus(res)):
            return _conflict_verdict(best, res, "foreign_author_consensus")
        if hard:
            v = ask(best, hard, "exact_title_hard_conflict")
            return v or _conflict_verdict(best, res, "exact_title_hard_conflict")
        # Venue/year conflicts and ambiguous fields go to the judge; with no answer, real.
        soft = [f for f in ("venue", "year") if getattr(m, f) is False]
        if m.ambiguous or soft:
            v = ask(best, m.ambiguous + soft,
                    "exact_title_soft_conflict" if soft else "exact_title_ambiguous")
            if v:
                return v
        return Verdict(
            is_real=True, basis="evidence_match", rule="exact_title_clean",
            agreements=m.agreements, conflicts=m.conflicts,
            confidence=1.0, tier=res.tier, evidence=best.work.to_dict(),
            reason="exact title match with no contradicting field",
        )

    # `title is None`: the comparison left the title undecided; skip the similarity gate.
    if m.ambiguous and (m.title is None or m.title_similarity >= 0.5):
        # An initial that fits no record author of that surname decides fake.
        if getattr(m, "initial_conflict", None):
            return _conflict_verdict(best, res, "initial_conflict")
        # Hard author/identifier conflicts are sent to the judge too.
        fields = m.ambiguous + [f for f in ("authors", "identifier")
                                if getattr(m, f) is False and f not in m.ambiguous]
        v = ask(best, fields, "near_title_ambiguous")
        if v:
            return v

    if m.title_containment >= 0.95 and m.title_similarity < conflict_floor:
        # One title contains the other (e.g. a truncated page title): decide on the other fields.
        hard = [f for f in ("authors", "identifier") if getattr(m, f) is False]
        if getattr(m, "initial_conflict", None):
            return _conflict_verdict(best, res, "initial_conflict")
        if hard:
            v = ask(best, hard, "truncated_record_hard_conflict")
            return v or _conflict_verdict(best, res, "truncated_record_hard_conflict")
        # Ambiguous fields go to the judge, as in the exact-title branch.
        if m.ambiguous:
            v = ask(best, m.ambiguous, "truncated_record_ambiguous")
            if v:
                return v
        return Verdict(
            is_real=True, basis="evidence_match", rule="truncated_record_clean",
            agreements=m.agreements, conflicts=m.conflicts,
            confidence=m.title_containment, tier=res.tier,
            evidence=best.work.to_dict(),
            reason="retrieved title is contained in the cited title (truncated "
                   "record) with no contradicting field",
        )

    if m.title_similarity >= conflict_floor:
        # Near-miss title against a real record.
        return _conflict_verdict(best, res, "near_title_conflict")

    # Best candidate is below the title threshold: treat as nothing found.
    return _absence_verdict(
        res, "below_title_threshold",
        (f"best candidate has title similarity {m.title_similarity:.2f}, "
         "below the threshold for treating it as the cited work"),
        confidence=m.title_similarity, evidence=best.work.to_dict(),
        may_gate=False)
