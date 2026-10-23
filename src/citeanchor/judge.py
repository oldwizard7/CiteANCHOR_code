"""LLM judge for citation/record field disagreements the rules cannot settle."""
from __future__ import annotations

import json
import os
from typing import Optional

from .runlog import get as _get_logger
from .records import FieldMatch, Query, Work

log = _get_logger("judge")

PROMPT = """\
A citation has been matched against a bibliographic record. Deterministic
rules resolved most fields but could not settle {ambiguous}.

CITATION AS PRINTED
  title:   {q_title}
  authors: {q_authors}
  year:    {q_year}
  venue:   {q_venue}
  id:      {q_id}

RETRIEVED RECORD (source: {source})
  title:   {w_title}
  authors: {w_authors}
  year:    {w_year}
  venue:   {w_venue}
  id:      {w_id}

FIELD COMPARISON (computed, not your judgement)
{table}

Decide whether the citation misrepresents this work.

Count as a citation ERROR:
- authors that are not authors of this work, or a substantially different author set
- a title that names a different contribution
- a venue or year the work was never published in

Do NOT count as an error:
- a truncated author list, "et al.", initials vs full given names, name order
- a given name that differs while the surname matches: people publish under
  aliases, English names and different romanisations, so a surname match stands
- a preprint cited as such when a published version also exists, or vice versa
- venue given as an acronym, abbreviation, sub-track, or series name
- a year that differs because preprint and publication differ
- a book cited under another edition, printing or publisher
- missing peripheral metadata

If the retrieved record is simply a *different paper* rather than a bad copy of
this one, answer "different_work" — that is not evidence either way.

Return ONLY JSON:
{{"verdict": "error" | "consistent" | "different_work",
  "offending_fields": ["authors", ...],
  "reason": "<one sentence>"}}"""

# Judge calls per question; the majority verdict is used.
VOTES = 3


def _fmt_table(fm: FieldMatch) -> str:
    author_detail = f"{fm.author_overlap:.2f} of cited authors present"
    if fm.tampered:
        author_detail += ("; SAME SURNAME BUT DIFFERENT GIVEN NAME: "
                          + ", ".join(fm.tampered))
    if getattr(fm, "initial_conflict", None):
        author_detail += ("; GIVEN NAME SHARES NO INITIAL WITH THE RECORD'S AUTHOR OF THAT SURNAME: "
                          + ", ".join(fm.initial_conflict))
    if getattr(fm, "foreign", None):
        author_detail += ("; NOT ON THE RECORD'S COMPLETE AUTHOR LIST: "
                          + ", ".join(fm.foreign))
    rows = [
        ("title", fm.title, f"similarity {fm.title_similarity:.2f}, "
                            f"containment {fm.title_containment:.2f}"),
        ("authors", fm.authors, author_detail),
        ("year", fm.year, ""),
        ("venue", fm.venue, f"score {fm.venue_score:.2f}"),
        ("identifier", fm.identifier, ""),
    ]
    out = []
    for name, verdict, detail in rows:
        state = {True: "agrees", False: "CONFLICTS", None: "undecided"}[verdict]
        if name in fm.ambiguous:
            state = "AMBIGUOUS (between thresholds)"
        out.append(f"  {name:<11} {state:<32} {detail}")
    return "\n".join(out)


# Appended only when the authors are in *hard* conflict.
_HARD_AUTHOR_TRIGGERS = ("exact_title_hard_conflict", "identifier_resolved_field_conflict")
_AUTHOR_ADDENDUM = """
About the author conflict specifically: a preprint and its published version have
the same authors. Do not explain an author conflict by supposing the record is an
earlier or later version of the work with a different author list.
"""


def addendum_for(fields, trigger: str, fm=None) -> str:
    """Extra prompt instructions for this trigger and fields, or ""."""
    out = ""
    if trigger in _HARD_AUTHOR_TRIGGERS and "authors" in (fields or []):
        out += _AUTHOR_ADDENDUM
    return out


def adjudicate_ambiguous(q: Query, w: Work, fm: FieldMatch,
                         client=None, model: Optional[str] = None,
                         *, fields: Optional[list[str]] = None,
                         trigger: str = "ambiguous") -> Optional[dict]:
    """Majority JSON verdict over VOTES judge calls, or None when no judge is available."""
    from .llm import model_name
    model = model or model_name()
    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            log.debug("judge unavailable (no OPENAI_API_KEY); rules decide alone")
            return None
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    disputed = fields or fm.ambiguous
    prompt = PROMPT.format(
        ambiguous=", ".join(disputed) or "one or more fields",
        q_title=q.title or "-", q_authors=", ".join(q.authors) or "-",
        q_year=q.year or "-", q_venue=q.venue or "-",
        q_id=q.doi or q.arxiv_id or "-",
        source=w.source,
        w_title=w.title or "-", w_authors=", ".join(w.authors) or "-",
        w_year=w.year or "-", w_venue=w.venue or "-",
        w_id=w.doi or w.arxiv_id or "-",
        table=_fmt_table(fm),
    )
    extra = addendum_for(disputed, trigger, fm)
    if extra:
        marker = "Return ONLY JSON"
        prompt = (prompt.replace(marker, extra.strip() + "\n\n" + marker, 1)
                  if marker in prompt else prompt + "\n" + extra)
    log.debug("judge asked (trigger=%s, fields=%s) about [%s] %r",
              trigger, disputed, w.source, (w.title or "")[:70])
    from .llm import chat
    votes = []
    for _ in range(VOTES):
        try:
            text = chat(client, prompt).choices[0].message.content or ""
        except Exception as exc:
            log.warning("judge call failed (%s: %s); falling back to rules",
                        type(exc).__name__, exc)
            break
        start, end = text.find("{"), text.rfind("}") + 1
        one = None
        if start != -1 and end > start:
            try:
                one = json.loads(text[start:end])
            except json.JSONDecodeError:
                one = None
        if one and one.get("verdict") in ("error", "consistent", "different_work"):
            one["_raw"] = text.strip()
            votes.append(one)
        else:
            log.warning("judge returned unusable output: %r", text[:200])
    if not votes:
        return None
    tally: dict = {}
    for v in votes:
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    top = max(tally.values())
    # Ties go to the earliest vote.
    winner = next(v["verdict"] for v in votes if tally[v["verdict"]] == top)
    out = next(v for v in votes if v["verdict"] == winner)
    out["_votes"] = [v["verdict"] for v in votes]

    out["_trigger"] = trigger
    out["_fields"] = disputed
    out["_model"] = model
    log.debug("judge -> %s %s: %s", out["verdict"],
              out.get("offending_fields", []), out.get("reason", ""))
    return out
