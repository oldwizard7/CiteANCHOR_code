#!/usr/bin/env python3
"""Score proposed sources against gold title, authors, venue and year (rules, then an LLM judge).

    python3 scripts/score_citeresolve.py --run runs/run.json --citeresolve DATA.json --out scores.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from match_rules import strict_authors, strict_title  # noqa: E402

JUDGE_MODEL = "gpt-6-sol"
VOTES = 3
PREPRINT = re.compile(r"arxiv|biorxiv|medrxiv|ssrn|openreview|research ?square|preprint|techrxiv|chemrxiv|psyarxiv|osf preprints|hal\b|zenodo", re.I)
STOP = {"the", "of", "on", "in", "and", "for", "a", "an", "proceedings", "conference", "international",
        "annual", "journal", "transactions", "symposium", "workshop", "th", "st", "nd", "rd", "vol", "volume"}

JUDGE_PROMPT = """Two bibliographic records describe the same citation: the benchmark's GOLD record and a
SYSTEM record that a citation-repair method produced. For each field listed under QUESTIONS, decide
whether the SYSTEM value is a correct value for the gold work.

Correct:
- title: the same work's title, allowing capitalisation, punctuation, a dropped or added subtitle,
  or the title of another version of the same work (e.g. the arXiv title of the published paper).
- authors: the same people as the gold list, allowing initials for given names, transliteration,
  accents, a name order change, and a list cut short with "et al."; NOT a list that adds, drops
  (beyond an "et al." cut) or substitutes people.
- venue: a venue where some real version of this work appeared (a preprint server and the published
  venue are both correct), or another name / abbreviation of the gold venue.
- year: the year of some real version of this work (preprint and publication years may differ).
Wrong: a different work, other people, or a venue / year in which no version of this work appeared.

GOLD
  title:   {g_title}
  authors: {g_authors}
  venue:   {g_venue}
  year:    {g_year}

SYSTEM
  title:   {p_title}
  authors: {p_authors}
  venue:   {p_venue}
  year:    {p_year}

QUESTIONS: {fields}

Return ONLY JSON with one key per field asked, each "ok" or "wrong", plus "reason" (one sentence).
Example: {{"venue": "ok", "year": "wrong", "reason": "..."}}"""


# ----------------------------------------------------------------- field rules ---

def _vtokens(v: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (v or "").lower()) if w not in STOP and len(w) > 1}


def _acronym(v: str) -> str:
    return "".join(w[0] for w in re.findall(r"[a-z]+", (v or "").lower()) if w not in STOP)


def venue_rule(pred: str, gold: str) -> Optional[bool]:
    """True/False when clear, None to ask the judge."""
    p, g = (pred or "").strip(), (gold or "").strip()
    if not g:
        return None                                   # caller treats as not scored
    if not p:
        return False                                  # gold has a venue, the prediction does not
    pl, gl = re.sub(r"\s+", " ", p.lower()), re.sub(r"\s+", " ", g.lower())
    if pl == gl or pl in gl or gl in pl:
        return True
    if bool(PREPRINT.search(p)) != bool(PREPRINT.search(g)):
        return True                                   # preprint and published version of one work
    tp, tg = _vtokens(p), _vtokens(g)
    if tp and tg and (tp <= tg or tg <= tp):
        return True
    ap, ag = _acronym(p), _acronym(g)
    for short, long_ in ((p, g), (g, p)):
        s = re.sub(r"[^A-Za-z]", "", short)
        if 2 <= len(s) <= 8 and s.isupper() and s.lower() in (_acronym(long_) or ""):
            return True
    return None


def year_rule(pred, gold) -> Optional[bool]:
    if gold in (None, "", "N/A"):
        return None
    if pred in (None, "", "N/A"):
        return False
    try:
        d = abs(int(str(pred)[:4]) - int(str(gold)[:4]))
    except ValueError:
        return None
    if d <= 1:
        return True
    return None                                       # 2+ years apart: left to the judge


# --------------------------------------------------------------------- judge ---

class Judge:
    def __init__(self, cache_path: Path):
        self.cache_path, self.lock, self.cache = cache_path, threading.Lock(), {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line); self.cache[d["key"]] = d
                except Exception:
                    pass
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(cache_path, "a", encoding="utf-8")
        self.calls = self.cached = 0
        self.cache_only, self.missed = False, []
        self._client = None

    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def ask(self, gold: dict, pred: dict, fields: list) -> dict:
        prompt = JUDGE_PROMPT.format(
            g_title=gold.get("title") or "-", g_authors=", ".join(gold.get("authors") or []) or "-",
            g_venue=gold.get("venue") or "-", g_year=gold.get("year") or "-",
            p_title=pred.get("title") or "-", p_authors=", ".join(pred.get("authors") or []) or "-",
            p_venue=pred.get("venue") or "-", p_year=pred.get("year") or "-",
            fields=", ".join(fields))
        key = hashlib.sha1((JUDGE_MODEL + "\n" + prompt).encode("utf-8")).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cached += 1
                return self.cache[key]["verdict"]
        if self.cache_only:                 # record the miss instead of calling the API
            with self.lock:
                self.missed.append({"gold": gold.get("title"), "pred": pred.get("title"), "fields": fields})
            return {f: None for f in fields}
        votes = []
        for _ in range(VOTES):
            try:
                r = self.client().chat.completions.create(
                    model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096)
                t = r.choices[0].message.content or ""
                d = json.loads(t[t.find("{"): t.rfind("}") + 1])
                votes.append({f: d.get(f) for f in fields} | {"reason": d.get("reason", "")})
            except Exception as exc:                  # failed votes are excluded from the majority
                votes.append({"error": str(exc)[:200]})
            good = [v for v in votes if "error" not in v]
            if len(good) >= 2 and all(good[0][f] == good[1][f] for f in fields):
                break
        good = [v for v in votes if "error" not in v]
        verdict = {}
        for f in fields:
            tally = collections.Counter(v.get(f) for v in good)
            verdict[f] = tally.most_common(1)[0][0] if tally else None
        rec = {"key": key, "model": JUDGE_MODEL,
               "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
               "fields": fields, "votes": votes, "verdict": verdict}
        with self.lock:
            self.calls += len(votes)
            self.cache[key] = rec
            self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.fh.flush()
        return verdict


# ---------------------------------------------------------------- predictions ---

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _key(citation: str) -> str:
    """SHA-1 of the whitespace-normalised, lowercased citation (the `citation_sha1` field)."""
    return hashlib.sha1(_norm(citation).encode("utf-8")).hexdigest()


def canon(c: dict) -> dict:
    """Map a data row's `label`/`source` keys to the `tag`/`gold` keys used here."""
    c = dict(c)
    if "tag" not in c:
        c["tag"] = "gold" if c.get("label") == "Recoverable" else "fabricated"
    if "gold" not in c and "source" in c:
        c["gold"] = c["source"]
    return c


def preds_from_run(run_path: Path, cf_rows: list) -> dict:
    run = json.load(open(run_path))["results"]
    by_key = {_key(e["citation"]): e for e in run}
    out = {}
    for cf in cf_rows:
        e = by_key.get(cf.get("citation_sha1") or _key(cf.get("citation", "")))
        if e is None:
            continue
        flagged = not e["Pred"]
        c = ((e.get("verification") or {}).get("correction") or {}) if flagged else {}
        work = None
        if flagged and (c.get("title") or "").strip():
            work = {k: c.get(k) for k in ("title", "authors", "venue", "year")}
        out[cf["id"]] = {"flagged": flagged, "work": work}
    return out


def preds_from_jsonl(path: Path) -> dict:
    """Predictions from JSONL lines {"id", "flagged": bool, "work": {title, authors, venue, year} | null}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["id"]] = {"flagged": bool(d.get("flagged")), "work": d.get("work")}
    return out


# -------------------------------------------------------------------- scoring ---

def score(cf_rows: list, preds: dict, judge: Judge, workers: int = 6) -> tuple[list, dict]:
    rows = []

    def one(cf):
        p = preds.get(cf["id"], {"flagged": False, "work": None})
        rec = {"id": cf["id"], "tag": cf["tag"], "flagged": p["flagged"],
               "named": bool(p["work"]), "fields": {}, "decided_by": {}}
        if cf["tag"] != "gold":
            rec["abstain"] = p["flagged"] and not p["work"]
            rec["joint"] = rec["abstain"]
            return rec
        if not (p["flagged"] and p["work"]):
            rec.update(source=False, repair=False, joint=False)
            return rec
        gold = dict(cf.get("gold") or {})
        no_struct = not gold.get("title")
        if no_struct:
            gold["title"] = cf.get("corrected_citation") or ""
        w = p["work"]
        verdict, ask = {}, []
        t = strict_title(w.get("title") or "", gold["title"], gold_is_reference=no_struct)
        verdict["title"] = True if t else None
        a = strict_authors(w.get("authors") or [], gold.get("authors") or [])
        verdict["authors"] = a if a is not None and a else (None if a is False else "n/a")
        verdict["venue"] = "n/a" if not (gold.get("venue") or "").strip() else venue_rule(w.get("venue"), gold.get("venue"))
        verdict["year"] = "n/a" if gold.get("year") in (None, "", "N/A") else year_rule(w.get("year"), gold.get("year"))
        for f, v in verdict.items():
            rec["decided_by"][f] = "rule" if v is not None else "judge"
            if v is None:
                ask.append(f)
        if ask:
            j = judge.ask(gold, w, ask)
            for f in ask:
                verdict[f] = (j.get(f) == "ok")
        rec["fields"] = verdict
        scored = [v for v in verdict.values() if v != "n/a"]
        rec["source"] = verdict["title"] is True
        rec["repair"] = all(v is True for v in scored)
        rec["joint"] = rec["repair"]
        return rec

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, cf_rows))
    g = [r for r in rows if r["tag"] == "gold"]; f = [r for r in rows if r["tag"] != "gold"]
    pct = lambda n, d: round(100 * n / d, 1) if d else None
    ident = sum(1 for r in g if r["flagged"] and r["named"])
    m = {
        # headline rates (percent)
        "Identified_recoverable": pct(ident, len(g)),              # flagged with a proposed source / recoverable
        "Recovered": pct(sum(r["repair"] for r in g), ident),      # proposals matching the gold source / proposals
        "Identified_unrecoverable": pct(sum(r["abstain"] for r in f), len(f)),   # flagged with None / unrecoverable
        "Task_success": pct(sum(r["joint"] for r in rows), len(rows)),
        # the same outcomes as shares of all rows of each class
        "Source": pct(sum(r["source"] for r in g), len(g)),        # recoverable rows whose proposed title is the gold title
        "Repair": pct(sum(r["repair"] for r in g), len(g)),        # recoverable rows repaired on every scored field
        "Abstain": pct(sum(r["abstain"] for r in f), len(f)),
        "Detected": pct(sum(r["flagged"] for r in rows), len(rows)),
        "Joint": pct(sum(r["joint"] for r in rows), len(rows)),
        "counts": {"recoverable": len(g), "fabricated": len(f),
                   "source": sum(r["source"] for r in g), "repair": sum(r["repair"] for r in g),
                   "abstain": sum(r["abstain"] for r in f), "detected": sum(r["flagged"] for r in rows),
                   "joint": sum(r["joint"] for r in rows)},
        "per_field_on_named_gold": {
            fld: dict(collections.Counter(
                ("n/a" if r["fields"][fld] == "n/a" else ("ok" if r["fields"][fld] else "wrong"))
                + "/" + r["decided_by"][fld] for r in g if r["fields"])) for fld in ("title", "authors", "venue", "year")},
    }
    return rows, m


def validate(cf_rows: list, preds: dict, rows: list, judge: Judge, n: int = 100, seed: int = 0) -> dict:
    """Agreement and Cohen's kappa between the judge and the rules on a sample of rule-decided fields."""
    by_id = {r["id"]: r for r in rows}
    pool = []
    for cf in cf_rows:
        r = by_id[cf["id"]]
        if cf["tag"] != "gold" or not r["fields"]:
            continue
        for f in ("title", "authors", "venue", "year"):
            if r["decided_by"][f] == "rule" and r["fields"][f] != "n/a":
                pool.append((cf, f, bool(r["fields"][f])))
    random.Random(seed).shuffle(pool)
    wrong = [x for x in pool if not x[2]]; ok = [x for x in pool if x[2]]
    k = min(len(wrong), n * 3 // 10)                  # at most 30% rule-"wrong", the rest rule-"ok"
    sample = wrong[:k] + ok[:n - k]
    def one(item):
        cf, f, rule_ok = item
        gold = dict(cf.get("gold") or {})
        if not gold.get("title"):
            gold["title"] = cf.get("corrected_citation") or ""
        j = judge.ask(gold, preds[cf["id"]]["work"], [f])
        return (rule_ok, j.get(f) == "ok")
    with ThreadPoolExecutor(6) as ex:
        agree = collections.Counter(ex.map(one, sample))
    tot = sum(agree.values())
    po = (agree[(True, True)] + agree[(False, False)]) / tot if tot else 0
    pr = (agree[(True, True)] + agree[(True, False)]) / tot if tot else 0
    pj = (agree[(True, True)] + agree[(False, True)]) / tot if tot else 0
    pe = pr * pj + (1 - pr) * (1 - pj)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": tot, "agreement": round(po, 3), "kappa": round(kappa, 3),
            "table": {f"rule_{'ok' if a else 'wrong'}_judge_{'ok' if b else 'wrong'}": v for (a, b), v in agree.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path)
    src.add_argument("--preds", type=Path)
    ap.add_argument("--citeresolve", type=Path, required=True, help="the CiteResolve benchmark file")
    ap.add_argument("--cache", type=Path, default=Path("runs/citeresolve_judge_cache.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-only", action="store_true",
                    help="never call the judge: an uncached field is counted as a miss and scored as wrong")
    args = ap.parse_args()
    cf = json.loads(args.citeresolve.read_text(encoding="utf-8"))
    cf = [canon(c) for c in (cf["data"] if isinstance(cf, dict) else cf)]
    preds = preds_from_run(args.run, cf) if args.run else preds_from_jsonl(args.preds)
    judge = Judge(args.cache)
    judge.cache_only = args.cache_only
    rows, m = score(cf, preds, judge, args.workers)
    v = validate(cf, preds, rows, judge, args.validate) if args.validate else None
    report = {"source": str(args.run or args.preds), "judge_model": JUDGE_MODEL, "metrics": m,
              "judge_validation": v, "judge_calls": judge.calls, "judge_cached": judge.cached, "judge_cache_misses": judge.missed,
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("Identified_recoverable", "Recovered", "Identified_unrecoverable", "Task_success")}))
    print(json.dumps({k: m[k] for k in ("Source", "Repair", "Abstain", "Detected", "Joint")}))
    print("counts:", m["counts"])
    print("per field on named recoverable rows:", json.dumps(m["per_field_on_named_gold"], ensure_ascii=False))
    print("judge validation:", v)
    print(f"judge calls {judge.calls}, cached {judge.cached}, cache misses {len(judge.missed)}; wrote {args.out}")


if __name__ == "__main__":
    main()
