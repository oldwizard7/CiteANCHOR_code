"""Tiered retrieval cascade: local cache, cited URL, identifiers, free APIs, then paid search."""
from __future__ import annotations

import concurrent.futures
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from .runlog import get as _get_logger
from .acl_mirror import ACLMirror
from .cache import RecordCache
from .connectors import (Arxiv, CommonCrawl, Crossref, DataCite, DBLP, EuropePMC, GoogleBooks,
                         OpenAlex, PubMed, SemanticScholar, SerpApi, Wayback)
from .normalize import normalize_title
from .records import (same_resource, FieldMatch, Query, Work, bare_record, compare,
                      edit_distance_at_most, extract_arxiv_id, extract_doi, norm_title,
                      surnames, title_tokens)

log = _get_logger("cascade")

# Minimum title similarity for a candidate to count as usable evidence.
TITLE_EVIDENCE_THRESHOLD = 0.75


@dataclass
class Candidate:
    work: Work
    match: FieldMatch

    def to_dict(self) -> dict:
        return {"work": self.work.to_dict(), "match": self.match.to_dict()}


@dataclass
class ConnectorCall:
    """Log entry for one request to one source."""
    connector: str
    tier: str
    query: str = ""
    n_results: int = 0
    n_eligible: int = 0          # results that passed the title filter
    elapsed_ms: int = 0
    error: str = ""
    # False for local sources (cache, ACL mirror); not counted in connector_calls.
    remote: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _registry_doi(doi: str) -> str:
    """DOI taken from a URL, without rendering suffixes such as .full.pdf or /abstract."""
    d = re.sub(r"(\.full)?(\.pdf|\.abstract|\.full-text)?$", "", (doi or "").strip(), flags=re.I)
    d = re.sub(r"/(full|abstract|pdf|epdf)$", "", d, flags=re.I)
    # Strip a version suffix such as v2 only for bioRxiv (10.1101) DOIs.
    if d.startswith("10.1101/"):
        d = re.sub(r"v\d+$", "", d)
    return d


def page_work(url: str, text: str, *, source: str = "url", extra: Optional[dict] = None) -> Work:
    """Work built from an HTML page's title and scholarly meta tags."""
    import html as _html
    # Some pages embed hundreds of KB of script before <title>.
    head = (text or "")[:600000]
    m = re.search(r"<title[^>]*>(.*?)</title>", head, re.S | re.I)
    title = _html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
    # First matching <meta> content, with either attribute order.
    def meta(*names):
        for n in names:
            for pat in (r'<meta[^>]+(?:property|name)=["\']' + re.escape(n) + r'["\'][^>]+content=["\']([^"\']+)',
                        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']' + re.escape(n) + r'["\']'):
                mm = re.search(pat, head, re.I)
                if mm:
                    return mm.group(1)
        return ""
    # Fall back to meta title tags, then to the first <h1>.
    if not title:
        title = _html.unescape(meta("og:title", "twitter:title", "citation_title", "dc.title")).strip()
    if not title:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", head, re.S | re.I)
        title = _html.unescape(re.sub(r"<[^>]+>|\s+", " ", h1.group(1)).strip()) if h1 else ""
    # Drop a trailing " - Site name" suffix.
    title = re.sub(r"\s+[-|–—]\s+[^-|–—]{1,40}$", "", title).strip()
    date = meta("article:published_time", "citation_publication_date", "citation_date",
                "datePublished", "og:article:published_time", "date", "dc.date")
    ym = re.search(r"\b(19[89]\d|20[0-4]\d)\b", date)
    # Scholarly author tags only, not the generic "author" tag.
    auth = meta("citation_author", "dc.creator")
    authors = [a.strip() for a in re.split(r",|;| and ", auth) if a.strip()] if auth else []
    # Keep the requested URL, not the final URL after redirects.
    return Work(title=title, url=url, source=source,
                year=int(ym.group(1)) if ym else None, authors=authors,
                extra={**(extra or {}), "date_meta": date})


def _select_key(c: Candidate):
    """Sort key for choosing the best candidate (larger is better)."""
    key = (c.match.title is True, c.match.title_similarity,
           c.match.year is not False, c.match.author_overlap)
    return key + (-_SOURCE_RANK.get(c.work.source, 20),)


# Source order for the final tie-break in `_select_key`; lower is preferred.
_SOURCE_RANK = {"crossref": 0, "arxiv": 1, "aclanthology": 2, "acl_mirror": 2, "pubmed": 3,
                "europepmc": 4, "datacite": 5, "openalex": 6, "semanticscholar": 7, "dblp": 8,
                "googlebooks": 9, "serpapi_scholar": 10, "serpapi": 11, "url": 12,
                "wayback": 13, "commoncrawl": 13, "cache": 14}


@dataclass
class SearchResult:
    query: Query
    candidates: list[Candidate] = field(default_factory=list)
    tier: str = "none"                       # which tier produced the evidence
    tiers_tried: list[str] = field(default_factory=list)
    calls: list[ConnectorCall] = field(default_factory=list)
    paid_calls: int = 0
    notes: list[str] = field(default_factory=list)
    # True if the cited URL answered 404/410 or its host does not resolve.
    url_gone: bool = False
    # RecoveryTrace dict from `requery.ladder`, if it ran.
    recovery: Optional[dict] = None

    @property
    def connector_calls(self) -> int:
        return sum(1 for c in self.calls if c.remote)

    @property
    def best(self) -> Optional[Candidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=_select_key)

    @property
    def best_index(self) -> Optional[int]:
        if not self.candidates:
            return None
        return max(range(len(self.candidates)),
                   key=lambda i: _select_key(self.candidates[i]))

    def record(self, connector: str, tier: str, **kw) -> ConnectorCall:
        call = ConnectorCall(connector=connector, tier=tier, **kw)
        self.calls.append(call)
        log.debug("  call %-16s tier=%-10s n=%-2d eligible=%-2d %4dms %s",
                  connector, tier, call.n_results, call.n_eligible,
                  call.elapsed_ms, call.error)
        return call

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict(),
            "tier": self.tier,
            "tiers_tried": self.tiers_tried,
            "connector_calls": self.connector_calls,
            "paid_calls": self.paid_calls,
            "url_gone": self.url_gone,
            "recovery": self.recovery,
            "notes": self.notes,
            "calls": [c.to_dict() for c in self.calls],
            "best_index": self.best_index,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ------------------------------------------------------------------- parsing ---

_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")


def parse_citation(raw: str) -> Query:
    """Rule-based parse of a raw citation string into a Query."""
    q = Query(raw=raw)
    q.doi = extract_doi(raw) or ""
    q.arxiv_id = extract_arxiv_id(raw) if "arxiv" in raw.lower() else ""
    years = _YEAR.findall(raw)
    q.year = int(years[-1]) if years else None

    # Drop URLs and arXiv ids, then split on sentence ends: authors, title, venue.
    body = re.sub(r"https?://\S+", " ", raw)
    body = re.sub(r"\barXiv:\s*\S+", " ", body, flags=re.I)
    parts = [p.strip() for p in re.split(r"(?<=[a-z0-9\)])\.\s+", body) if p.strip()]
    if len(parts) >= 2:
        q.authors = [a.strip() for a in re.split(r",| and ", parts[0]) if a.strip()]
        q.title = parts[1].rstrip(".")
        if len(parts) >= 3:
            q.venue = parts[2].rstrip(".")
    elif parts:
        q.title = parts[0].rstrip(".")
    return q


# ------------------------------------------------------------------- cascade ---

class SearchCascade:
    def __init__(self, *, cache: Optional[RecordCache] = None,
                 use_paid: bool = True, rows: int = 5, workers: int = 6,
                 acl: Optional[ACLMirror] = None, fanout_timeout: float = 25.0):
        self.cache = cache if cache is not None else RecordCache()
        # Offline ACL Anthology index; unused until scripts/build_acl_mirror.py --build has run.
        self.acl = acl if acl is not None else ACLMirror()
        self.rows = rows
        self.workers = workers
        self.use_paid = use_paid
        self.fanout_timeout = fanout_timeout
        self.crossref = Crossref()
        self.datacite = DataCite()
        self.arxiv = Arxiv()
        self.serp = SerpApi()
        self.free = [self.crossref, OpenAlex(), self.arxiv, EuropePMC(),
                     SemanticScholar(), DBLP(), PubMed()]
        # Google Books only when GOOGLE_BOOKS_API_KEY is set.
        self.books = GoogleBooks() if GoogleBooks.enabled() else None
        if self.books is not None:
            self.free.append(self.books)
        # Wayback Machine fallback for dead URLs; disabled if unreachable at start-up.
        self.wayback = None
        if Wayback.enabled():
            wb = Wayback()
            if wb.reachable():
                self.wayback = wb
            else:
                log.warning("wayback: archive.org unreachable from this host; archived-page "
                            "fallback disabled for this run")
        # Common Crawl as a second archive, probed the same way.
        self.commoncrawl = None
        if CommonCrawl.enabled():
            cc = CommonCrawl()
            if cc.reachable():
                self.commoncrawl = cc
            else:
                log.warning("commoncrawl: index unreachable from this host; disabled for this run")
        self._available: Optional[list] = None
        self._lock = threading.Lock()

    def probe(self) -> dict:
        """Health-check the free connectors and put failing ones last; returns {name: ok}."""
        status = {}
        if not self.free:
            with self._lock:
                self._available = []
            return status
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.free)) as ex:
            futs = {ex.submit(c.healthy): c for c in self.free}
            for f in concurrent.futures.as_completed(futs):
                c = futs[f]
                try:
                    status[c.name] = f.result()
                except Exception:
                    status[c.name] = False
        with self._lock:
            up = [c for c in self.free if status.get(c.name)]
            down = [c for c in self.free if not status.get(c.name)]
            if down:
                log.warning("probe failed for %s — kept, but tried last",
                            [c.name for c in down])
            self._available = up + down
        return status

    @property
    def available(self) -> list:
        if self._available is None:
            self.probe()
        return self._available

    # Keyword hints for `_route`; arXiv and the biomedical APIs are queried only on a match.
    _ARXIV_HINT = re.compile(r"arxiv|preprint|corr\b", re.I)
    _BIOMED_HINT = re.compile(
        r"pubmed|medline|lancet|nature medicine|bioinformatics|genom|cell |"
        r"clinical|patient|cancer|protein|biolog|medic|health|nejm|jama",
        re.I)

    def _route(self, q: Query) -> list:
        """Free connectors to query for `q`: not cooling down, filtered by kind and keyword hints."""
        blob = f"{q.raw} {q.venue}"
        picked = []
        for c in self.available:
            if c.cooling():
                continue
            if c.name == "arxiv" and not (q.arxiv_id or self._ARXIV_HINT.search(blob)):
                continue
            if c.name in ("europepmc", "pubmed") and not self._BIOMED_HINT.search(blob):
                continue
            # Google Books only for books and reports.
            if c.name == "googlebooks" and getattr(q, "source_type", "") not in ("book", "report"):
                continue
            picked.append(c)
        # If nothing passes the filters, use every connector not cooling down.
        return picked or [c for c in self.available if not c.cooling()]

    # Maximum edit distance between normalized titles for a non-web record to be eligible.
    MAX_TITLE_EDITS = 10

    @classmethod
    def _eligible(cls, q: Query, w: Work) -> bool:
        if w.source.startswith("serpapi"):
            return True
        # Same URL, arXiv id or DOI: eligible whatever the title.
        if q.url and same_resource(q.url, w.url):
            return True
        if q.arxiv_id and w.arxiv_id and \
                q.arxiv_id.split("v")[0] == w.arxiv_id.split("v")[0]:
            return True
        if q.doi and w.doi and q.doi.lower() == w.doi.lower():
            return True
        a, b = normalize_title(q.title), normalize_title(w.title)
        if not a or not b:
            return False
        if abs(len(a) - len(b)) > cls.MAX_TITLE_EDITS:
            # Length gap beyond the edit bound: accept only if one title contains the other.
            return b in a or a in b
        return edit_distance_at_most(a, b, cls.MAX_TITLE_EDITS)

    # HTTP statuses taken as "resource gone"; other failures are inconclusive.
    _GONE = (404, 410)
    # Sentinel for a gone resource; None means undetermined.
    GONE = object()

    def _fetch_url(self, res: SearchResult, url: str):
        """Fetch the cited URL: Work on 200, GONE on 404/410 or unresolvable host, else None."""
        import requests
        from .connectors import UA, TIMEOUT
        t0 = time.perf_counter()
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        except requests.exceptions.ConnectionError as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            dead = ("nodename nor servname" in str(exc) or "Name or service not known" in str(exc)
                    or "NameResolutionError" in str(exc))
            res.record("url", "url", query=url, elapsed_ms=ms,
                       error=("host does not resolve" if dead else f"unreachable: {exc}"))
            res.notes.append(f"{url} {'does not resolve' if dead else 'was unreachable'}")
            return self.GONE if dead else None
        except Exception as exc:
            res.record("url", "url", query=url,
                       elapsed_ms=int((time.perf_counter() - t0) * 1000),
                       error=f"{type(exc).__name__}: {exc}")
            return None
        ms = int((time.perf_counter() - t0) * 1000)
        if r.status_code in self._GONE:
            res.record("url", "url", query=url, elapsed_ms=ms,
                       error=f"HTTP {r.status_code} — not found")
            res.notes.append(f"{url} returns HTTP {r.status_code}")
            return self.GONE
        if r.status_code != 200:
            res.record("url", "url", query=url, elapsed_ms=ms,
                       error=f"HTTP {r.status_code} — inconclusive")
            res.notes.append(f"{url} answered HTTP {r.status_code}; existence undecided")
            return None
        res.record("url", "url", query=url, n_results=1, n_eligible=1, elapsed_ms=ms)
        return page_work(url, r.text, source="url",
                         extra={"http": r.status_code, "final_url": r.url})

    def _fetch_archived(self, res: SearchResult, url: str, year):
        """Archived copy of `url` near `year` (Wayback, then Common Crawl) as a Work, or None."""
        for arch in (self.wayback, self.commoncrawl):
            if arch is None:
                continue
            t0 = time.perf_counter()
            snap = arch.snapshot(url, year)
            if not snap:
                res.record(arch.name, "url", query=url, n_results=0, n_eligible=0,
                           elapsed_ms=int((time.perf_counter() - t0) * 1000))
                continue
            r = arch.page(snap["url"] if arch.name == "wayback" else snap)
            ms = int((time.perf_counter() - t0) * 1000)
            if not r:
                res.record(arch.name, "url", query=url, n_results=0, n_eligible=0, elapsed_ms=ms,
                           error="capture listed but not served")
                continue
            text = getattr(r, "archived_text", None) or r.text
            res.record(arch.name, "url", query=url, n_results=1, n_eligible=1, elapsed_ms=ms)
            res.notes.append(f"{url} is archived ({arch.name}); capture {snap['timestamp'][:8]} used")
            log.debug("url: live fetch failed; archived copy (%s, %s) stands in",
                      arch.name, snap["timestamp"][:8])
            return page_work(url, text, source=arch.name,
                             extra={"http": r.status_code, "snapshot": snap["timestamp"],
                                    "archived_url": snap.get("url") if arch.name == "wayback"
                                    else snap.get("filename")})
        return None

    def _free_tier(self, res: SearchResult, q: Query, strict: bool = False) -> bool:
        """Tier 2 fan-out over free connectors; True if it settled `res` (top 10 kept as candidates)."""
        conns = self._route(q)
        if conns:
            res.tiers_tried.append("free")
            skipped = [c.name for c in self.available if c not in conns]
            log.debug("fan-out to %s (skipped %s)", [c.name for c in conns], skipped)
            author = q.authors[0] if q.authors else ""
            found: list[Work] = []
            starts: dict = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {}
                for c in conns:
                    futs[ex.submit(c.search, q.title, author, self.rows)] = c
                    starts[c.name] = time.perf_counter()
                try:
                    # One deadline for the whole fan-out; unfinished calls are cancelled.
                    for f in concurrent.futures.as_completed(futs, timeout=self.fanout_timeout):
                        c = futs[f]
                        ms = int((time.perf_counter() - starts[c.name]) * 1000)
                        try:
                            works = f.result() or []
                        except Exception as exc:
                            res.record(c.name, "free", query=q.title, elapsed_ms=ms,
                                       error=f"{type(exc).__name__}: {exc}")
                            res.notes.append(f"{c.name} failed: {exc}")
                            continue
                        eligible = [w for w in works if self._eligible(q, w)]
                        res.record(c.name, "free", query=q.title, n_results=len(works),
                                   n_eligible=len(eligible), elapsed_ms=ms)
                        for w in works:
                            if w not in eligible:
                                log.debug("    dropped [%s] %r (title too far from query)",
                                          c.name, (w.title or "")[:70])
                        found.extend(eligible)
                except concurrent.futures.TimeoutError:
                    slow = [c.name for f, c in futs.items() if not f.done()]
                    res.notes.append(f"fan-out deadline hit; still pending: {slow}")
                    for f, c in futs.items():
                        if not f.done():
                            res.record(c.name, "free", query=q.title,
                                       elapsed_ms=int(self.fanout_timeout * 1000),
                                       error="fan-out deadline")
                        f.cancel()
            if found:
                cands = [Candidate(w, compare(q, w)) for w in found]
                cands.sort(key=lambda c: -c.match.title_similarity)
                # In strict mode (books) only a strong match settles.
                if self._usable(cands) and (not strict or self._strong(cands)):
                    self._remember(q, [c.work for c in cands[:10]])
                    res.candidates, res.tier = cands[:10], "free"
                    return True
                res.candidates = cands[:10]

        return False

    def _paid_engine(self, res: SearchResult, q: Query, engine: str) -> bool:
        """One SerpApi search ("scholar" or "google"); True if its results are usable."""
        res.tiers_tried.append(f"paid:{engine}")
        author = q.authors[0] if q.authors else ""
        res.paid_calls += 1
        fn = self.serp.search_scholar if engine == "scholar" else self.serp.search
        works = self._fetch(res, "serpapi_scholar" if engine == "scholar" else "serpapi",
                            "paid", fn, q.title, author, self.rows)
        if not works:
            return False
        cands = [Candidate(w, compare(q, w)) for w in works]
        cands.sort(key=lambda c: -c.match.title_similarity)
        if engine == "google":
            res.notes.append("evidence is web search only — no structured record")
            cands = self._resolve_web_ids(res, q, cands) + cands
            cands = self._enrich(res, q, cands) or cands
        # Paid results go ahead of earlier candidates.
        res.candidates, res.tier = (cands + res.candidates)[:10], "paid"
        return self._usable(cands)

    # Max authorless web hits per Google call whose identifier is resolved.
    MAX_WEB_IDS = 3

    def _resolve_web_ids(self, res: SearchResult, q: Query,
                         cands: list[Candidate]) -> list[Candidate]:
        """For a paper, resolve DOIs/arXiv ids found in authorless web hits to registry records."""
        if getattr(q, "source_type", "") != "paper":
            return []
        todo, seen = [], set()
        for c in sorted(cands, key=lambda c: -c.match.title_similarity):
            w = c.work
            if w.authors or not bare_record(w):
                continue
            if w.arxiv_id:
                key = ("arxiv", w.arxiv_id)
            elif w.doi:
                key = ("doi", _registry_doi(w.doi))
            else:
                continue
            if not key[1] or key in seen:
                continue
            seen.add(key)
            todo.append((key, w))
            if len(todo) >= self.MAX_WEB_IDS:
                break
        found: list[Candidate] = []
        for (kind, ident), hit in todo:
            if kind == "arxiv":
                works = self._fetch(res, "arxiv/id", "enrich", self.arxiv.fetch_id, ident,
                                    keep_all=True)
                from_address = True
            else:
                works = self._fetch(res, "crossref/doi", "enrich", self.crossref.fetch_doi, ident,
                                    keep_all=True)
                from_address = _registry_doi(extract_doi(hit.url or "") or "") == ident
            for w in works:
                if not w.authors:
                    continue
                # A record from a DOI not in the hit's URL must share half the hit's title tokens.
                if not from_address:
                    a, b = set(title_tokens(hit.title)), set(title_tokens(w.title))
                    if not a or len(a & b) / len(a) < 0.5:
                        continue
                found.append(Candidate(w, compare(q, w)))
        if found:
            log.debug("web ids: %d registry record(s) from %s", len(found),
                      [f"{k}:{i}" for (k, i), _ in todo])
            res.notes.append(f"resolved {len(found)} identifier(s) carried by web hits")
        return found

    def free_probe(self, res: SearchResult, q: Query, probe: Query, label: str,
                   *, eligible=None) -> list[Work]:
        """Free fan-out for query `probe`, filtered against citation `q`; calls logged under `label`."""
        conns = [c for c in self._route(probe) if not c.paid]
        if not conns:
            return []
        author = probe.authors[0] if probe.authors else ""
        found: list[Work] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(c.search, probe.title, author, self.rows): c for c in conns}
            try:
                for f in concurrent.futures.as_completed(futs, timeout=self.fanout_timeout):
                    c = futs[f]
                    try:
                        works = f.result() or []
                    except Exception as exc:
                        res.record(f"{label}/{c.name}", "recover", query=probe.title,
                                   error=f"{type(exc).__name__}: {exc}")
                        continue
                    keep = [w for w in works if (eligible or self._eligible)(q, w)]
                    res.record(f"{label}/{c.name}", "recover", query=probe.title,
                               n_results=len(works), n_eligible=len(keep))
                    found.extend(keep)
            except concurrent.futures.TimeoutError:
                for f in futs:
                    f.cancel()
        return found

    def paid_probe(self, res: SearchResult, q: Query, engine: str, label: str,
                   *, raw_q: str, year: Optional[int] = None,
                   eligible=None) -> list[Work]:
        """One SerpApi search with verbatim query `raw_q`, filtered against citation `q`."""
        if not self.use_paid:
            return []
        res.paid_calls += 1
        fn = (self.serp.search_scholar if engine == "scholar" else self.serp.search)
        kw = {"raw_q": raw_q} | ({"year": year} if engine == "scholar" and year else {})
        works = self._fetch(res, f"{label}/{engine}", "recover", fn,
                            q.title, "", self.rows, **kw) or []
        if engine == "google":
            # Also resolve identifiers carried by web hits (papers only).
            works = [c.work for c in self._resolve_web_ids(
                res, q, [Candidate(w, compare(q, w)) for w in works])] + works
        return [w for w in works if (eligible or self._eligible)(q, w)]

    def _score(self, q: Query, works: list[Work]) -> list[Candidate]:
        return [Candidate(w, compare(q, w)) for w in works if self._eligible(q, w)]

    @staticmethod
    def _strong(cands: list[Candidate]) -> bool:
        """True if the best candidate has an exact title and no author conflict."""
        if not cands:
            return False
        b = max(cands, key=_select_key).match
        return b.title is True and b.authors is not False

    @staticmethod
    def _usable(cands: list[Candidate]) -> bool:
        return any(c.match.title_similarity >= TITLE_EVIDENCE_THRESHOLD or
                   c.match.title_containment >= 0.95 or
                   c.match.identifier is True for c in cands)

    def _fetch(self, res: SearchResult, conn, tier: str, fn, *args,
               remote: bool = True, keep_all: bool = False, **kw) -> list[Work]:
        """Call one source and log the call; returns eligible Works, or all of them if `keep_all`."""
        t0 = time.perf_counter()
        try:
            works = fn(*args, **kw) or []
            if isinstance(works, Work):
                works = [works]
        except Exception as exc:
            res.record(conn, tier, query=res.query.title, remote=remote,
                       elapsed_ms=int((time.perf_counter() - t0) * 1000),
                       error=f"{type(exc).__name__}: {exc}")
            res.notes.append(f"{conn} failed: {exc}")
            return []
        eligible = works if keep_all else [w for w in works if self._eligible(res.query, w)]
        res.record(conn, tier, query=res.query.title, n_results=len(works),
                   n_eligible=len(eligible), remote=remote,
                   elapsed_ms=int((time.perf_counter() - t0) * 1000))
        # Log records dropped by the title filter.
        for w in works:
            if w not in eligible:
                log.debug("    dropped [%s] %r (title too far from query)",
                          conn, (w.title or "")[:70])
        return eligible

    def _enrich(self, res: SearchResult, q: Query,
                cands: list[Candidate]) -> Optional[list[Candidate]]:
        """Search free connectors on the top web hit's title; new candidates + `cands`, or None."""
        best = max(cands, key=_select_key, default=None)
        if best is None or not q.authors:
            return None
        if best.work.authors:                    # already a structured record
            return None
        title = (best.work.title or "").strip()
        # Strip site suffixes (" - PMC") and ellipses from the web title.
        import html as _html
        title = _html.unescape(title)
        title = re.sub(r"\s+[-|–—]\s+(PMC|Semantic Scholar|arXiv|ACL Anthology|ScienceDirect|"
                       r"SpringerLink|IEEE Xplore|ResearchGate|Kaggle|GitHub|EPFL|Nature|"
                       r"PubMed|bioRxiv|OpenReview|Papers With Code)\s*$", "", title, flags=re.I)
        title = re.sub(r"\s*(\.\.\.|…)\s*", " ", title).strip(" .")
        if len(title) < 12 or normalize_title(title) == normalize_title(q.title):
            return None                          # too short, or same as the cited title

        probe = Query(raw=q.raw, title=title, authors=q.authors, year=q.year,
                      venue=q.venue, doi=q.doi, arxiv_id=q.arxiv_id,
                      parser=q.parser)
        conns = [c for c in self._route(probe) if c.name != "serpapi"]
        if not conns:
            return None
        log.debug("enrich: re-querying %s with the web title %r",
                  [c.name for c in conns], title[:70])
        found: list[Work] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(c.search, title, q.authors[0], self.rows): c
                    for c in conns}
            try:
                for f in concurrent.futures.as_completed(futs, timeout=self.fanout_timeout):
                    c = futs[f]
                    try:
                        works = f.result() or []
                    except Exception as exc:
                        res.record(c.name, "enrich", query=title,
                                   error=f"{type(exc).__name__}: {exc}")
                        continue
                    # Eligible against the searched title, and must have authors.
                    ok = [w for w in works if w.authors and self._eligible(probe, w)]
                    res.record(c.name, "enrich", query=title, n_results=len(works),
                               n_eligible=len(ok))
                    found.extend(ok)
            except concurrent.futures.TimeoutError:
                for f in futs:
                    f.cancel()
        if not found:
            log.debug("enrich: no structured record found; web evidence stands")
            return None
        # Scored against the citation, not the probe.
        enriched = [Candidate(w, compare(q, w)) for w in found]
        enriched.sort(key=lambda c: -c.match.title_similarity)
        log.debug("enrich: %d structured record(s); best authors=%.2f",
                  len(enriched), enriched[0].match.author_overlap)
        res.notes.append(f"enriched via free connectors on the web title "
                         f"({len(enriched)} structured record(s))")
        return enriched + cands

    def search(self, raw_or_query) -> SearchResult:
        q = raw_or_query if isinstance(raw_or_query, Query) else parse_citation(raw_or_query)
        res = SearchResult(query=q)
        log.debug("query parser=%s title=%r authors=%s year=%s venue=%r doi=%r arxiv=%r",
                  q.parser or "?", q.title, q.authors[:4], q.year, q.venue,
                  q.doi, q.arxiv_id)

        def settle(cands: list[Candidate], tier: str) -> SearchResult:
            res.candidates, res.tier = cands, tier
            b = res.best
            log.debug("tier=%s settled on [%s] %r sim=%.2f cont=%.2f authors=%.2f",
                      tier, b.work.source, (b.work.title or "")[:70],
                      b.match.title_similarity, b.match.title_containment,
                      b.match.author_overlap)
            return res

        # -- Tier 0: local (cache, then the offline ACL mirror) ---------------
        res.tiers_tried.append("local")
        # Local sources may not settle a citation that has a DOI or arXiv id.
        def _honours_identifier(cs: list) -> bool:
            return not (q.doi or q.arxiv_id)

        kind = getattr(q, "source_type", "") or "unknown"

        def _settles(cs: list) -> bool:
            # For a paper, `bare_record` candidates cannot settle.
            if kind == "paper":
                cs = [c for c in cs if not bare_record(c.work)]
            return self._usable(cs) and _honours_identifier(cs) and self._strong(cs)

        cached = self.cache.get(q.title) if q.title else []
        if cached:
            cands = self._score(q, cached)
            res.record("cache", "local", query=q.title, n_results=len(cached),
                       n_eligible=len(cands), remote=False)
            if _settles(cands):
                return settle(cands, "cache")
            if cands:
                res.candidates = cands

        if q.title and self.acl.ready:
            hits = self._fetch(res, "acl_mirror", "local", self.acl.lookup,
                               q.title, self.rows, remote=False)
            if hits:
                cands = [Candidate(w, compare(q, w)) for w in hits]
                if _settles(cands):
                    self._remember(q, hits)
                    return settle(cands, "acl_mirror")
                if cands and not res.candidates:
                    res.candidates = cands

        # -- Routing by kind ---------------------------------------------------
        # Addressable kinds fetch the cited URL first and skip the free tier.
        addressable = kind in ("repo", "dataset", "web", "report")
        book = kind == "book"

        # -- Tier URL: the address is the identifier --------------------------
        if q.url and addressable:
            res.tiers_tried.append("url")
            w = self._fetch_url(res, q.url)
            if w is self.GONE:
                # Flag it and keep searching.
                res.url_gone = True
                w = None
            if w is None or not (w.title or "").strip():
                # No live page or no title: try an archived capture near the cited year.
                w2 = self._fetch_archived(res, q.url, q.year)
                if w2 is not None:
                    w = w2
                    res.url_gone = False
            if w is not None:
                c = Candidate(w, compare(q, w))
                m = c.match
                # Low title bar: enough to rule out a soft-404 page.
                if m.title is True or m.title_containment >= 0.3 or m.title_similarity >= 0.3:
                    self._remember(q, [w])
                    return settle([c], "url")
                else:
                    res.candidates = [c]
                    res.notes.append("page at the cited address does not resemble the "
                                     "cited title; kept as context, searching on")

        # -- Tier 1: resolve an identifier the citation already carries -------
        if (q.doi or q.arxiv_id):
            res.tiers_tried.append("identifier")
            works = []
            if q.doi:
                got = self._fetch(res, "crossref/doi", "identifier",
                                  self.crossref.fetch_doi, q.doi)
                if not got:
                    # Not a Crossref DOI: try DataCite.
                    got = self._fetch(res, "datacite/doi", "identifier",
                                      self.datacite.fetch_doi, q.doi, keep_all=True)
                works.extend(got)
                if not got:
                    res.notes.append(f"DOI {q.doi} did not resolve at Crossref or DataCite")
            if q.arxiv_id:
                got = self._fetch(res, "arxiv/id", "identifier",
                                  self.arxiv.fetch_id, q.arxiv_id)
                works.extend(got)
                if not got:
                    res.notes.append(f"arXiv id {q.arxiv_id} did not resolve")
            if works:
                cands = self._score(q, works)
                # Same authors, other title: the API gives the latest title, so add earlier versions.
                if q.arxiv_id and any(c.match.identifier is True and c.match.title is not True
                                      and c.match.authors is not False for c in cands):
                    v1 = self._fetch(res, "arxiv/v1", "identifier", self.arxiv.fetch_versions,
                                     q.arxiv_id, keep_all=True)
                    cands = [Candidate(w, compare(q, w)) for w in v1 if w.title] + cands
                # Other title and other authors: mark the identifier unreliable and search by title.
                unrelated = [c for c in cands if c.match.identifier is True
                             and c.match.title is False and c.match.authors is False]
                for c in unrelated:
                    c.work.extra = {**(c.work.extra or {}), "identifier_unreliable": True}
                if unrelated:
                    res.notes.append("the cited identifier resolves to an unrelated work; treated "
                                     "as a wrong identifier and searched by title")
                reliable = [c for c in cands if c not in unrelated]
                # Registry records first, then earlier candidates.
                merged = cands + [c for c in res.candidates if c not in cands]
                if reliable and (self._usable(reliable) or any(c.match.identifier for c in reliable)):
                    self._remember(q, [c.work for c in reliable])
                    return settle(merged, "identifier")
                res.candidates, res.tier = merged, "identifier"

        # -- Tier 2: free connectors, non-addressable kinds only --------------
        if not addressable:
            if self._free_tier(res, q, strict=book):
                return res

        # -- Tier 3: paid, engines in the order that suits the kind -----------
        if self.use_paid:
            # Google first for addressable kinds, Scholar first otherwise.
            engines = (["google", "scholar"] if addressable else
                       ["scholar", "google"])
            for engine in engines:
                if self._paid_engine(res, q, engine):
                    return res

        if res.candidates:
            log.debug("no tier produced usable evidence; keeping %d weak candidate(s)",
                      len(res.candidates))
            return res
        log.debug("nothing retrieved in any tier")
        res.tier = "none"
        return res

    def _remember(self, q: Query, works: list[Work]) -> None:
        try:
            self.cache.put(q.title, works)
        except Exception:
            pass
