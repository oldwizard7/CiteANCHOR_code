"""Bibliographic connectors; each maps its provider's records onto `Work`."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from .runlog import get as _get_logger
from .records import Work, extract_arxiv_id, extract_doi

log = _get_logger("connectors")

CONTACT = os.getenv("CITEANCHOR_CONTACT", "ctong29@wisc.edu")   # polite-pool contact for Crossref / OpenAlex
UA = {"User-Agent": f"CiteAnchor/0.1 (mailto:{CONTACT})"}
TIMEOUT = 20


class RateLimiter:
    """Minimum spacing between calls to one provider, shared across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            gap = time.time() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.time()


class Connector:
    name = "base"
    paid = False
    limiter = RateLimiter(0.0)
    backoff = 2.0            # seconds, doubled per retry
    # Cooldowns in seconds: how long a tripped connector is skipped, by failure kind.
    cooldown_429 = 45.0          # retries exhausted on a transient rate limit
    cooldown_unreachable = 120.0  # timeouts and connection errors
    cooldown_blocked = 1800.0    # an explicit refusal: 401/403 or a block page
    cooldown_quota_max = 3600.0  # upper bound when Retry-After gives a quota reset
    attempts_429 = 4             # a 429 is usually gone by the next request
    backoff_429 = 1.5            # seconds, x1.7 per retry: 1.5, 2.6, 4.3
    last_error = ""          # why the most recent call produced nothing

    # Circuit breaker, shared per connector class.
    _cool_until: float = 0.0
    _cool_lock = threading.Lock()

    @classmethod
    def cooling(cls) -> float:
        """Seconds left before this connector should be tried again."""
        return max(0.0, cls._cool_until - time.time())

    @classmethod
    def _trip(cls, seconds: float, why: str) -> None:
        with cls._cool_lock:
            cls._cool_until = max(cls._cool_until, time.time() + seconds)
        cls.last_error = why
        cls.trips = getattr(cls, "trips", 0) + 1
        log.warning("connector %s parked for %.0fs: %s", cls.name, seconds, why)

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        raise NotImplementedError

    def _get(self, url: str, *, expect: str = "", attempts: int = 3,
             headers: Optional[dict] = None, **kw) -> Optional[requests.Response]:
        """GET with retries on 429/5xx; None on failure or a body that is not `expect`."""
        if self.cooling():
            return None

        n429 = 0
        i = 0
        while i < attempts:
            self.limiter.wait()
            try:
                r = requests.get(url, headers={**UA, **(headers or {})},
                                 timeout=TIMEOUT, **kw)
            except Exception as exc:
                i += 1
                if i < attempts:
                    time.sleep(self.backoff)
                    continue
                self._trip(self.cooldown_unreachable, f"unreachable: {type(exc).__name__}")
                return None
            if r.status_code == 200:
                body = r.content[:400].lstrip()
                # some providers answer a blocked client with HTTP 200 and an HTML error page
                if body.startswith(b"<!DOCTYPE html") or body.startswith(b"<html"):
                    self._trip(self.cooldown_blocked, "blocked: HTML error page")
                    return None
                if expect == "json" and not body.startswith((b"{", b"[")):
                    self.last_error = f"non-JSON body: {body[:60]!r}"
                    return None
                if expect == "xml" and not body.startswith(b"<"):
                    self.last_error = f"non-XML body: {body[:60]!r}"
                    return None
                return r
            if r.status_code == 429:
                # a long Retry-After means a spent quota (park); otherwise a short rate limit (retry)
                try:
                    wait = float(r.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0.0
                if wait > 120:
                    self._trip(min(wait, self.cooldown_quota_max),
                               f"HTTP 429 — quota exhausted, resets in {wait / 3600:.1f} h")
                    return None
                n429 += 1
                if n429 < self.attempts_429:
                    time.sleep(max(wait, self.backoff_429 * (1.7 ** (n429 - 1))))
                    continue
                self._trip(self.cooldown_429, f"HTTP 429 — still rate limited after {n429} tries")
                return None
            if r.status_code in (403, 401):
                self._trip(self.cooldown_blocked, f"HTTP {r.status_code} — refused")
                return None
            i += 1
            if r.status_code in (500, 502, 503, 504) and i < attempts:
                time.sleep((2 ** (i - 1)) * self.backoff)
                continue
            self.last_error = f"HTTP {r.status_code}"
            return None
        return None

    def healthy(self, attempts: int = 2) -> bool:
        """True if a known-title search returns a result; the breaker state is restored afterwards."""
        cls = type(self)
        with cls._cool_lock:
            before = cls._cool_until
        try:
            for i in range(attempts):
                try:
                    if self.search("Attention Is All You Need", rows=1):
                        return True
                except Exception:
                    pass
                if i + 1 < attempts:
                    time.sleep(max(1.0, self.limiter.min_interval))
            return False
        finally:
            with cls._cool_lock:
                cls._cool_until = before


# ------------------------------------------------------------------- Crossref ---

class Crossref(Connector):
    name = "crossref"
    limiter = RateLimiter(0.15)

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(f"{title} {author}".strip())
        r = self._get(f"https://api.crossref.org/works?rows={rows}"
                      f"&select=title,author,issued,container-title,DOI,URL,type,volume,page"
                      f"&query.bibliographic={q}&mailto={CONTACT}", expect="json")
        if not r:
            return []
        out = []
        for it in r.json().get("message", {}).get("items", []):
            out.append(self._to_work(it))
        return out

    def fetch_doi(self, doi: str) -> Optional[Work]:
        r = self._get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
                      f"?mailto={CONTACT}", expect="json")
        if not r:
            return None
        return self._to_work(r.json().get("message", {}))

    def _to_work(self, it: dict) -> Work:
        titles = it.get("title") or [""]
        authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
                   for a in it.get("author", [])]
        parts = (it.get("issued") or {}).get("date-parts") or [[None]]
        year = parts[0][0] if parts and parts[0] else None
        container = it.get("container-title") or [""]
        return Work(
            title=titles[0] if titles else "",
            authors=[a for a in authors if a],
            year=int(year) if isinstance(year, int) else None,
            venue=container[0] if container else "",
            doi=it.get("DOI", "") or "",
            url=it.get("URL", "") or "",
            volume=str(it.get("volume") or ""),
            pages=str(it.get("page") or ""),
            source=self.name,
            extra={"type": it.get("type", "")},
        )


# ------------------------------------------------------------------- DataCite ---

class DataCite(Connector):
    """DOI lookup for DataCite-registered DOIs (datasets, software, repository deposits); no search."""
    name = "datacite"
    limiter = RateLimiter(0.2)

    def fetch_doi(self, doi: str) -> Optional[Work]:
        r = self._get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}", expect="json")
        if not r:
            return None
        a = ((r.json().get("data") or {}).get("attributes") or {})
        titles = [t.get("title", "") for t in a.get("titles") or [] if t.get("title")]
        if not titles:
            return None
        authors = []
        for c in a.get("creators") or []:
            given, family = c.get("givenName"), c.get("familyName")
            name = " ".join(x for x in (given, family) if x) or (c.get("name") or "")
            if "," in name and not (given or family):          # "Family, Given"
                last, first = name.split(",", 1)
                name = f"{first.strip()} {last.strip()}"
            if name.strip():
                authors.append(name.strip())
        year = a.get("publicationYear")
        return Work(
            title=re.sub(r"\s+", " ", titles[0]).strip(), authors=authors,
            year=int(year) if str(year or "").isdigit() else None,
            venue=a.get("publisher") if isinstance(a.get("publisher"), str)
            else ((a.get("publisher") or {}).get("name") or ""),
            doi=a.get("doi", "") or doi, url=a.get("url", "") or f"https://doi.org/{doi}",
            source=self.name,
            extra={"type": (a.get("types") or {}).get("resourceTypeGeneral", "")})


# -------------------------------------------------------------------- Wayback ---

class Wayback(Connector):
    """Archived copies of a URL from the Internet Archive's Wayback Machine."""
    name = "wayback"
    limiter = RateLimiter(1.0)

    @staticmethod
    def enabled() -> bool:
        return True

    def reachable(self, timeout: float = 5.0) -> bool:
        try:
            r = requests.head("https://archive.org/", headers=UA, timeout=timeout,
                              allow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        return []                       # not a search engine; see `snapshot`

    def snapshot(self, url: str, year: Optional[int] = None) -> Optional[dict]:
        """The archived copy closest to the end of `year`: {"url", "timestamp"} or None."""
        params = {"url": url}
        if year:
            params["timestamp"] = f"{int(year)}1231"
        r = self._get("https://archive.org/wayback/available?" + urllib.parse.urlencode(params),
                      expect="json")
        if not r:
            return None
        try:
            snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
        except ValueError:
            return None
        if not snap.get("available") or not snap.get("url"):
            return None
        return {"url": snap["url"], "timestamp": str(snap.get("timestamp") or "")}

    def page(self, snapshot_url: str) -> Optional[requests.Response]:
        """The archived page's original bytes (`id_` in the snapshot path drops the archive's banner)."""
        m = re.match(r"^(https?://web\.archive\.org/web/)(\d+)(/.*)$", snapshot_url)
        raw = f"{m.group(1)}{m.group(2)}id_{m.group(3)}" if m else snapshot_url
        return self._get(raw)


# ---------------------------------------------------------------- Common Crawl ---

class CommonCrawl(Connector):
    """Archived copies of a URL from Common Crawl (CDX index lookup, then a WARC range request)."""
    name = "commoncrawl"
    limiter = RateLimiter(0.5)
    CRAWLS = 8

    @staticmethod
    def enabled() -> bool:
        return True

    def __init__(self):
        self._collections: Optional[list] = None

    def reachable(self, timeout: float = 5.0) -> bool:
        try:
            r = requests.get("https://index.commoncrawl.org/collinfo.json", headers=UA, timeout=timeout)
            if r.status_code != 200:
                return False
            self._collections = [c["id"] for c in r.json()][: self.CRAWLS]
            return bool(self._collections)
        except Exception:
            return False

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        return []                       # not a search engine; see `snapshot`

    def snapshot(self, url: str, year: Optional[int] = None) -> Optional[dict]:
        """The first HTTP-200 capture of `url` in the most recent crawls, or None."""
        if not self._collections and not self.reachable():
            return None
        for coll in self._collections or []:
            r = self._get(f"https://index.commoncrawl.org/{coll}-index?"
                          + urllib.parse.urlencode({"url": url, "output": "json", "limit": 5}),
                          attempts=1)
            if not r or not r.text.strip():
                continue
            for line in r.text.strip().splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if str(d.get("status")) == "200" and d.get("filename"):
                    return {"url": url, "timestamp": str(d.get("timestamp") or ""),
                            "filename": d["filename"], "offset": int(d["offset"]),
                            "length": int(d["length"]), "crawl": coll}
        return None

    def page(self, snap: dict) -> Optional[requests.Response]:
        """The archived record's HTML, decompressed from its WARC slice."""
        import gzip, io
        start = snap["offset"]; end = start + snap["length"] - 1
        # A range request is answered with 206, which `_get` would refuse.
        self.limiter.wait()
        try:
            r = requests.get("https://data.commoncrawl.org/" + snap["filename"],
                             headers={**UA, "Range": f"bytes={start}-{end}"}, timeout=TIMEOUT)
        except Exception:
            return None
        if r.status_code not in (200, 206):
            return None
        try:
            raw = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8", "replace")
        except Exception:
            return None
        # WARC header, then the HTTP response (headers, blank line, body)
        body = raw.split("\r\n\r\n", 2)
        r.archived_text = body[-1] if body else raw
        return r


# --------------------------------------------------------------- Google Books ---

class GoogleBooks(Connector):
    """Google Books volume search; disabled without GOOGLE_BOOKS_API_KEY."""
    name = "googlebooks"
    limiter = RateLimiter(0.7)          # 100 requests per minute per project

    @staticmethod
    def enabled() -> bool:
        return bool(os.getenv("GOOGLE_BOOKS_API_KEY", ""))

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        key = os.getenv("GOOGLE_BOOKS_API_KEY", "")
        if not key or not title:
            return []
        q = f'intitle:"{title}"' + (f" inauthor:{author.split()[-1]}" if author.strip() else "")
        r = self._get("https://www.googleapis.com/books/v1/volumes?q=" + urllib.parse.quote(q)
                      + f"&maxResults={min(max(rows, 1), 10)}&printType=books"
                      + "&fields=items(id,volumeInfo(title,subtitle,authors,publisher,"
                      "publishedDate,industryIdentifiers,canonicalVolumeLink,infoLink))"
                      + f"&key={urllib.parse.quote(key)}", expect="json")
        if not r:
            return []
        out = []
        for it in (r.json().get("items") or [])[:rows]:
            v = it.get("volumeInfo") or {}
            t = (v.get("title") or "").strip()
            if not t:
                continue
            sub = (v.get("subtitle") or "").strip()
            authors = [a.strip() for a in (v.get("authors") or []) if a and a.strip()]
            if not authors:
                continue
            date = str(v.get("publishedDate") or "")
            isbn = next((x.get("identifier") for x in (v.get("industryIdentifiers") or [])
                         if x.get("type") == "ISBN_13"), "") or \
                   next((x.get("identifier") for x in (v.get("industryIdentifiers") or [])
                         if x.get("type") == "ISBN_10"), "")
            out.append(Work(
                # the API returns the subtitle separately; rejoin as "Title: Subtitle"
                title=f"{t}: {sub}" if sub else t, authors=authors,
                year=int(date[:4]) if date[:4].isdigit() else None,
                venue=(v.get("publisher") or "").strip(),
                url=v.get("canonicalVolumeLink") or v.get("infoLink") or "",
                source=self.name,
                extra={"subtitle": sub, "isbn": isbn or "", "volume_id": it.get("id", "")},
            ))
        return out


# ------------------------------------------------------------------- OpenAlex ---

class OpenAlex(Connector):
    name = "openalex"
    limiter = RateLimiter(0.15)

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(title)
        # optional; without a key requests share the free per-IP daily budget
        key = os.getenv("OPENALEX_API_KEY", "")
        r = self._get(f"https://api.openalex.org/works?per-page={rows}"
                      f"&search={q}&mailto={CONTACT}"
                      + (f"&api_key={urllib.parse.quote(key)}" if key else ""), expect="json")
        if not r:
            return []
        out = []
        for it in r.json().get("results", []):
            ids = it.get("ids", {}) or {}
            loc = (it.get("primary_location") or {}).get("source") or {}
            bib = it.get("biblio") or {}
            out.append(Work(
                title=it.get("title") or "",
                authors=[(a.get("author") or {}).get("display_name", "")
                         for a in it.get("authorships", [])],
                year=it.get("publication_year"),
                venue=loc.get("display_name", "") or "",
                doi=(ids.get("doi") or "").replace("https://doi.org/", ""),
                arxiv_id=extract_arxiv_id(ids.get("openalex", "") or "") or "",
                url=it.get("doi") or ids.get("openalex", "") or "",
                volume=str(bib.get("volume") or ""),
                pages=str(bib.get("first_page") or ""),
                source=self.name,
                extra={"type": it.get("type", "")},
            ))
        return out


# ---------------------------------------------------------------------- arXiv ---

class Arxiv(Connector):
    name = "arxiv"
    limiter = RateLimiter(3.5)   # arXiv asks for one request per 3 seconds
    backoff = 10.0               # a 429 here persists for minutes, not seconds
    NS = {"a": "http://www.w3.org/2005/Atom"}

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(f'ti:"{title}"')
        r = self._get(f"https://export.arxiv.org/api/query?max_results={rows}&search_query={q}", expect="xml")
        return self._parse(r.text) if r else []

    def fetch_id(self, arxiv_id: str) -> Optional[Work]:
        """Resolve an arXiv id via the API, falling back to DataCite and then the abs page."""
        r = self._get(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}", expect="xml")
        works = self._parse(r.text) if r else []
        if works:
            return works[0]
        return self._fetch_id_datacite(arxiv_id) or self._fetch_id_abs(arxiv_id)

    @staticmethod
    def _alt_get(url: str, **kw):
        """Plain GET outside `_get`, so the arXiv rate limiter and breaker do not apply."""
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT, **kw)
            return r if r.status_code == 200 else None
        except Exception:
            return None

    def _fetch_id_datacite(self, arxiv_id: str) -> Optional[Work]:
        bare = arxiv_id.split("v")[0]
        # arXiv ids are DOIs under the 10.48550 prefix, registered with DataCite
        r = self._alt_get(f"https://api.datacite.org/dois/10.48550/arXiv.{urllib.parse.quote(bare)}")
        if not r:
            return None
        try:
            a = r.json()["data"]["attributes"]
        except Exception:
            return None
        titles = a.get("titles") or [{}]
        authors = [c.get("name") or " ".join(
            x for x in (c.get("givenName"), c.get("familyName")) if x)
            for c in (a.get("creators") or [])]
        year = a.get("publicationYear")
        return Work(
            title=re.sub(r"\s+", " ", titles[0].get("title", "")),
            authors=[x for x in authors if x],
            year=int(year) if str(year).isdigit() else None,
            venue="arXiv", arxiv_id=bare,
            doi=f"10.48550/arXiv.{bare}",
            url=f"https://arxiv.org/abs/{bare}",
            source=self.name, extra={"via": "datacite"})

    _META = re.compile(r'<meta[^>]*\bname="(citation_[a-z_]+)"[^>]*\bcontent="([^"]*)"', re.I)

    def fetch_version(self, arxiv_id: str, version: int = 1) -> Optional[Work]:
        """One version of a preprint, read from its abs page."""
        w = self._fetch_id_abs(arxiv_id, version=version)
        if w is not None:
            w.extra = {**(w.extra or {}), "version": f"v{version}"}
        return w

    _VERSIONS = re.compile(r"\[v(\d+)\]")
    MAX_VERSIONS = 6

    def fetch_versions(self, arxiv_id: str) -> list[Work]:
        """Every version but the latest, oldest first (version list read from the v1 abs page)."""
        first = self.fetch_version(arxiv_id, 1)
        if first is None:
            return []
        out = [first]
        latest = max((int(n) for n in self._VERSIONS.findall(first.extra.get("_history", ""))),
                     default=1)
        for v in range(2, min(latest, self.MAX_VERSIONS + 1)):   # the latest version comes from the API
            w = self.fetch_version(arxiv_id, v)
            if w is not None:
                out.append(w)
        for w in out:
            (w.extra or {}).pop("_history", None)
        return out

    def _fetch_id_abs(self, arxiv_id: str, version: Optional[int] = None) -> Optional[Work]:
        bare = arxiv_id.split("v")[0]
        self.limiter.wait()          # same rate limit as the API
        r = self._alt_get("https://arxiv.org/abs/" + urllib.parse.quote(bare)
                          + (f"v{version}" if version else ""))
        if not r:
            return None
        meta: dict = {}
        for name, content in self._META.findall(r.text):
            meta.setdefault(name.lower(), []).append(content)
        title = (meta.get("citation_title") or [""])[0]
        if not title:
            return None
        date = (meta.get("citation_date") or meta.get("citation_online_date") or [""])[0]
        # citation_author is "Last, First M."; reorder to "First M. Last"
        authors = []
        for a in meta.get("citation_author", []):
            if "," in a:
                last, first = a.split(",", 1)
                a = f"{first.strip()} {last.strip()}"
            authors.append(a.strip())
        return Work(
            title=re.sub(r"\s+", " ", title), authors=authors,
            year=int(date[:4]) if date[:4].isdigit() else None,
            venue="arXiv", arxiv_id=bare,
            url=f"https://arxiv.org/abs/{bare}",
            source=self.name,
            extra={"via": "abs_meta",
                   "_history": " ".join(re.findall(r"\[v\d+\]", r.text)) if version == 1 else ""})

    def _parse(self, xml: str) -> list[Work]:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []
        out = []
        for e in root.findall("a:entry", self.NS):
            def txt(tag):
                el = e.find(f"a:{tag}", self.NS)
                return (el.text or "").strip() if el is not None else ""
            eid = txt("id")
            published = txt("published")
            out.append(Work(
                title=re.sub(r"\s+", " ", txt("title")),
                authors=[(a.find("a:name", self.NS).text or "")
                         for a in e.findall("a:author", self.NS)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue="arXiv",
                arxiv_id=extract_arxiv_id(eid) or "",
                url=eid,
                source=self.name,
            ))
        return out


# ------------------------------------------------------------------ EuropePMC ---

class EuropePMC(Connector):
    name = "europepmc"
    limiter = RateLimiter(0.2)

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(f'TITLE:"{title}"')
        r = self._get("https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                      f"?format=json&pageSize={rows}&query={q}", expect="json")
        if not r:
            return []
        out = []
        for it in r.json().get("resultList", {}).get("result", []):
            out.append(Work(
                title=it.get("title", "") or "",
                authors=[a.strip() for a in (it.get("authorString") or "").split(",") if a.strip()],
                year=int(it["pubYear"]) if str(it.get("pubYear", "")).isdigit() else None,
                venue=it.get("journalTitle", "") or "",
                doi=it.get("doi", "") or "",
                url=f"https://europepmc.org/article/{it.get('source','')}/{it.get('id','')}",
                source=self.name,
            ))
        return out


# ------------------------------------------------------------ SemanticScholar ---

class SemanticScholar(Connector):
    """Semantic Scholar paper search; set S2_API_KEY (unauthenticated requests get 429)."""
    name = "semanticscholar"
    limiter = RateLimiter(2.0)   # the API's published limit is 1 request/s
    # The API returns transient 429s at any spacing: more retries, short waits, short park.
    attempts_429 = 7
    backoff_429 = 1.2            # 1.2, 2.0, 3.5, 5.9, 10, 17 s
    cooldown_429 = 10.0

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        key = os.getenv("S2_API_KEY", "")
        q = urllib.parse.quote(title)
        r = self._get(
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?limit={rows}&fields=title,authors,year,venue,externalIds,url,journal&query={q}",
            expect="json", headers={"x-api-key": key} if key else None)
        if r is None:
            return []
        out = []
        for it in r.json().get("data", []):
            ext = it.get("externalIds") or {}
            jnl = it.get("journal") or {}
            out.append(Work(
                title=it.get("title", "") or "",
                authors=[a.get("name", "") for a in it.get("authors", [])],
                year=it.get("year"),
                venue=it.get("venue", "") or "",
                doi=ext.get("DOI", "") or "",
                arxiv_id=ext.get("ArXiv", "") or "",
                url=it.get("url", "") or "",
                volume=str(jnl.get("volume") or "").strip(),
                pages=str(jnl.get("pages") or "").strip(),
                source=self.name,
            ))
        return out


# ----------------------------------------------------------------------- DBLP ---

class DBLP(Connector):
    name = "dblp"
    limiter = RateLimiter(0.5)

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(title)
        r = self._get(f"https://dblp.org/search/publ/api?format=json&h={rows}&q={q}", expect="json")
        if not r:
            return []
        hits = (r.json().get("result", {}).get("hits", {}) or {}).get("hit", [])
        out = []
        for h in hits:
            info = h.get("info", {})
            a = (info.get("authors") or {}).get("author", [])
            a = a if isinstance(a, list) else [a]
            out.append(Work(
                title=(info.get("title") or "").rstrip("."),
                authors=[x.get("text", "") if isinstance(x, dict) else str(x) for x in a],
                year=int(info["year"]) if str(info.get("year", "")).isdigit() else None,
                venue=info.get("venue", "") or "",
                doi=info.get("doi", "") or "",
                url=info.get("ee", "") or info.get("url", "") or "",
                source=self.name,
            ))
        return out


# -------------------------------------------------------------------- SerpApi ---

def scholar_venue(summary: str) -> str:
    """The venue in a Google Scholar result's `publication_info.summary`, or ""."""
    s = (summary or "").strip()
    if not s:
        return ""
    year = re.compile(r",?\s*(?:19|20)\d\d\s*$")
    # "A Author, B Author - Venue, 2021 - publisher.com"
    if " - " in s:
        parts = [p.strip() for p in s.split(" - ")]
        if len(parts) < 2:
            return ""
        v = year.sub("", parts[1]).strip(" ,")
        if not v or re.fullmatch(r"(?:19|20)\d\d", v) or re.fullmatch(r"[\w.-]+\.[a-z]{2,}", v):
            return ""
        return v
    # "A Author, B AuthorVenue, 2021•Publisher": the byline runs into the venue
    head = year.sub("", s.split("•")[0]).strip(" ,")
    if "…" in head:                           # byline cut short: the venue follows the last "…"
        v = head.rsplit("…", 1)[1].strip(" ,")
        return v
    last = head.rsplit(", ", 1)[-1]           # "B AuthorVenue ..." -> "Venue ..."
    m = re.match(r"^[A-Z]{1,4} \S*?[a-zß-ÿ](?=[A-Z])(.*)$", last)
    if m:
        return m.group(1).strip(" ,")
    return "" if ", " in head else head        # a byline with no venue after it


class SerpApi(Connector):
    """Google and Google Scholar search through SerpApi (paid; needs SERPAPI_API_KEY)."""
    name = "serpapi"
    paid = True
    limiter = RateLimiter(0.5)

    # error substrings that mean the account's search quota is used up
    _EXHAUSTED = ("run out of searches", "your account has run out", "no plan",
                  "plan has expired", "out of searches")
    exhausted = False
    exhausted_after = None

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def _note_exhausted(self, msg: str) -> None:
        with self._lock:
            if SerpApi.exhausted:
                return
            SerpApi.exhausted = True
            SerpApi.exhausted_after = self.calls
        log.error("SerpApi quota EXHAUSTED after %d calls — every later paid lookup "
                  "returns nothing and non-paper rows will read as absent. Top up and "
                  "re-run the rows after this point. (%s)", self.calls, msg[:120])

    def _call(self, params: dict):
        key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or ""
        if not key or SerpApi.exhausted:
            return None
        import serpapi
        from .runlog import redact
        self.limiter.wait()
        try:
            data = serpapi.Client(api_key=key).search(params)
        except Exception as exc:
            # The client echoes the request URL, key included, in its message.
            self.last_error = redact(str(exc))[:200]
            if any(p in self.last_error.lower() for p in self._EXHAUSTED):
                self._note_exhausted(self.last_error)
            return None
        # some failures come back as an "error" field instead of an exception
        err = data.get("error") if isinstance(data, dict) else None
        if err:
            self.last_error = redact(str(err))[:200]
            if any(p in self.last_error.lower() for p in self._EXHAUSTED):
                self._note_exhausted(self.last_error)
            return None
        with self._lock:
            self.calls += 1
        return data

    def search(self, title: str, author: str = "", rows: int = 5,
               *, raw_q: Optional[str] = None) -> list[Work]:
        """Google web search; `raw_q` replaces the default quoted-title-plus-author query."""
        q = raw_q if raw_q else f'"{title}"' + (f" {author}" if author else "")
        data = self._call({"engine": "google", "q": q, "google_domain": "google.com",
                           "hl": "en", "gl": "us", "num": 10})
        if not data:
            return []
        out = []
        for it in (data.get("organic_results") or [])[:rows]:
            link = it.get("link", "") or ""
            snippet = it.get("snippet", "") or ""
            out.append(Work(
                title=it.get("title", "") or "",
                url=link,
                doi=extract_doi(link) or extract_doi(snippet) or "",
                arxiv_id=extract_arxiv_id(link) or "",
                source=self.name,
                extra={"snippet": snippet, "displayed_link": it.get("displayed_link", "")},
            ))
        return out

    _SCHOLAR_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")

    def search_scholar(self, title: str, author: str = "", rows: int = 5,
                       *, raw_q: Optional[str] = None,
                       year: Optional[int] = None) -> list[Work]:
        """Google Scholar search; authors, year and venue are read from `publication_info`."""
        q = raw_q if raw_q else f'"{title}"' + (f" {author}" if author else "")
        params = {"engine": "google_scholar", "q": q, "hl": "en", "num": 10}
        if year:
            # restrict results to year ± 1
            params["as_ylo"], params["as_yhi"] = year - 1, year + 1
        data = self._call(params)
        if not data:
            return []
        out = []
        for it in (data.get("organic_results") or [])[:rows]:
            info = it.get("publication_info") or {}
            summary = info.get("summary", "") or ""
            authors = [a.get("name", "") for a in (info.get("authors") or []) if a.get("name")]
            if not authors and summary:
                # "A Author, B Author - 1975 - books.google.com": authors precede the first " - "
                head = summary.split(" - ")[0]
                authors = [a.strip() for a in head.split(",") if a.strip()]
            years = self._SCHOLAR_YEAR.findall(summary)
            link = it.get("link", "") or ""
            out.append(Work(
                title=it.get("title", "") or "",
                authors=authors,
                year=int(years[0]) if years else None,
                venue=scholar_venue(summary),
                url=link,
                doi=extract_doi(link) or "",
                arxiv_id=extract_arxiv_id(link) or "",
                source="serpapi_scholar",
                # type "Citation": an entry built only from reference lists, with no document behind it
                extra={"summary": summary, "snippet": it.get("snippet", "") or "",
                       "type": it.get("type", "") or "",
                       "cited_by": ((it.get("inline_links") or {}).get("cited_by") or {}).get("total")},
            ))
        return out


# --------------------------------------------------------------------- PubMed ---

class PubMed(Connector):
    name = "pubmed"
    limiter = RateLimiter(0.4)     # NCBI asks for <=3 req/s unauthenticated
    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, title: str, author: str = "", rows: int = 5) -> list[Work]:
        q = urllib.parse.quote(f"{title}[Title]")
        r = self._get(f"{self.EUTILS}/esearch.fcgi?db=pubmed&retmode=json"
                      f"&retmax={rows}&term={q}", expect="json")
        if not r:
            return []
        ids = (r.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        r2 = self._get(f"{self.EUTILS}/esummary.fcgi?db=pubmed&retmode=json"
                       f"&id={','.join(ids)}", expect="json")
        if not r2:
            return []
        result = r2.json().get("result", {})
        out = []
        for pmid in ids:
            it = result.get(pmid) or {}
            if not it.get("title"):
                continue
            year = (it.get("pubdate") or "")[:4]
            doi = next((x.get("value", "") for x in it.get("articleids", [])
                        if x.get("idtype") == "doi"), "")
            out.append(Work(
                title=re.sub(r"\.$", "", it.get("title", "")),
                authors=[a.get("name", "") for a in it.get("authors", [])],
                year=int(year) if year.isdigit() else None,
                venue=it.get("fulljournalname") or it.get("source", "") or "",
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source=self.name,
            ))
        return out


FREE_CONNECTORS = [Crossref, OpenAlex, Arxiv, EuropePMC, SemanticScholar, DBLP, PubMed]
