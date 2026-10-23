"""Local SQLite title index of the ACL Anthology BibTeX dump."""
from __future__ import annotations

import argparse
import gzip
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import requests

from .records import Work, norm_title

DUMP_URL = "https://aclanthology.org/anthology.bib.gz"
DB_PATH = Path(__file__).with_name("acl_anthology.sqlite3")

# BibTeX case-protection braces, not content: "{M}odel" is "Model".
_BRACES = re.compile(r"[{}]")
_ENTRY = re.compile(r"@(\w+)\s*\{([^,]+),(.*?)\n\}", re.S)
_KEY = re.compile(r"(\w+)\s*=\s*")


def _fields(body: str) -> dict:
    """Fields of one BibTeX entry body, with quoted, braced or bare values."""
    out, i, n = {}, 0, len(body)
    while i < n:
        m = _KEY.search(body, i)
        if not m:
            break
        k, j = m.group(1).lower(), m.end()
        if j >= n:
            break
        if body[j] == '"':                       # quoted: ends at an unescaped quote at brace depth 0
            depth, e = 0, j + 1
            while e < n:
                c = body[e]
                if c == "\\":
                    e += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif c == '"' and depth <= 0:
                    break
                e += 1
            out.setdefault(k, body[j + 1:e]); i = e + 1
        elif body[j] == "{":                     # braced: ends at the matching brace
            depth, e = 0, j
            while e < n:
                c = body[e]
                if c == "\\":
                    e += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            out.setdefault(k, body[j + 1:e]); i = e + 1
        else:                                    # bare: a number or a month macro
            e = body.find("\n", j)
            e = n if e < 0 else e
            out.setdefault(k, body[j:e].strip().rstrip(",")); i = e + 1
    return out


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _BRACES.sub("", s)).strip()


def _authors(raw: str) -> list[str]:
    """Split 'Surname, Given and Surname, Given' into 'Given Surname' names."""
    out = []
    for chunk in re.split(r"\s+and\s+", _clean(raw)):
        chunk = chunk.strip().rstrip(",")
        if not chunk:
            continue
        if "," in chunk:
            last, first = chunk.split(",", 1)
            out.append(f"{first.strip()} {last.strip()}".strip())
        else:
            out.append(chunk)
    return out


def parse_dump(text: str):
    for m in _ENTRY.finditer(text):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        if kind.lower() == "proceedings":
            continue  # volume front matter, not a citable paper
        fields = _fields(body + "\n")
        title = _clean(fields.get("title", ""))
        if not title:
            continue
        year = fields.get("year", "")
        yield {
            "key": key,
            "title": title,
            "authors": _authors(fields.get("author", "")),
            "year": int(year) if year.strip().isdigit() else None,
            "venue": _clean(fields.get("booktitle") or fields.get("journal") or "ACL Anthology"),
            "doi": _clean(fields.get("doi", "")),
            "url": _clean(fields.get("url", "")),
        }


class ACLMirror:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10)
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    @property
    def ready(self) -> bool:
        if not self.path.exists():
            return False
        try:
            return self._conn().execute("SELECT COUNT(*) FROM papers").fetchone()[0] > 0
        except sqlite3.Error:
            return False

    def build(self, url: str = DUMP_URL) -> int:
        print(f"[acl] downloading {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        text = gzip.decompress(r.content).decode("utf-8", errors="replace")
        print(f"[acl] parsing {len(text)/1e6:.1f} MB of BibTeX")

        conn = self._conn()
        conn.executescript("""
            DROP TABLE IF EXISTS papers;
            CREATE TABLE papers (
                key TEXT PRIMARY KEY, norm TEXT NOT NULL, title TEXT,
                authors TEXT, year INTEGER, venue TEXT, doi TEXT, url TEXT);
        """)
        rows, n = [], 0
        for e in parse_dump(text):
            rows.append((e["key"], norm_title(e["title"]), e["title"],
                         " and ".join(e["authors"]), e["year"], e["venue"],
                         e["doi"], e["url"]))
            if len(rows) >= 5000:
                conn.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?)", rows)
                n += len(rows); rows = []
        if rows:
            conn.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?)", rows)
            n += len(rows)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_norm ON papers(norm)")
        conn.commit()
        print(f"[acl] indexed {n} papers -> {self.path}")
        return n

    def lookup(self, title: str, rows: int = 5) -> list[Work]:
        """Exact normalized-title hit, then a prefix probe for truncated titles."""
        if not self.ready or not title:
            return []
        key = norm_title(title)
        c = self._conn()
        hits = c.execute(
            "SELECT title, authors, year, venue, doi, url FROM papers WHERE norm = ? LIMIT ?",
            (key, rows)).fetchall()
        if not hits and len(key) > 25:
            hits = c.execute(
                "SELECT title, authors, year, venue, doi, url FROM papers "
                "WHERE norm LIKE ? LIMIT ?", (key[:25] + "%", rows)).fetchall()
        return [Work(title=t, authors=[a for a in (au or "").split(" and ") if a],
                     year=y, venue=v, doi=d or "", url=u or "", source="aclanthology")
                for t, au, y, v, d, u in hits]

    def stats(self) -> dict:
        if not self.ready:
            return {"ready": False, "path": str(self.path)}
        n, lo, hi = self._conn().execute(
            "SELECT COUNT(*), MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL").fetchone()
        return {"ready": True, "papers": n, "years": f"{lo}-{hi}",
                "path": str(self.path),
                "mb": round(self.path.stat().st_size / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--lookup")
    a = ap.parse_args()
    m = ACLMirror()
    if a.build:
        m.build()
    if a.stats or not (a.build or a.lookup):
        print(m.stats())
    if a.lookup:
        for w in m.lookup(a.lookup):
            print(f"  {w.title!r} ({w.year}) {w.authors[:3]} {w.url}")


if __name__ == "__main__":
    main()
