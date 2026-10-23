"""Record dataclasses, citation-string extractors and field-level comparison."""
from __future__ import annotations

import html as _html
import re
from urllib.parse import urlparse
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional

from .source_type import BOOK
from .normalize import (normalize_venue, similarity,
                        venues_equivalent_heuristic)

# ------------------------------------------------------------------ normalize ---

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


# Letters that NFKD does not decompose into a base letter plus a combining mark.
_NO_DECOMPOSITION = str.maketrans({
    "Đ": "D", "đ": "d", "Ð": "D", "ð": "d", "Ł": "L", "ł": "l", "Ø": "O", "ø": "o",
    "ß": "ss", "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe", "ı": "i", "Þ": "Th", "þ": "th",
    "Ħ": "H", "ħ": "h", "Ŧ": "T", "ŧ": "t"})


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.translate(_NO_DECOMPOSITION))
                   if not unicodedata.combining(c))


# BibTeX accent macros such as \'e or \"{o}, and letter macros such as \ss.
_TEX_ACCENT = re.compile(r"""\\[`'^"~=.uvHtcdbkr]\s*\{?\s*\\?([A-Za-z])\s*\}?""")
_TEX_LETTER = {"\\ss": "ss", "\\o": "o", "\\O": "O", "\\l": "l", "\\L": "L",
               "\\ae": "ae", "\\AE": "AE", "\\aa": "a", "\\AA": "A", "\\i": "i", "\\j": "j"}


def clean_text(s: str) -> str:
    """Undo HTML entities and BibTeX accent macros."""
    if not s or ("&" not in s and "\\" not in s and "{" not in s):
        return s or ""
    s = _html.unescape(s)
    s = _TEX_ACCENT.sub(r"\1", s)
    for k, v in _TEX_LETTER.items():
        s = re.sub(re.escape(k) + r"(?![A-Za-z])\s*", v, s)
    return s.replace("{", "").replace("}", "")


# British-to-American spelling rewrites, applied to both sides of a title comparison.
_SPELLING = [
    (re.compile(r"(?<=\w{3})our\b"), "or"),            # behaviour, colour, labour
    (re.compile(r"is(e|ed|es|ing|ation|ational)\b"), r"iz\1"),
    (re.compile(r"ys(e|ed|es|ing)\b"), r"yz\1"),       # analyse
    (re.compile(r"\bcentre\b|\bmetre\b|\blitre\b|\bfibre\b|\btheatre\b|"
                r"\bcalibre\b|\bspectre\b|\blustre\b"), lambda m: m.group(0)[:-2] + "er"),
    (re.compile(r"ogue\b"), "og"),                      # catalogue, dialogue
    (re.compile(r"ence\b"), "ense"),                    # defence, licence
    (re.compile(r"(?<=[aeiou])([lpt])\1(?=(?:ing|ed|er|or)\b)"), r"\1"),  # modelling
    (re.compile(r"\bprogramme\b"), "program"),
    (re.compile(r"(?<=\w)(ae|oe)(?=\w)"), "e"),        # anaemia, oestrogen
]


def _respell(s: str) -> str:
    for rx, rep in _SPELLING:
        s = rx.sub(rep, s)
    return s


def norm_title(s: str, *, glue: bool = False) -> str:
    """Normalized title; `glue` joins hyphenated words instead of splitting them."""
    if not s:
        return ""
    s = strip_accents(clean_text(s).lower())
    s = s.replace("-", "" if glue else " ")
    s = _PUNCT.sub(" ", s)
    return _respell(_SPACE.sub(" ", s).strip())


def title_tokens(s: str, *, glue: bool = False) -> list[str]:
    return norm_title(s, glue=glue).split()


# Trailing truncation markers in author names: "et al.", "and others", "etc.".
_ETAL = re.compile(r"\b(et\.?\s*al\.?|and\s+others|,?\s*etc\.?)\s*$", re.I)
_NOT_SURNAMES = {"al", "others", "etc"}

# Trailing all-caps initials, as in "Smith J" or "Smith AB" (PubMed / Europe PMC style).
_TRAILING_INITIALS = re.compile(r"[A-Z]{1,3}")

# Generational and degree suffixes, skipped when picking the surname.
_SUFFIX = re.compile(r"^(jr|sr|ii|iii|iv|v|phd|md)\.?$", re.I)


# Volume and first page: "21(1):010126", "3(10):589–604", "vol. 12, pp. 1-14".
_VOLPAGE = re.compile(
    r"(?<!\d)(?P<vol>\d{1,4})\s*(?:\(\s*\d{1,4}\s*\))?\s*:\s*(?P<page>[A-Za-z]?\d+)"
    r"(?:\s*[\u2013\u2014-]+\s*[A-Za-z]?\d+)?")
_VOL_ONLY = re.compile(r"\bvol(?:ume)?\.?\s*(\d{1,4})\b", re.I)
_PAGE_ONLY = re.compile(r"\bpp?\.\s*([A-Za-z]?\d+)", re.I)


def extract_volume_pages(raw: str) -> tuple[str, str]:
    """Volume and first page as written in the citation string, or ("", "")."""
    if not raw:
        return "", ""
    m = _VOLPAGE.search(raw)
    if m:
        return m.group("vol"), m.group("page")
    v = _VOL_ONLY.search(raw)
    pg = _PAGE_ONLY.search(raw)
    return (v.group(1) if v else ""), (pg.group(1) if pg else "")


# A volume or page number: up to six digits with an optional letter on either side.
_LOCATOR_SHAPE = re.compile(r"[A-Za-z]?\d{1,6}[A-Za-z]?$")


def _is_locator(s: str) -> bool:
    return bool(_LOCATOR_SHAPE.fullmatch((s or "").strip()))


def _locator_eq(a: str, b: str) -> bool:
    """Compare volume or page strings, ignoring leading zeros and the end of a page range."""
    norm = lambda x: re.sub(r"^0+(?=\d)", "", (x or "").strip().lower())
    a, b = norm(a), norm(str(b).split("-")[0].split("\u2013")[0])
    return bool(a) and bool(b) and a == b


# An apostrophe inside a word, dropped from names before comparison.
_INNER_APOS = re.compile(r"(?<=\w)['\u2019`\u00b4](?=\w)")


def surname(name: str) -> str:
    """Accent- and case-folded surname of "First Last", "Last, First" or "Last F"."""
    if not name:
        return ""
    name = _INNER_APOS.sub("", _ETAL.sub("", clean_text(name).strip()))
    if "," in name:
        name = name.split(",", 1)[0]
    # Join hyphenated names into one token.
    name = re.sub(r"(?<=\w)-(?=\w)", "", name)
    # Case is kept here for the trailing-initials test below.
    parts = [p for p in _PUNCT.sub(" ", name).split() if p]
    if not parts:
        return ""
    # Drop suffixes; with trailing initials ("Smith J") the surname is the first token.
    while len(parts) > 1 and _SUFFIX.match(parts[-1]):
        parts.pop()
    pick = parts[0] if len(parts) > 1 and _TRAILING_INITIALS.fullmatch(parts[-1]) \
        else parts[-1]
    return strip_accents(pick.lower())


def surname_keys(name: str) -> set[str]:
    """Surname variants for matching: the full surname and, if hyphenated, its last part."""
    keys = {surname(name)}
    n = _INNER_APOS.sub("", _ETAL.sub("", clean_text(name or "").strip()))
    n = n.split(",", 1)[0] if "," in n else n
    toks = [p for p in n.split() if p]
    while len(toks) > 1 and _SUFFIX.match(_PUNCT.sub("", toks[-1])):
        toks.pop()
    last = toks[-1:] or [""]
    if "-" in last[0]:                              # "Smith-Jones" is also cited as "Jones"
        keys.add(strip_accents(_PUNCT.sub("", last[0].rsplit("-", 1)[1]).lower()))
    return {k for k in keys if len(k) > 1 and k not in _NOT_SURNAMES}


def same_surname(a: str, b: str) -> bool:
    return bool(surname_keys(a) & surname_keys(b))


def surnames(names: list[str]) -> set[str]:
    return {k for n in names for k in surname_keys(n)}


def given_name(name: str) -> str:
    """Folded given name(s), i.e. everything except the surname; "" if none."""
    if not name:
        return ""
    n = _INNER_APOS.sub("", _ETAL.sub("", clean_text(name).strip()))
    if "," in n:                      # "Last, First": the given name follows the comma
        rest = n.split(",", 1)[1]
    else:
        parts = [p for p in _PUNCT.sub(" ", n).split() if p]
        # Drop generational suffixes ("Jr", "III") first.
        while len(parts) > 1 and _SUFFIX.match(parts[-1]):
            parts.pop()
        if len(parts) < 2:
            return ""
        # All trailing initials ("Smith J", "Smith C A") belong to the given name.
        k = len(parts)
        while k > 1 and _TRAILING_INITIALS.fullmatch(parts[k - 1]):
            k -= 1
        rest = " ".join(parts[k:]) if k < len(parts) else " ".join(parts[:-1])
    # Split run-together initials ("JA" -> "J A") unless the whole name is upper-case.
    if not n.replace(".", "").replace(",", "").replace(" ", "").replace("-", "").isupper():
        rest = " ".join(" ".join(t) if (len(t) <= 3 and t.isupper() and t.isalpha()) else t
                        for t in _PUNCT.sub(" ", rest).split())
    return strip_accents(_PUNCT.sub(" ", rest).strip().lower())


def edit_distance_at_most(left: str, right: str, max_distance: int) -> bool:
    """Bounded Levenshtein — bails out as soon as the whole row exceeds the cap."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > max_distance:
        return False
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, rc in enumerate(right, start=1):
            value = min(current[j - 1] + 1, previous[j] + 1,
                        previous[j - 1] + (0 if lc == rc else 1))
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


# Transliterated umlauts and eszett ("oe", "ue", "ae", "ss"), folded to one letter.
_TRANSLIT = ((("oe", "o"), ("ue", "u"), ("ae", "a"), ("ss", "s")))


def _fold_given(s: str) -> str:
    for a, b in _TRANSLIT:
        s = s.replace(a, b)
    return s


def names_match(cited: str, retrieved: str) -> bool:
    """Same person: surnames agree and given names or initials are compatible."""
    if not same_surname(cited, retrieved):
        # Same name tokens in either order (family name first or last).
        ta = sorted(strip_accents(t.lower()) for t in _PUNCT.sub(" ", clean_text(cited)).split())
        tb = sorted(strip_accents(t.lower()) for t in _PUNCT.sub(" ", clean_text(retrieved)).split())
        return len(ta) >= 2 and ta == tb and all(len(t) > 1 for t in ta)
    a, b = given_name(cited), given_name(retrieved)
    if not a or not b:
        return True                   # one side has no given name
    ta = [_fold_given(t) for t in a.split()]
    tb = [_fold_given(t) for t in b.split()]
    fa, fb = ta[0], tb[0]

    # If either side has only initials, any shared initial is a match.
    if all(len(t) == 1 for t in ta) or all(len(t) == 1 for t in tb):
        inits_a = {t[0] for t in ta}
        inits_b = {t[0] for t in tb}
        return bool(inits_a & inits_b)

    # Any spelled-out given name in common is a match.
    words_a = {t for t in ta if len(t) > 1}
    words_b = {t for t in tb if len(t) > 1}
    if words_a & words_b:
        return True

    if len(fa) == 1 or len(fb) == 1:  # one side leads with an initial
        return fa[0] == fb[0]
    # First given names match if equal, one is a prefix of the other, or one edit apart.
    return (fa == fb or fa.startswith(fb) or fb.startswith(fa)
            or edit_distance_at_most(fa, fb, 1))


URL_RE = re.compile(r"https?://[^\s,;)\]}>]+", re.I)
_TLD = (r"(?:com|org|net|edu|gov|io|ai|co|uk|de|cn|jp|fr|ch|info|dev|app|xyz|me|us|eu|ac|"
        r"edu\.[a-z]{2}|ac\.[a-z]{2}|co\.[a-z]{2}|gov\.[a-z]{2}|org\.[a-z]{2})")
# Characters stripped from the end of an extracted URL.
_URL_TAIL = ".,;:)\]}>'\""


def extract_url(text: str) -> Optional[str]:
    """First URL in the text, after closing spaces left by PDF line wrapping."""
    if not text:
        return None
    # Close line-wrap spaces after "/" or ":" and inside hostnames ("www. example.org").
    despaced = re.sub(r"(?<=[/:])\s+", "", text)
    despaced = re.sub(r"(?<=\.)\s+(?=[a-z0-9-]+\.[a-z0-9])", "", despaced)
    # Close spaces around the dot before a top-level domain ("doi . org", "doi. org").
    despaced = re.sub(r"(?<=[a-z0-9])\s+\.\s+(?=" + _TLD + r"(?:[/:?#]|\b))", ".", despaced)
    despaced = re.sub(r"(?<=\.)\s+(?=" + _TLD + r"(?:[/:?#]|\b))", "", despaced)
    m = URL_RE.search(despaced)
    if not m:
        return None
    url = m.group(0)
    # Cut at a dot followed by a capitalized word (the next sentence).
    url = re.split(r"\.(?=[A-Z][a-z])", url)[0]
    return url.rstrip(_URL_TAIL) or None


def same_resource(a: str, b: str) -> bool:
    """Compare host without "www." and path without trailing slash, case-insensitively."""
    if not a or not b:
        return False
    try:
        pa, pb = urlparse(a), urlparse(b)
    except Exception:
        return False
    ha = pa.netloc.lower().removeprefix("www.")
    hb = pb.netloc.lower().removeprefix("www.")
    if not ha or ha != hb:
        return False
    return pa.path.rstrip("/").lower() == pb.path.rstrip("/").lower()


ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.I)


def extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = ARXIV_RE.search(text.replace(" ", ""))
    return m.group(1) if m else None


# A line-wrap space inside a DOI suffix, before a digit: "10.1000/abc. 123".
_DOI_BREAK = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]*[.\-/_:])\s+(?=\d)")


def extract_doi(text: str) -> Optional[str]:
    if not text:
        return None
    for _ in range(3):
        joined = _DOI_BREAK.sub(r"\1", text)
        if joined == text:
            break
        text = joined
    m = DOI_RE.search(text)
    return m.group(1).rstrip(".,;") if m else None


# --------------------------------------------------------------------- record ---

@dataclass
class Work:
    """One candidate record returned by a connector."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    volume: str = ""
    pages: str = ""
    source: str = ""          # which connector produced this
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Source-name prefixes of web search hits and fetched pages.
_WEB_SOURCES = ("serpapi", "url", "wayback", "commoncrawl")


# Max "cited by" count for a Scholar [CITATION] entry to be treated as reference-only.
REFERENCE_ONLY_MAX_CITERS = 2


def reference_only(w: Work) -> bool:
    """Scholar [CITATION] or link-less entry with at most REFERENCE_ONLY_MAX_CITERS citers."""
    if w.source != "serpapi_scholar":
        return False
    ex = w.extra or {}
    if (ex.get("type") or "").lower() != "citation" and (w.url or "").strip():
        return False
    n = ex.get("cited_by")
    if n is None:
        return False
    try:
        return int(n) <= REFERENCE_ONLY_MAX_CITERS
    except (TypeError, ValueError):
        return False


def bare_record(w: Work) -> bool:
    """Reference-only entry, or a record with no authors and no registry identifier."""
    if reference_only(w):
        return True
    if w.authors:
        return False
    return not (w.doi or w.arxiv_id) or (w.source or "").startswith(_WEB_SOURCES)


@dataclass
class Query:
    """The citation being verified, as parsed from the raw string."""
    raw: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    # Volume and first page, from `extract_volume_pages`.
    volume: str = ""
    pages: str = ""
    # Parser used: "llm" or "regex".
    parser: str = ""
    # paper / book / repo / dataset / web / report / unknown; set by `parsing.build_query`.
    source_type: str = "unknown"
    # How `source_type` was set: "rules" or "llm".
    type_via: str = "rules"
    # URL given in the citation, if any.
    url: str = ""

    def __post_init__(self):
        # Split author strings joined by semicolons ("A. Smith; B. Jones").
        if any(";" in (a or "") for a in (self.authors or [])):
            self.authors = split_author_strings(self.authors)

    def to_dict(self) -> dict:
        return asdict(self)


def split_author_strings(authors: list) -> list:
    """Split semicolon-joined author strings; a piece "Last, F." becomes "F. Last"."""
    out = []
    for a in authors or []:
        for piece in (a or "").split(";"):
            piece = piece.strip().strip(",").strip()
            if not piece or piece.lower() in ("et al", "et al.", "and others", "others"):
                continue
            if piece.lower().endswith(" et al.") or piece.lower().endswith(" et al"):
                piece = piece.rsplit(" et al", 1)[0].strip()
            if piece.count(",") == 1:
                last, first = [x.strip() for x in piece.split(",")]
                if last and first:
                    piece = f"{first} {last}"
            out.append(piece)
    return out


# ----------------------------------------------------------------- comparison ---

@dataclass
class FieldMatch:
    """Per-field result: True = match, False = conflict, None = missing or ambiguous."""
    title: Optional[bool] = None
    authors: Optional[bool] = None
    year: Optional[bool] = None
    venue: Optional[bool] = None
    identifier: Optional[bool] = None
    locator: Optional[bool] = None          # volume / first page agreement
    title_similarity: float = 0.0
    title_containment: float = 0.0
    author_overlap: float = 0.0
    venue_score: float = 0.0
    # Cited authors whose surname is on the record but whose given name is not.
    tampered: list[str] = field(default_factory=list)
    # [authors, "authors" in ambiguous] before the foreign-author and initials checks.
    authors_base: Optional[list] = None
    # Cited authors that match nobody in a record whose author list is complete.
    foreign: list[str] = field(default_factory=list)
    # Same, on a complete record listing fewer authors than the citation.
    foreign_longer: list[str] = field(default_factory=list)
    # Cited authors whose initials fit nobody of that surname on the record.
    initial_conflict: list[str] = field(default_factory=list)
    # Fields whose score fell between the two thresholds.
    ambiguous: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Raw citation string; not compared.
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def conflicts(self) -> list[str]:
        return [k for k in ("title", "authors", "year", "venue", "identifier")
                if getattr(self, k) is False]

    @property
    def agreements(self) -> list[str]:
        return [k for k in ("title", "authors", "year", "venue", "identifier")
                if getattr(self, k) is True]


_ORG = re.compile(
    r"\b(administration|association|foundation|institute|institutes|university|"
    r"college|team|teams|contributors|community|inc|ltd|llc|corporation|corp|"
    r"lab|labs|laboratory|group|office|department|agency|commission|committee|academy|academies|"
    r"society|press|news|council|bureau|ministry|organization|organisation|"
    r"project|consortium|initiative|alliance|the white house|"
    r"google|openai|microsoft|anthropic|deepmind|nvidia|meta|apple|amazon|ibm|"
    r"intel|huawei|baidu|alibaba|tencent|xai|mistral|hugging face|github|"
    r"nist|noaa|nasa|unesco|who|oecd|imf|un|eu|epfl|mit|cmu)\b", re.I)


# Government bodies as authors.
_ORG_GOV = re.compile(r"\b(parliament|congress|senate|government|court|european union|"
                      r"european commission|united nations|state council)\b", re.I)


_ORG_AI = re.compile(r"\bAI\b")      # case-sensitive: "Acme AI", not the given name "Ai"


def _corporate(authors: list[str]) -> bool:
    """Whether the author list names organisations rather than people."""
    if len(authors) != 1:
        # Two or three authors: all must look corporate and at least one must match an org pattern.
        if 1 < len(authors) <= 3:
            org = lambda a: bool(_ORG.search(a or "") or _ORG_GOV.search(a or "") or _ORG_AI.search(a or ""))
            return all(org(a) or (a or "").strip().isupper() or " " not in (a or "").strip()
                       for a in authors) and any(org(a) for a in authors)
        return False
    a = authors[0].strip()
    # One author: an org pattern, a single token, or all capitals.
    return bool(_ORG.search(a) or _ORG_GOV.search(a) or _ORG_AI.search(a)) or " " not in a or a.isupper()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# Per field: (match at or above, conflict below); scores in between are ambiguous.
THRESHOLDS = {
    "title":   (0.85, 0.45),
    "authors": (0.60, 0.25),
    "venue":   (0.70, 0.30),
}

# Minimum token count of the shorter title for title containment to be scored.
MIN_CONTAINMENT_TOKENS = 4


def _graded(fm: "FieldMatch", score: float, field_name: str) -> Optional[bool]:
    """True, False, or None (added to `fm.ambiguous`) by the field's thresholds."""
    hi, lo = THRESHOLDS[field_name]
    if score >= hi:
        return True
    if score < lo:
        return False
    fm.ambiguous.append(field_name)
    return None


def author_match_score(cited: list[str], retrieved: list[str]) -> float:
    """Fraction of cited authors (truncation markers excluded) that match someone in the record."""
    usable = [a for a in cited
              if len(surname(a)) > 1 and surname(a) not in _NOT_SURNAMES]
    if not usable or not retrieved:
        return 0.0
    return sum(1 for a in usable
               if any(names_match(a, r) for r in retrieved)) / len(usable)


def tampered_authors(cited: list[str], retrieved: list[str]) -> list[str]:
    """Cited authors whose surname is on the record but whose spelled-out given name is not."""
    out = []
    for a in cited:
        s = surname(a)
        ga = given_name(a)
        # Skip cited names whose first given name is only an initial.
        if len(s) <= 1 or s in _NOT_SURNAMES or len(ga.split()[0] if ga else "") < 2:
            continue
        namesakes = [r for r in retrieved if same_surname(a, r)]
        if not namesakes or any(names_match(a, r) for r in namesakes):
            continue
        # Flag only if some namesake on the record has a spelled-out first given name.
        def _first(name: str) -> str:
            g = (given_name(name) or "").split()
            return g[0] if g else ""
        if any(len(_first(r)) > 1 for r in namesakes):
            out.append(a)
    return out


# Sources whose records list every author of a work.
COMPLETE_AUTHOR_SOURCES = frozenset({"arxiv", "crossref", "openalex", "aclanthology",
                                     "europepmc", "dblp", "pubmed", "semanticscholar",
                                     "datacite", "googlebooks"})
_SCHOLAR_CUT = re.compile(r"\u2026|\.\.\.")


def complete_author_list(w: "Work") -> bool:
    """Complete-author source, or a Scholar summary whose author run has no ellipsis."""
    if w.source in COMPLETE_AUTHOR_SOURCES:
        return True
    if w.source == "serpapi_scholar":
        head = ((w.extra or {}).get("summary") or "").split(" - ")[0]
        return bool(head.strip()) and not _SCHOLAR_CUT.search(head)
    return False


# Maximum number of cited authors for the foreign-author and initials checks.
FOREIGN_MAX_CITED = 15


def _dl_distance(a: str, b: str, cap: int = 3) -> int:
    """Damerau-Levenshtein distance (adjacent transposition counts once), capped."""
    if abs(len(a) - len(b)) >= cap:
        return cap
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return min(prev[-1], cap)


def _first_initial(name: str) -> str:
    g = (given_name(name) or "").strip()
    return strip_accents(g[:1].lower()) if g else ""


def _close_surname(a: str, b: str) -> bool:
    """Surnames that differ only by a typo or truncation (edit limits grow with length)."""
    ia, ib = _first_initial(a), _first_initial(b)
    for x in surname_keys(a):
        for y in surname_keys(b):
            n = min(len(x), len(y))
            if n >= 5 and len(x) == len(y) and sorted(x) == sorted(y) and _dl_distance(x, y) == 1:
                return True                                   # adjacent transposition
            if n >= 6 and _dl_distance(x, y) <= 1:
                return True
            if n >= 7 and ia and ia == ib and _dl_distance(x, y) <= 2:
                return True
            lo, hi = sorted((x, y), key=len)
            if len(lo) >= 4 and hi.startswith(lo):
                return True
    return False


def _initials(name: str) -> list[str]:
    g = re.sub(r"[^\w\s-]", " ", given_name(name) or "")
    out = []
    for t in re.split(r"[\s-]+", g):
        if not t:
            continue
        # "CA" is two initials run together; "Ca" or "Carlos" is one name.
        out.extend(strip_accents(ch.lower()) for ch in t) if (len(t) <= 3 and t.isupper()) \
            else out.append(strip_accents(t[0].lower()))
    return out


def initial_conflicts(cited: list[str], record: list[str]) -> list[str]:
    """Cited authors sharing no initial with any record author of the same surname."""
    out = []
    for a in cited:
        ia = _initials(a)
        sakes = [b for b in record if same_surname(a, b)]
        if not ia or not sakes or any(names_match(a, b) for b in sakes):
            continue
        if not any((not _initials(b)) or (set(ia) & set(_initials(b))) for b in sakes):
            out.append(a)
    return out


def initials_only_conflicts(cited: list[str], record: list[str]) -> list[str]:
    """Initials-only cited authors fitting no namesake on the record (swaps excepted)."""
    def is_initials(a: str) -> bool:
        g = re.sub(r"[^\w\s-]", " ", given_name(a) or "")
        toks = [t for t in re.split(r"[\s-]+", g) if t]
        return bool(toks) and all(len(t) == 1 or (len(t) <= 3 and t.isupper()) for t in toks)

    out = []
    for i, a in enumerate(cited):
        if not is_initials(a):
            continue
        ia = set(_initials(a))
        sakes = [b for b in record if same_surname(a, b)]
        if not ia or not sakes:
            continue
        if any((not _initials(b)) or (ia & set(_initials(b))) for b in sakes):
            continue
        others = [b for b in record if not same_surname(a, b)
                  and any(same_surname(c, b) for j, c in enumerate(cited) if j != i)]
        if any(ia & set(_initials(b)) for b in others):
            continue                      # initials swapped between cited authors
        out.append(a)
    return out


def foreign_authors(cited: list[str], record: list[str]) -> list[str]:
    """Cited authors who match nobody in `record` under any of the name tolerances."""
    def words(n: str) -> set:
        return {t for t in _PUNCT.sub(" ", strip_accents(n or "").lower()).split() if len(t) >= 3}

    def either_order(a: str, b: str) -> bool:
        # Either surname appears among the other name's words (name order differs).
        return surname(a) in words(b) or surname(b) in words(a)

    return [a for a in cited
            if not any(names_match(a, b) or same_surname(a, b) or _close_surname(a, b)
                       or either_order(a, b) for b in record)]


_ELLIPSIS = re.compile(r"\s*(\.\.\.|\u2026)\s*$")


def truncated_prefix(record_title: str, cited_title: str) -> bool:
    """`record_title` ends in an ellipsis and its 4+ tokens are a prefix of `cited_title`."""
    if not _ELLIPSIS.search(record_title or ""):
        return False
    rt = title_tokens(_ELLIPSIS.sub("", record_title))
    ct = title_tokens(cited_title)
    return len(rt) >= 4 and ct[:len(rt)] == rt


def compare(q: Query, w: Work, *, year_tolerance: int = 2) -> FieldMatch:
    """Field-by-field comparison of a citation against one candidate record."""
    fm = FieldMatch()
    fm.raw = getattr(q, "raw", "") or ""

    # -- title ---------------------------------------------------------------
    qt, wt = title_tokens(q.title), title_tokens(w.title)
    if qt and wt:
        qs, ws = set(qt), set(wt)
        # With hyphens present, also score the glued tokens and keep the better overlap.
        if "-" in (q.title or "") or "-" in (w.title or ""):
            gq, gw = set(title_tokens(q.title, glue=True)), set(title_tokens(w.title, glue=True))
            if len(gq & gw) / max(len(gq), len(gw)) > len(qs & ws) / max(len(qs), len(ws)):
                qs, ws = gq, gw
        inter = len(qs & ws)
        shorter = min(len(qs), len(ws))
        fm.title_similarity = inter / max(len(qs), len(ws))
        # Shorter-title containment; 0.0 below MIN_CONTAINMENT_TOKENS (raw ratio in notes).
        short_but_fully_contained = False
        if shorter >= MIN_CONTAINMENT_TOKENS:
            fm.title_containment = inter / shorter
        else:
            raw = inter / shorter
            fm.notes.append(
                f"containment suppressed: shorter title has {shorter} token(s), "
                f"raw ratio would be {raw:.2f}")
            short_but_fully_contained = raw >= 0.99

        if norm_title(q.title) == norm_title(w.title) or \
                norm_title(q.title, glue=True) == norm_title(w.title, glue=True):
            fm.title = True
        elif truncated_prefix(w.title, q.title):
            fm.title = True
            fm.notes.append("record title is the cited title cut short by an ellipsis")
        else:
            fm.title = _graded(fm, fm.title_similarity, "title")
            # A short title fully contained in the other is left ambiguous.
            if short_but_fully_contained and fm.title is not True:
                fm.title = None
                if "title" not in fm.ambiguous:
                    fm.ambiguous.append("title")
            elif fm.title_containment >= 0.95:
                fm.notes.append("one title is contained in the other — likely truncated")
            elif fm.title_similarity >= 0.9:
                fm.notes.append("title differs only slightly — inspect wording")

    # -- authors -------------------------------------------------------------
    # Corporate authors are not compared; `authors` stays None.
    if q.authors and w.authors and not _corporate(q.authors):
        fm.author_overlap = author_match_score(q.authors, w.authors)
        fm.tampered = tampered_authors(q.authors, w.authors)
        fm.authors = _graded(fm, fm.author_overlap, "authors")
        # Cited names with a usable surname (same filter as `author_match_score`).
        cited = [a for a in q.authors if len(surname(a)) > 1 and surname(a) not in _NOT_SURNAMES]
        rec_surnames = surnames(w.authors)
        foreign = [a for a in cited if surname(a) not in rec_surnames]
        if fm.tampered and not foreign:
            # Every cited surname is on the record and only given names differ: a match.
            fm.authors = True
            fm.ambiguous = [f for f in fm.ambiguous if f != "authors"]
            fm.notes.append("given names differ but every cited surname is in the record: "
                            + ", ".join(fm.tampered))
        elif fm.tampered and fm.authors is not False:
            # Given-name mismatches turn a match into ambiguous.
            fm.authors = None
            if "authors" not in fm.ambiguous:
                fm.ambiguous.append("authors")
            fm.notes.append("cited author(s) share a surname with the record but "
                            "not the given name: " + ", ".join(fm.tampered))
        fm.authors_base = [fm.authors, "authors" in fm.ambiguous]
        # Papers: a cited author missing from a complete record at least as long is a conflict.
        if (fm.title is True
                and getattr(q, "source_type", "") == "paper"
                and complete_author_list(w)
                and 1 <= len(cited) <= FOREIGN_MAX_CITED and len(w.authors) >= len(cited)):
            fm.foreign = foreign_authors(cited, w.authors)
            if fm.foreign:
                fm.authors = False
                fm.ambiguous = [f for f in fm.ambiguous if f != "authors"]
                fm.notes.append(f"not on the record's author list ({len(w.authors)} authors, "
                                f"{w.source}): " + ", ".join(fm.foreign))
        # More cited authors than on this complete record: store the extras in `foreign_longer`.
        if (fm.title is True
                and getattr(q, "source_type", "") == "paper"
                and complete_author_list(w) and len(w.authors) >= 3
                and 2 <= len(cited) <= FOREIGN_MAX_CITED and len(w.authors) < len(cited)
                and not any("et al" in a.lower() or "others" in a.lower() for a in q.authors)
                and any(names_match(cited[0], b) or same_surname(cited[0], b) for b in w.authors)):
            extra = [a for a in foreign_authors(cited, w.authors) if not _corporate([a])]
            if extra and extra[0] != cited[0]:
                fm.foreign_longer = extra
                fm.notes.append(f"citation names {len(cited)} authors, this complete record "
                                f"{len(w.authors)}; not on it: " + ", ".join(extra))
        # Tampered authors sharing no initial with any namesake on a complete record: a conflict.
        if (fm.title is True and fm.tampered
                and getattr(q, "source_type", "") == "paper"
                and w.source in COMPLETE_AUTHOR_SOURCES
                and 1 <= len(cited) <= FOREIGN_MAX_CITED and len(w.authors) >= len(cited)):
            fm.initial_conflict = initial_conflicts(fm.tampered, w.authors)
            if fm.initial_conflict:
                fm.authors = False
                fm.ambiguous = [f for f in fm.ambiguous if f != "authors"]
                fm.notes.append("given name shares no initial with the record's author of "
                                "that surname: " + ", ".join(fm.initial_conflict))
        # Initials-only cited authors that fit no namesake on the record: a conflict.
        if (getattr(q, "source_type", "") == "paper"
                and 1 <= len(cited) <= FOREIGN_MAX_CITED):
            hits = initials_only_conflicts(cited, w.authors)
            if hits:
                fm.initial_conflict = list(fm.initial_conflict) + hits
                fm.authors = False
                fm.ambiguous = [f for f in fm.ambiguous if f != "authors"]
                fm.notes.append("cited initials fit no record author of that surname: "
                                + ", ".join(hits))
        # Books: record authors that are a strict subset of the cited authors count as a match.
        if (fm.authors is not True
                and getattr(q, "source_type", "") == "book" and fm.title is True
                and w.authors and len(w.authors) < len(cited)
                and all(any(names_match(b, a) for a in cited) for b in w.authors)):
            fm.authors = True
            fm.ambiguous = [f for f in fm.ambiguous if f != "authors"]
            fm.notes.append("every author of this edition is cited; the extra cited names are "
                            "read as another edition's co-authors")
        if fm.author_overlap == 0.0:
            fm.notes.append("no cited surname appears in the retrieved record")

    # -- volume / first page -------------------------------------------------
    # Compared only when the title matches and both sides have a well-formed value.
    if fm.title is True:
        vq, pq = (q.volume or ""), (q.pages or "")
        vw, pw = (w.volume or ""), (w.pages or "")
        checks = [(vq, vw, "volume"), (pq, pw, "first page")]
        seen = [(a, b, n) for a, b, n in checks
                if a and b and _is_locator(a) and _is_locator(str(b).split("-")[0])]
        if seen:
            bad = [n for a, b, n in seen if not _locator_eq(a, b)]
            fm.locator = not bad
            if bad:
                fm.conflicts.append("locator")
                fm.notes.append("cited " + " and ".join(bad) + " disagree with the record: "
                                + ", ".join(f"{n} {a!r} vs {b!r}" for a, b, n in seen))
            else:
                fm.agreements.append("locator")

    # -- year ----------------------------------------------------------------
    if q.year and w.year:
        fm.year = abs(q.year - w.year) <= year_tolerance

    # -- venue ---------------------------------------------------------------
    if q.venue and w.venue:
        if venues_equivalent_heuristic(q.venue, w.venue):
            fm.venue_score, fm.venue = 1.0, True
        else:
            fm.venue_score = similarity(normalize_venue(q.venue), normalize_venue(w.venue))
            fm.venue = _graded(fm, fm.venue_score, "venue")

    # -- identifier ----------------------------------------------------------
    if q.doi and w.doi:
        a, b = q.doi.lower().rstrip("/"), w.doi.lower().rstrip("/")
        fm.identifier = a == b
        # A DOI that is a prefix of the other (10+ chars, with a "/") is treated as truncated.
        if not fm.identifier:
            short, long = sorted((a, b), key=len)
            if "/" in short and len(short) >= 10 and long.startswith(short):
                fm.identifier = None
                fm.notes.append("identifier not compared: the cited DOI is a prefix "
                                "of the record's, i.e. truncated rather than different")
    elif q.arxiv_id and w.arxiv_id:
        fm.identifier = q.arxiv_id.split("v")[0] == w.arxiv_id.split("v")[0]

    # -- book editions ----------------------------------------------------------
    # Books with a matching title: a venue or year conflict becomes not compared.
    if (getattr(q, "source_type", "") or "") == BOOK and fm.title is True:
        for field in ("venue", "year"):
            if getattr(fm, field) is False:
                setattr(fm, field, None)
                fm.notes.append(f"{field} not compared: editions of a book differ in {field}")

    return fm
