"""Classify a citation's source type with regex rules, falling back to an LLM."""
from __future__ import annotations

import re
from typing import Optional

from .runlog import get as _get_logger

log = _get_logger("srctype")

PAPER, BOOK, REPO, DATASET, WEB, REPORT, UNKNOWN = (
    "paper", "book", "repo", "dataset", "web", "report", "unknown")

# types the academic connectors index
COVERED_BY_ACADEMIC_APIS = frozenset({PAPER})

# --- book -------------------------------------------------------------------
# publisher imprints (not the bare word "press")
_IMPRINT = re.compile(
    r"\b("
    r"wiley|springer(?:[- ]verlag)?|elsevier|mcgraw[- ]?hill|routledge|blackwell|"
    r"o'?reilly|butterworth[- ]?heinemann|longmans|basic books|prentice[- ]hall|"
    r"addison[- ]wesley|morgan kaufmann|academic press|pergamon|thieme|"
    r"crc press|chapman (?:and|&) hall|no starch|manning|apress|packt|"
    r"harvard university press|yale university press|princeton university press|"
    r"(?:mit|cambridge|oxford|stanford|chicago|columbia|citic) (?:university )?press|"
    r"[a-z]+ university press|"
    r"cup archive|dover publications|pearson|cengage|sage publications|wolters kluwer|"
    r"bloomsbury|gulf professional|csli publications|world scientific|de gruyter|"
    r"palgrave|macmillan|penguin|random house|simon (?:and|&) schuster|hachette"
    r")\b", re.I)
_BOOKISH = re.compile(
    r"\b(isbn\b|\d(?:st|nd|rd|th) ed(?:ition|\.)|handbook of|encyclopedia|"
    r"encyclopaedia|textbook|lecture notes in|chapter \d+)\b", re.I)

# --- repo -------------------------------------------------------------------
_REPO = re.compile(r"\b(github\.com|gitlab\.com|bitbucket\.org|huggingface\.co/(?!datasets))", re.I)

# --- dataset ----------------------------------------------------------------
# dataset hosting platforms
_DATASET = re.compile(
    r"\b(kaggle(?:\.com)?|zenodo|figshare|dryad|openml|physionet|"
    r"huggingface\.co/datasets|data\.gov|dataverse)\b", re.I)

# --- report -----------------------------------------------------------------
_REPORT = re.compile(
    r"\b(technical report|tech\. rep\.|white ?paper|working paper|"
    r"(?:ph\.?d\.?|master'?s|doctoral|bachelor'?s) (?:thesis|dissertation)|dissertation|"
    r"rfc \d+|iso[/ ]?\d|ieee std|nist sp|policy brief|guidebook)\b"
    r"|\b[a-z0-9-]+\.gov\b|\bthe white house\b|\bnational (?:oceanic|institutes?|bureau)\b",
    re.I)
# a university or institute as publisher: after a sentence break, before the year
_UNIVERSITY_PUBLISHER = re.compile(
    r"\.\s*(?:[A-Z][\w.&-]*\s+){0,4}"
    r"(?:University|Institute of Technology|College|Polytechnic)"
    r"\b[^.]{0,60}(?:19|20)\d\d")

# a spelled-out month before the year, e.g. "June 2024"
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(?:19|20)\d\d\b", re.I)

_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
# hosts that publish papers
_ACADEMIC_HOST = re.compile(
    r"\b(arxiv\.org|biorxiv|medrxiv|aclanthology|openreview|proceedings\.\w+|"
    r"dl\.acm\.org|ieeexplore|link\.springer|sciencedirect|nature\.com|"
    r"science\.org|pubmed|ncbi\.nlm|doi\.org|jmlr\.org|mlr\.press)\b", re.I)

# --- web --------------------------------------------------------------------
_WEB = re.compile(
    r"\b(blog|weblog|newsletter|press release|documentation|docs?\.|"
    # `\d+` so that the closing `\b` can match after a whole number
    r"retrieved from|\[?accessed:?(?: on)?\s*\d+|\bnews\b|"
    r"medium\.com|substack\.com|wikipedia\.org|stackoverflow\.com|reddit\.com|"
    r"youtube\.com|twitter\.com|x\.com/)\b", re.I)

# --- paper ------------------------------------------------------------------
_PAPER_VENUE = re.compile(
    r"\b(in proceedings of|proceedings of the|"
    r"journal of|transactions on|annals of|"
    r"conference on|workshop on|symposium on|"
    r"nature \w+|lancet|nejm|jama|plos|bmj|"
    r"neurips|nips\b|icml|iclr|aaai|ijcai|emnlp|naacl|coling|cvpr|iccv|eccv|"
    r"kdd\b|sigir|uist|osdi|sosp|nsdi|usenix|infocom|sigcomm|sigmod|vldb"
    r")\b", re.I)
# short or common venue names, matched only against the parsed venue field
_VENUE_ONLY = re.compile(
    r"^\W*(nature|science|cell|acl|www|chi|letters|review [a-e])\b", re.I)
_PREPRINT = re.compile(r"\barxiv\b|\bbiorxiv\b|\bmedrxiv\b|\bpreprint\b|\bssrn\b", re.I)
# volume(issue):page, volume:first–last, or pp. first–last, e.g. "6(1-2):3–15", "2:303–360"
_VOLPAGES = re.compile(
    r"\b\d+\s*\(\s*[\d\-–]+\s*\)\s*[:,]\s*\d+"
    r"|\b\d{1,4}\s*:\s*\d{1,5}\s*[-–]\s*\d{1,5}\b"
    r"|\bpp?\.\s*\d+\s*[-–]\s*\d+")




def classify(raw: str, venue: str = "", doi: str = "", arxiv_id: str = "") -> str:
    """Source type of a citation from its reference string and parsed fields; rules apply in order."""
    # drop whitespace after "/" or ":" (line wraps inside URLs)
    blob = re.sub(r"(?<=[/:])\s+", "", f"{raw} {venue}")
    urls = " ".join(_URL.findall(blob))
    # the text with URLs removed
    prose = _URL.sub(" ", blob)

    academic = bool(_ACADEMIC_HOST.search(urls))
    if arxiv_id or _PREPRINT.search(prose) or academic:
        return PAPER
    if _REPO.search(blob):
        return REPO
    if _DATASET.search(blob):
        return DATASET
    # volume/issue/pages marks a journal article, even with a publisher name or URL present
    if _VOLPAGES.search(prose):
        return PAPER
    if _IMPRINT.search(prose) or _BOOKISH.search(prose):
        return BOOK
    if _REPORT.search(blob) or _UNIVERSITY_PUBLISHER.search(prose):
        return REPORT
    # a named venue takes precedence over a non-academic URL
    if _PAPER_VENUE.search(prose) or (venue and _VENUE_ONLY.search(venue)):
        return PAPER
    # a non-academic URL or a spelled-out month marks a web page
    if (urls and not _ACADEMIC_HOST.search(urls)) or _MONTH_YEAR.search(prose):
        return WEB
    # checked after the imprint rule, since book chapters also have DOIs
    if doi:
        return PAPER
    if _PAPER_VENUE.search(prose) or (venue and _VENUE_ONLY.search(venue)):
        return PAPER
    if _VOLPAGES.search(prose):
        return PAPER
    if _WEB.search(prose):
        return WEB
    # fallback: a leading list of personal names marks a paper
    if _looks_like_author_list(prose):
        return PAPER
    return UNKNOWN


# a personal name: one to five capitalised words or initials, e.g. "J. S. Smith", "Jean-Paul Doe"
_TOKEN = r"(?:[A-Z]\.?|[A-Z][\w'\u00c0-\u024f]*(?:-[\w'\u00c0-\u024f]+)*)"
_PERSON = re.compile(rf"^{_TOKEN}(?:\s+{_TOKEN}){{0,4}}$")
# journal or venue names, checked when only one personal name is found
_JOURNALISH = re.compile(
    r"\b(neurocomputing|biometrika|econometrica|bioinformatics|nature|science|cell|lancet|"
    r"plos \w+|nucleic acids research|frontiers in|"
    r"[a-z]+ (?:journal|review|letters|transactions|studies|quarterly|informatics)|"
    r"journal of|annals of|proceedings of|"
    # a conference acronym followed by a year, e.g. "ACL 2023", "NeurIPS'19"
    r"(?:acl|emnlp|naacl|coling|neurips|nips|icml|iclr|cvpr|iccv|eccv|aaai|ijcai|"
    r"kdd|sigir|www|chi|uai|aistats|colt|interspeech|icassp)\s*['\u2019]?\s*\d{2,4})\b", re.I)


def _looks_like_author_list(text: str) -> bool:
    """True if the text opens with two or more personal names, or one name with "et al." or a venue."""
    # the author block ends at the first period after a word (not after an initial)
    head = re.split(r"(?<=[A-Za-z\u00c0-\u024f]{2})\.\s+(?=[A-Z\"\u201c])|\?\s", text, maxsplit=1)[0]
    etal = bool(re.search(r"\bet al\b", head, re.I))
    head = re.sub(r",?\s*(?:and\s+)?et al\.?", "", head, flags=re.I)
    parts = [p.strip(" .,") for p in re.split(r",\s*|\s+and\s+", head) if p.strip(" .,")]
    names = sum(1 for p in parts[:8] if _PERSON.match(p))
    if names >= 2:
        return True
    if names == 1 and etal:
        return True
    return names == 1 and bool(_JOURNALISH.search(text))


_TYPES = frozenset({PAPER, BOOK, REPO, DATASET, WEB, REPORT, UNKNOWN})


CLASSIFY_PROMPT = """\
What kind of work does this reference string cite? Rules could not tell.

REFERENCE:
{citation}

Pick exactly one:

  paper    an article: journal, conference, workshop, preprint, or working paper.
           A single author and an unfamiliar journal name still make it a paper.
  book     a monograph, textbook, handbook, dictionary, or edited volume —
           whoever the publisher is, and even if no publisher is named.
  report   a thesis or dissertation, a standard, a government or institutional
           document, a technical report, a manual.
  web      a blog post, news story, product or documentation page, model card,
           announcement — anything read on a website rather than published.
  dataset  a dataset, corpus, benchmark distribution, or problem set.
  repo     a software repository or software package as such.
  unknown  only if the string genuinely does not say — for instance a bare
           title with no author, venue, publisher, or address.

Judge the *form* of the work from how the reference is written. Do not judge
whether the work exists — that is decided elsewhere, and guessing here would
prejudge it.

Reply with JSON only:
{{"type": "paper|book|report|web|dataset|repo|unknown", "reason": "a few words"}}"""


def classify_llm(raw: str, client) -> str:
    """Source type from an LLM for a citation the rules left `unknown`; `unknown` on any failure."""
    if client is None:
        return UNKNOWN
    import json
    from .llm import chat
    try:
        text = chat(client, CLASSIFY_PROMPT.format(citation=raw))
        body = text.choices[0].message.content or ""
        start, end = body.find("{"), body.rfind("}")
        kind = json.loads(body[start:end + 1]).get("type", UNKNOWN)
        kind = str(kind).strip().lower()
        if kind in _TYPES:
            log.debug("source_type: rules said unknown, model says %s", kind)
            return kind
        log.warning("source_type: model returned %r, keeping unknown", kind)
    except Exception as exc:
        log.warning("source_type: model classification failed (%s); keeping unknown",
                    type(exc).__name__)
    return UNKNOWN


def absence_is_evidence(source_type: str) -> bool:
    """True if the academic connectors index this source type."""
    return source_type in COVERED_BY_ACADEMIC_APIS
