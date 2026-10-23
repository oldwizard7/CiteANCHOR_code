"""Venue and text normalization: alias table, acronym and word-truncation heuristics."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


VENUE_ALIASES = {
    # NeurIPS
    "nips": "neurips",
    "neurips": "neurips",
    "advances in neural information processing systems": "neurips",
    # ICML
    "icml": "icml",
    "international conference on machine learning": "icml",
    "proceedings of the international conference on machine learning": "icml",
    # ICLR
    "iclr": "iclr",
    "international conference on learning representations": "iclr",
    # CVPR
    "cvpr": "cvpr",
    "ieee conference on computer vision and pattern recognition": "cvpr",
    "ieee/cvf conference on computer vision and pattern recognition": "cvpr",
    # ECCV / ICCV
    "eccv": "eccv",
    "european conference on computer vision": "eccv",
    "iccv": "iccv",
    "ieee international conference on computer vision": "iccv",
    # ACL
    "acl": "acl",
    "association for computational linguistics": "acl",
    "annual meeting of the association for computational linguistics": "acl",
    # EMNLP
    "emnlp": "emnlp",
    "empirical methods in natural language processing": "emnlp",
    # NAACL
    "naacl": "naacl",
    "north american chapter of the association for computational linguistics": "naacl",
    # AAAI
    "aaai": "aaai",
    "aaai conference on artificial intelligence": "aaai",
    # AI journal
    "artificial intelligence": "ai_journal",
    "artif intell": "ai_journal",
    # Nature family
    "nat": "nature",
    "nature": "nature",
    # Science
    "science": "science",
    # JMLR
    "jmlr": "jmlr",
    "journal of machine learning research": "jmlr",
    "j mach learn res": "jmlr",
    "j of mach learn res": "jmlr",
    # PMLR
    "pmlr": "pmlr",
    "proceedings of machine learning research": "pmlr",
    # SIAM journals
    "sicomp": "siam_j_comput",
    "siam j comput": "siam_j_comput",
    "siam journal on computing": "siam_j_comput",
    "siam j optim": "siam_j_optim",
    "siam journal on optimization": "siam_j_optim",
    "siam j numer anal": "siam_j_numer_anal",
    "siam journal on numerical analysis": "siam_j_numer_anal",
    # COLT
    "colt": "colt",
    "conference on learning theory": "colt",
    "annual conference computational learning theory": "colt",
    "annual conference on computational learning theory": "colt",
    "annual conference on learning theory": "colt",
    # AISTATS
    "aistats": "aistats",
    "international conference on artificial intelligence and statistics": "aistats",
    "artificial intelligence and statistics": "aistats",
    # UAI
    "uai": "uai",
    "conference on uncertainty in artificial intelligence": "uai",
    # KDD
    "kdd": "kdd",
    "sigkdd": "kdd",
    "acm sigkdd": "kdd",
    "knowledge discovery and data mining": "kdd",
    # IJCAI
    "ijcai": "ijcai",
    "international joint conference on artificial intelligence": "ijcai",
    # AAMAS
    "aamas": "aamas",
    "autonomous agents and multiagent systems": "aamas",
    # STOC / FOCS
    "stoc": "stoc",
    "acm symposium on theory of computing": "stoc",
    "symposium on theory of computing": "stoc",
    "focs": "focs",
    "ieee symposium on foundations of computer science": "focs",
    "foundations of computer science": "focs",
    # SODA
    "soda": "soda",
    "acm siam symposium on discrete algorithms": "soda",
    "symposium on discrete algorithms": "soda",
    # TPAMI / PAMI
    "tpami": "tpami",
    "pami": "tpami",
    "ieee transactions on pattern analysis and machine intelligence": "tpami",
    "ieee trans pattern anal mach intell": "tpami",
    # Mathematics of Operations Research
    "math oper res": "math_oper_res",
    "mathematics of operations research": "math_oper_res",
    # Econometrica
    "econometrica": "econometrica",
    # Mathematical Programming
    "math program": "math_program",
    "mathematical programming": "math_program",
    # arXiv
    "arxiv": "arxiv",
    "corr": "arxiv",
    "arxiv preprint": "arxiv",
    "preprint": "arxiv",
    "preprint arxiv": "arxiv",
    "arxiv.org": "arxiv",
}


NAME_ALIASES = {
    "mike": "michael",
    "matt": "matthew",
    "alex": "alexander",
    "ben": "benjamin",
    "chris": "christopher",
    "dan": "daniel",
    "jon": "jonathan",
    "nick": "nicholas",
}


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_math_markup(value: str) -> str:
    if not value:
        return ""
    value = value.replace("\\times", " x ")
    value = value.replace("\\cdot", " x ")
    value = value.replace("×", " x ")
    value = value.replace("\\(", " ")
    value = value.replace("\\)", " ")
    value = value.replace("\\[", " ")
    value = value.replace("\\]", " ")
    value = value.replace("$", " ")
    value = value.replace("`", " ")
    return value


_VENUE_STRIP_RE = re.compile(
    r",?\s*(?:"
    r"pages?\s*[\d\s]+|"                    # pages 123 or pages 1 5
    r"vol(?:ume)?\.?\s*[\d]+|"              # volume 5 / vol. 5
    r"[\d]+\s*\([^)]*\)\s*[\d\s]+|"        # 12(3) 45 67
    r"[\d]+\s+[\d]{2,}[\d\s]*|"            # 18 279284 (vol:pages after normalize)
    r"pp\.?\s*[\d\s]+"                      # pp 1 5
    r")"
)


_VENUE_STOPWORDS = {
    "of", "on", "the", "and", "for", "in", "to", "a", "an", "at", "by",
    "proceedings", "proc", "international", "intl", "annual",
    "conference", "conf", "symposium", "symp", "workshop", "meeting",
    "journal", "trans", "transactions",
}


_VENUE_STOPWORDS = {
    "of", "on", "the", "and", "for", "in", "to", "a", "an", "at", "by",
    "proceedings", "proc", "international", "intl", "annual",
    "conference", "conf", "symposium", "symp", "workshop", "meeting",
    "journal", "trans", "transactions",
}


def _letters_only(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _is_acronym_of(short: str, long: str) -> bool:
    """True if `short` is an acronym of `long`: initials of all words or of the non-stopwords."""
    s = _letters_only(short)
    if not s or len(s) < 2 or len(s) > 10:
        return False
    if " " in short.strip():
        return False

    words = [w for w in _letters_only(long).replace(" ", "|").split("|") if w]
    words = [w.lower() for w in re.split(r"\W+", long or "") if w]
    if not words:
        return False

    init_all = "".join(w[0] for w in words)
    init_content = "".join(w[0] for w in words if w not in _VENUE_STOPWORDS)

    candidates = {init_all, init_content}
    for init in candidates:
        if not init:
            continue
        if s == init:
            return True
        # Prefix match either way, when both sides have at least 3 letters.
        if len(s) >= 3 and len(init) >= 3 and (init.startswith(s) or s.startswith(init)):
            return True
    return False


def _is_word_truncation_of(short: str, long: str) -> bool:
    """True if `short` abbreviates `long` word by word (e.g. "Artif. Intell."), ignoring stopwords."""
    if not short or not long:
        return False
    short_tokens = [t for t in re.split(r"[.\s,]+", short.lower()) if t and re.match(r"^[a-z]+$", t)]
    if len(short_tokens) < 2:
        return False

    long_words = [w.lower() for w in re.split(r"\W+", long or "") if w and re.match(r"^[a-z]+$", w, re.IGNORECASE)]
    long_content = [w for w in long_words if w not in _VENUE_STOPWORDS]

    if len(short_tokens) != len(long_content):
        return False

    for st, lw in zip(short_tokens, long_content):
        if len(st) < 2:
            return False
        if not lw.startswith(st):
            return False
    return True


def venues_equivalent_heuristic(a: str, b: str) -> bool:
    """True if two venue strings match by alias, acronym or word-by-word abbreviation."""
    if not a or not b:
        return False
    na = normalize_venue(a)
    nb = normalize_venue(b)
    if na and nb and na == nb:
        return True
    if _is_acronym_of(a, b) or _is_acronym_of(b, a):
        return True
    if _is_word_truncation_of(a, b) or _is_word_truncation_of(b, a):
        return True
    return False


def normalize_venue(venue: str) -> str:
    cleaned = normalize_text(venue)
    cleaned = re.sub(r"^in\s+", "", cleaned)
    # Strip a trailing year
    cleaned = re.sub(r",?\s*(19|20)\d{2}\w?$", "", cleaned)
    # Strip volume/page info
    cleaned = _VENUE_STRIP_RE.sub("", cleaned).strip()
    # Strip leftover standalone numbers
    cleaned = re.sub(r"\b\d+\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    if cleaned in VENUE_ALIASES:
        return VENUE_ALIASES[cleaned]
    # Otherwise use the longest alias (over 3 characters) contained in the venue.
    sorted_aliases = sorted(VENUE_ALIASES.items(), key=lambda x: -len(x[0]))
    for alias, canonical in sorted_aliases:
        if len(alias) > 3 and alias in cleaned:
            return canonical
    return cleaned


def similarity(a: str, b: str) -> float:
    """Character-level SequenceMatcher ratio; 0.0 if either string is empty."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def normalize_title(title: str) -> str:
    """Normalized title with math markup and all spaces removed."""
    return normalize_text(_normalize_math_markup(title)).replace(" ", "")
