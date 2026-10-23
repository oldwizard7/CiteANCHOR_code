"""Logging setup: console stream, log file grouped per citation, credential redaction."""
from __future__ import annotations

import datetime
import logging
import os
import re
import threading
from pathlib import Path

from .paths import TZ

# Per-thread label of the citation being processed.
_local = threading.local()


def set_item(label: str | None) -> None:
    """Tag this thread's log lines. Passing None ends the item and flushes it."""
    previous = getattr(_local, "item", None)
    _local.item = label
    if label is None and previous is not None:
        for h in _BUFFERS:
            h.flush_item(previous)


def _item() -> str:
    return getattr(_local, "item", "") or "-"


class _ItemFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.item = _item()
        return True


_SECRET_ENV = ("OPENAI_API_KEY", "SERPAPI_API_KEY", "SERPAPI_API_KEY2", "SERPAPI_KEY",
               "GEMINI_API_KEY", "HF_TOKEN", "S2_API_KEY", "GITHUB_TOKEN",
               "OPENALEX_API_KEY", "GOOGLE_BOOKS_API_KEY")
# `api_key=` is SerpApi's and OpenAlex's shape, `key=` Google's.
_QUERY_KEY = re.compile(r"((?:api_)?key=)[0-9A-Za-z._\-]{16,}")


def redact(text) -> str:
    """Replace URL query keys and configured secret values with <redacted>."""
    text = _QUERY_KEY.sub(r"\1<redacted>", str(text))
    for name in _SECRET_ENV:
        value = os.getenv(name) or ""
        if len(value) >= 16:
            text = text.replace(value, "<redacted>")
    return text


class _RedactFilter(logging.Filter):
    """Redact credentials in each record's message."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


# Buffering handlers, so `set_item(None)` can tell them an item is complete.
_BUFFERS: list = []


class _ByItemHandler(logging.Handler):
    """Buffer each citation's records and write them as one block when it ends."""

    def __init__(self, target: logging.Handler):
        super().__init__(level=target.level)
        self.target = target
        self._by_item: dict[str, list[logging.LogRecord]] = {}
        self._lock2 = threading.Lock()
        _BUFFERS.append(self)

    def emit(self, record: logging.LogRecord) -> None:
        item = getattr(record, "item", "-")
        if item == "-":
            self.target.emit(record)
            return
        with self._lock2:
            self._by_item.setdefault(item, []).append(record)

    def flush_item(self, item: str) -> None:
        with self._lock2:
            records = self._by_item.pop(item, [])
        for r in records:
            self.target.emit(r)
        self.target.flush()

    def close(self) -> None:
        with self._lock2:
            leftovers = list(self._by_item.items())
            self._by_item.clear()
        for _, records in leftovers:
            for r in records:
                self.target.emit(r)
        self.target.flush()
        self.target.close()
        super().close()


class _TZFormatter(logging.Formatter):
    """Formatter with timestamps in `paths.TZ`."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, TZ)
        return dt.strftime(datefmt or "%H:%M:%S")


def setup(log_path: str | Path | None = None, *, console_level: int = logging.INFO,
          file_level: int = logging.DEBUG) -> Path | None:
    """Attach handlers to the `citeanchor` logger. Returns the log path, if any."""
    root = logging.getLogger("citeanchor")
    root.setLevel(min(console_level, file_level))
    root.propagate = False
    for h in list(root.handlers):        # idempotent: re-running must not double up
        root.removeHandler(h)
        h.close()
    _BUFFERS.clear()

    # Filters go on handlers: logger filters do not see records from child loggers.
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(_TZFormatter("%(message)s"))
    console.addFilter(_ItemFilter())
    console.addFilter(_RedactFilter())
    root.addHandler(console)

    if log_path is None:
        return None
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(_TZFormatter(
        "%(asctime)s %(levelname).1s [%(item)s] %(name)s: %(message)s"))
    fh.addFilter(_ItemFilter())
    grouped = _ByItemHandler(fh)
    grouped.addFilter(_ItemFilter())
    grouped.addFilter(_RedactFilter())
    root.addHandler(grouped)
    return path


def get(name: str) -> logging.Logger:
    """Loggers are named `citeanchor.<module>` so one setup() call covers them all."""
    return logging.getLogger(f"citeanchor.{name}")
