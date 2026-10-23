"""Timestamped output paths."""
from __future__ import annotations

import datetime
from pathlib import Path

TZ = datetime.timezone.utc


def now_stamp() -> str:
    """Current UTC time as YYYYMMDDTHHMMSSUTC."""
    return datetime.datetime.now(TZ).strftime("%Y%m%dT%H%M%S%Z")


def stamped(path: str | Path, stamp: str | None = None) -> Path:
    """Insert a timestamp before the suffix: run.json -> run_<stamp>.json."""
    p = Path(path)
    return p.with_name(f"{p.stem}_{stamp or now_stamp()}{p.suffix}")


def resolve_output(path: str | Path, *, timestamp: bool = True,
                   stamp: str | None = None) -> Path:
    out = stamped(path, stamp) if timestamp else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
