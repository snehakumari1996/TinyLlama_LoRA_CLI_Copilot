"""Small text utilities shared across the package."""
from __future__ import annotations

import html
import re

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(txt: str) -> str:
    """Collapse whitespace and unescape HTML entities."""
    return _WS_RE.sub(" ", html.unescape(txt)).strip()


def normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used as a stable key for exact + near-exact duplicate detection across
    the data pipeline (`collect_data` and `split_data` both rely on this).
    """
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", text.lower())).strip()
