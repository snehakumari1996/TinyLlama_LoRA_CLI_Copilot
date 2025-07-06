"""Misc. helper functions (currently only text cleaning)."""
import re, html

def clean_text(txt: str) -> str:
    """Collapse whitespace & unescape HTML."""
    return re.sub(r"\s+", " ", html.unescape(txt)).strip()