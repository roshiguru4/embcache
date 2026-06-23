"""Text normalization for cache keys.

Normalization makes trivially-different inputs (extra spaces, leading/trailing
whitespace) collapse to the same cache key, so they hit instead of paying for a
re-embed. Whitespace handling is on by default; lowercasing is opt-in because it
can change meaning for some embedding use cases.
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str, *, lowercase: bool = False) -> str:
    """Return a normalized copy of ``text``.

    Always strips leading/trailing whitespace and collapses internal runs of
    whitespace to a single space. If ``lowercase`` is true, also lowercases.

    Args:
        text: The raw input text.
        lowercase: If true, lowercase the text after whitespace normalization.

    Returns:
        The normalized text.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        collapsed = collapsed.lower()
    return collapsed
