"""Cache key construction.

Keys are model-aware: the model name is folded into the hash so vectors produced
by different models never collide. Switching models therefore invalidates the
cache correctly with zero extra bookkeeping.
"""

from __future__ import annotations

import hashlib

from .normalize import normalize


def make_key(text: str, model: str, *, lowercase: bool = False) -> str:
    """Build a cache key for ``text`` under ``model``.

    The text is normalized first, then hashed together with the model name as
    ``sha256(normalized_text + ":" + model)``.

    Args:
        text: The raw input text (will be normalized).
        model: The embedding model name. Folded into the key so different
            models never share cached vectors.
        lowercase: Passed through to normalization.

    Returns:
        A hex digest string usable as a backend key.
    """
    normalized = normalize(text, lowercase=lowercase)
    payload = f"{normalized}:{model}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
