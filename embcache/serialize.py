"""Vector <-> bytes serialization.

Vectors are stored as packed float32 bytes, not JSON. JSON for a 1,536-element
embedding is bloated (each float becomes ~15-20 ASCII chars) and slow to parse.
Packed float32 is a flat 4 bytes per element and round-trips through the stdlib
``array`` module with no third-party dependency.

The float32 precision is intentional: embeddings are direction vectors used for
cosine similarity, where float32 is the de-facto standard and the precision loss
versus float64 is irrelevant.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence

# 'f' is the C float (32-bit) type code for the array module.
_TYPECODE = "f"


def dumps(vector: Sequence[float]) -> bytes:
    """Pack a vector of floats into float32 bytes.

    Args:
        vector: The embedding as a sequence of floats.

    Returns:
        Little-/native-endian packed float32 bytes (4 bytes per element).
    """
    return array(_TYPECODE, vector).tobytes()


def loads(blob: bytes) -> list[float]:
    """Unpack float32 bytes back into a list of floats.

    Args:
        blob: Bytes produced by :func:`dumps`.

    Returns:
        The embedding as a list of Python floats.

    Raises:
        ValueError: If the byte length is not a multiple of 4.
    """
    if len(blob) % 4 != 0:
        raise ValueError(
            f"blob length {len(blob)} is not a multiple of 4; not valid float32 data"
        )
    arr: "array[float]" = array(_TYPECODE)
    arr.frombytes(blob)
    return arr.tolist()
