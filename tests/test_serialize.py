import math

import pytest

from embcache.serialize import dumps, loads


def test_round_trip_preserves_values():
    vec = [0.0, 1.0, -1.0, 0.5, -0.25]
    out = loads(dumps(vec))
    assert out == pytest.approx(vec)


def test_round_trip_large_vector():
    vec = [i * 0.001 for i in range(1536)]
    out = loads(dumps(vec))
    assert len(out) == 1536
    assert out == pytest.approx(vec, abs=1e-6)


def test_empty_vector():
    assert loads(dumps([])) == []


def test_float32_is_four_bytes_per_element():
    assert len(dumps([1.0, 2.0, 3.0])) == 12


def test_packed_bytes_smaller_than_json():
    import json

    vec = [0.123456789] * 1536
    packed = dumps(vec)
    as_json = json.dumps(vec).encode("utf-8")
    assert len(packed) < len(as_json)


def test_invalid_blob_length_raises():
    with pytest.raises(ValueError):
        loads(b"\x00\x00\x00")  # 3 bytes, not a multiple of 4


def test_special_values_round_trip():
    out = loads(dumps([float("inf"), float("-inf")]))
    assert out[0] == float("inf")
    assert out[1] == float("-inf")
    nan_out = loads(dumps([float("nan")]))
    assert math.isnan(nan_out[0])
