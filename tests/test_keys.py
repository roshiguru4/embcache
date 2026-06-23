from embcache.keys import make_key


def test_same_text_same_model_same_key():
    assert make_key("the cat sat", "m1") == make_key("the cat sat", "m1")


def test_normalization_applied_to_keys():
    assert make_key("  the   cat sat ", "m1") == make_key("the cat sat", "m1")


def test_different_model_different_key():
    # Model-aware keys: switching models must never collide.
    assert make_key("the cat sat", "m1") != make_key("the cat sat", "m2")


def test_different_text_different_key():
    assert make_key("cat", "m1") != make_key("dog", "m1")


def test_key_is_sha256_hex():
    key = make_key("anything", "m1")
    assert len(key) == 64
    int(key, 16)  # raises if not valid hex


def test_lowercase_changes_key():
    assert make_key("The Cat", "m1") != make_key("The Cat", "m1", lowercase=True)
    assert make_key("The Cat", "m1", lowercase=True) == make_key("the cat", "m1")
