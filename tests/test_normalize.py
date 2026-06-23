from embcache.normalize import normalize


def test_strips_leading_and_trailing_whitespace():
    assert normalize("  hello  ") == "hello"


def test_collapses_internal_whitespace():
    assert normalize("the   cat\tsat\non  mat") == "the cat sat on mat"


def test_trivial_differences_collapse_to_same():
    assert normalize(" the cat sat ") == normalize("the  cat   sat")


def test_lowercase_off_by_default():
    assert normalize("The Cat") == "The Cat"


def test_lowercase_opt_in():
    assert normalize("The Cat", lowercase=True) == "the cat"


def test_empty_string():
    assert normalize("   ") == ""
