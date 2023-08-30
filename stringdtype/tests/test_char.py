import numpy as np
import pytest
from numpy.testing import assert_array_equal

from stringdtype import StringDType

TEST_DATA = ["hello", "Ae¢☃€ 😊", "entry\nwith\nnewlines", "entry\twith\ttabs"]


@pytest.fixture
def string_array():
    return np.array(TEST_DATA, dtype=StringDType())


@pytest.fixture
def unicode_array():
    return np.array(TEST_DATA, dtype=np.str_)


UNARY_FUNCTIONS = [
    "str_len",
    "capitalize",
    "expandtabs",
    "isalnum",
    "isalpha",
    "isdigit",
    "islower",
    "isspace",
    "istitle",
    "isupper",
    "lower",
    "splitlines",
    "swapcase",
    "title",
    "upper",
    "isnumeric",
    "isdecimal",
]


@pytest.mark.parametrize("function_name", UNARY_FUNCTIONS)
def test_unary(string_array, unicode_array, function_name):
    func = getattr(np.char, function_name)
    sres = func(string_array)
    ures = func(unicode_array)
    if sres.dtype == StringDType():
        ures = ures.astype(StringDType())
    assert_array_equal(sres, ures)


# None means that the argument is a string array
BINARY_FUNCTIONS = [
    ("add", (None, None)),
    ("multiply", (None, 2)),
    ("mod", ("format: %s", None)),
    ("center", (None, 25)),
    ("count", (None, "A")),
    ("encode", (None, "UTF-8")),
    ("endswith", (None, "lo")),
    ("find", (None, "A")),
    ("index", (None, "e")),
    ("join", ("-", None)),
    ("ljust", (None, 12)),
    ("partition", (None, "A")),
    ("replace", (None, "A", "B")),
    ("rfind", (None, "A")),
    ("rindex", (None, "e")),
    ("rjust", (None, 12)),
    ("rpartition", (None, "A")),
    ("split", (None, "A")),
    ("startswith", (None, "A")),
    ("zfill", (None, 12)),
]


@pytest.mark.parametrize("function_name, args", BINARY_FUNCTIONS)
def test_binary(string_array, unicode_array, function_name, args):
    func = getattr(np.char, function_name)
    if args == (None, None):
        sres = func(string_array, string_array)
        ures = func(unicode_array, unicode_array)
    elif args[0] is None:
        sres = func(string_array, *args[1:])
        ures = func(string_array, *args[1:])
    elif args[1] is None:
        sres = func(args[0], string_array)
        ures = func(args[0], string_array)
    else:
        # shouldn't ever happen
        raise RuntimeError
    if sres.dtype == StringDType():
        ures = ures.astype(StringDType())
    assert_array_equal(sres, ures)


def test_strip(string_array, unicode_array):
    rjs = np.char.rjust(string_array, 25)
    rju = np.char.rjust(unicode_array, 25)

    ljs = np.char.ljust(string_array, 25)
    lju = np.char.ljust(unicode_array, 25)

    assert_array_equal(
        np.char.lstrip(rjs),
        np.char.lstrip(rju).astype(StringDType()),
    )

    assert_array_equal(
        np.char.rstrip(ljs),
        np.char.rstrip(lju).astype(StringDType()),
    )

    assert_array_equal(
        np.char.strip(ljs),
        np.char.strip(lju).astype(StringDType()),
    )

    assert_array_equal(
        np.char.strip(rjs),
        np.char.strip(rju).astype(StringDType()),
    )
