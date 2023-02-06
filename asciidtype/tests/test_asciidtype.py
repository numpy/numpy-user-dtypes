import os
import pickle
import re
import tempfile

import numpy as np
import pytest

from asciidtype import ASCIIDType, ASCIIScalar


def test_dtype_creation():
    dtype = ASCIIDType(4)
    assert str(dtype) == "ASCIIDType(4)"


def test_scalar_creation():
    dtype = ASCIIDType(7)
    ASCIIScalar("string", dtype)


def test_creation_with_explicit_dtype():
    dtype = ASCIIDType(7)
    arr = np.array(["hello", "this", "is", "an", "array"], dtype=dtype)
    assert repr(arr) == (
        "array(['hello', 'this', 'is', 'an', 'array'], dtype=ASCIIDType(7))"
    )


def test_creation_from_scalar():
    data = [
        ASCIIScalar("hello", ASCIIDType(6)),
        ASCIIScalar("array", ASCIIDType(7)),
    ]
    arr = np.array(data)
    assert repr(arr) == ("array(['hello', 'array'], dtype=ASCIIDType(7))")


def test_creation_truncation():
    inp = ["hello", "this", "is", "an", "array"]

    dtype = ASCIIDType(5)
    arr = np.array(inp, dtype=dtype)
    assert repr(arr) == (
        "array(['hello', 'this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
    )

    dtype = ASCIIDType(4)
    arr = np.array(inp, dtype=dtype)
    assert repr(arr) == (
        "array(['hell', 'this', 'is', 'an', 'arra'], dtype=ASCIIDType(4))"
    )

    dtype = ASCIIDType(1)
    arr = np.array(inp, dtype=dtype)
    assert repr(arr) == (
        "array(['h', 't', 'i', 'a', 'a'], dtype=ASCIIDType(1))"
    )
    assert arr.tobytes() == b"htiaa"

    dtype = ASCIIDType()
    arr = np.array(["hello", "this", "is", "an", "array"], dtype=dtype)
    assert repr(arr) == ("array(['', '', '', '', ''], dtype=ASCIIDType(0))")
    assert arr.tobytes() == b""


def test_casting_to_asciidtype():
    for dtype in (None, ASCIIDType(5)):
        arr = np.array(["this", "is", "an", "array"], dtype=dtype)

        assert repr(arr.astype(ASCIIDType(7))) == (
            "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(7))"
        )

        assert repr(arr.astype(ASCIIDType(5))) == (
            "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
        )

        assert repr(arr.astype(ASCIIDType(4))) == (
            "array(['this', 'is', 'an', 'arra'], dtype=ASCIIDType(4))"
        )

        assert repr(arr.astype(ASCIIDType(1))) == (
            "array(['t', 'i', 'a', 'a'], dtype=ASCIIDType(1))"
        )

        # assert repr(arr.astype(ASCIIDType())) == (
        #    "array(['', '', '', '', ''], dtype=ASCIIDType(0))"
        # )


def test_casting_safety():
    arr = np.array(["this", "is", "an", "array"])
    assert repr(arr.astype(ASCIIDType(6), casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(6))"
    )
    assert repr(arr.astype(ASCIIDType(5), casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
    )
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Cannot cast array data from dtype('<U5') to ASCIIDType(4) "
            "according to the rule 'safe'"
        ),
    ):
        assert repr(arr.astype(ASCIIDType(4), casting="safe")) == (
            "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
        )
    assert repr(arr.astype(ASCIIDType(4), casting="unsafe")) == (
        "array(['this', 'is', 'an', 'arra'], dtype=ASCIIDType(4))"
    )

    arr = np.array(["this", "is", "an", "array"], dtype=ASCIIDType(5))
    assert repr(arr.astype(ASCIIDType(6), casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(6))"
    )
    assert repr(arr.astype(ASCIIDType(5), casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
    )
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Cannot cast array data from ASCIIDType(5) to ASCIIDType(4) "
            "according to the rule 'safe'"
        ),
    ):
        assert repr(arr.astype(ASCIIDType(4), casting="safe")) == (
            "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
        )
    assert repr(arr.astype(ASCIIDType(4), casting="unsafe")) == (
        "array(['this', 'is', 'an', 'arra'], dtype=ASCIIDType(4))"
    )

    arr = np.array(["this", "is", "an", "array"], dtype=ASCIIDType(5))
    assert repr(arr.astype("U6", casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype='<U6')"
    )
    assert repr(arr.astype("U5", casting="safe")) == (
        "array(['this', 'is', 'an', 'array'], dtype='<U5')"
    )
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Cannot cast array data from ASCIIDType(5) to dtype('<U4') "
            "according to the rule 'safe'"
        ),
    ):
        assert repr(arr.astype("U4", casting="safe")) == (
            "array(['this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
        )
    assert repr(arr.astype("U4", casting="unsafe")) == (
        "array(['this', 'is', 'an', 'arra'], dtype='<U4')"
    )


def test_unicode_to_ascii_to_unicode():
    arr = np.array(["hello", "this", "is", "an", "array"])
    ascii_arr = arr.astype(ASCIIDType(5))
    for dtype in ["U5", np.unicode_, np.str_]:
        round_trip_arr = ascii_arr.astype(dtype)
        np.testing.assert_array_equal(arr, round_trip_arr)


def test_creation_fails_with_non_ascii_characters():
    inps = [
        ["😀", "¡", "©", "ÿ"],
        ["😀", "hello", "some", "ascii"],
        ["hello", "some", "ascii", "😀"],
    ]
    for inp in inps:
        with pytest.raises(
            TypeError,
            match="Can only store ASCII text in a ASCIIDType array.",
        ):
            np.array(inp, dtype=ASCIIDType(5))


def test_casting_fails_with_non_ascii_characters():
    inps = [
        ["😀", "¡", "©", "ÿ"],
        ["😀", "hello", "some", "ascii"],
        ["hello", "some", "ascii", "😀"],
    ]
    for inp in inps:
        arr = np.array(inp)
        with pytest.raises(
            TypeError,
            match="Can only store ASCII text in a ASCIIDType array.",
        ):
            arr.astype(ASCIIDType(5))


@pytest.mark.parametrize(
    ("input1", "input2", "expected"),
    [
        (["hello "], ["world"], ["hello world"]),
        (["a", "b", "c"], ["aa", "bbbb", "cc"], ["aaa", "bbbbb", "ccc"]),
        (["aa", "bbbb", "cc"], ["a", "b", "c"], ["aaa", "bbbbb", "ccc"]),
    ],
)
def test_addition(input1, input2, expected):
    maxlen1 = max([len(i) for i in input1])
    maxlen2 = max([len(i) for i in input2])
    maxlene = max([len(e) for e in expected])
    input1 = np.array(input1, dtype=ASCIIDType(maxlen1))
    input2 = np.array(input2, dtype=ASCIIDType(maxlen2))
    expected = np.array(expected, dtype=ASCIIDType(maxlene))
    np.testing.assert_array_equal(input1 + input2, expected)


@pytest.mark.parametrize(
    ("input1", "input2", "expected"),
    [
        (["hello", "world"], ["hello", "world"], [True, True]),
        (["hello ", "world"], ["hello", "world"], [False, True]),
        (["hello", "world"], ["h", "w"], [False, False]),
    ],
)
def test_equality(input1, input2, expected):
    maxlen1 = max([len(i) for i in input1])
    maxlen2 = max([len(i) for i in input2])
    input1 = np.array(input1, dtype=ASCIIDType(maxlen1))
    input2 = np.array(input2, dtype=ASCIIDType(maxlen2))
    expected = np.array(expected, dtype=np.bool_)
    np.testing.assert_array_equal(input1 == input2, expected)
    np.testing.assert_array_equal(input2 == input1, expected)


def test_insert_scalar_directly():
    dtype = ASCIIDType(5)
    arr = np.array(["some", "array"], dtype=dtype)
    val = arr[0]
    arr[1] = val
    np.testing.assert_array_equal(arr, np.array(["some", "some"], dtype=dtype))


def test_pickle():
    dtype = ASCIIDType(6)
    arr = np.array(["this", "is", "an", "array"], dtype=dtype)
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        pickle.dump([arr, dtype], f)

    with open(f.name, "rb") as f:
        res = pickle.load(f)

    np.testing.assert_array_equal(arr, res[0])
    assert res[1] == dtype

    os.remove(f.name)
