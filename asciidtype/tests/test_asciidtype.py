import numpy as np

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
    assert arr.tobytes() == b"h\x00t\x00i\x00a\x00a\x00"

    dtype = ASCIIDType()
    arr = np.array(["hello", "this", "is", "an", "array"], dtype=dtype)
    assert repr(arr) == ("array(['', '', '', '', ''], dtype=ASCIIDType(0))")
    assert arr.tobytes() == b"\x00\x00\x00\x00\x00"


def test_casting_to_asciidtype():
    arr = np.array(["hello", "this", "is", "an", "array"], dtype=ASCIIDType(5))

    assert repr(arr.astype(ASCIIDType(7))) == (
        "array(['hello', 'this', 'is', 'an', 'array'], dtype=ASCIIDType(7))"
    )

    assert repr(arr.astype(ASCIIDType(5))) == (
        "array(['hello', 'this', 'is', 'an', 'array'], dtype=ASCIIDType(5))"
    )

    assert repr(arr.astype(ASCIIDType(4))) == (
        "array(['hell', 'this', 'is', 'an', 'arra'], dtype=ASCIIDType(4))"
    )

    assert repr(arr.astype(ASCIIDType(1))) == (
        "array(['h', 't', 'i', 'a', 'a'], dtype=ASCIIDType(1))"
    )

    assert repr(arr.astype(ASCIIDType())) == (
        "array(['', '', '', '', ''], dtype=ASCIIDType(0))"
    )
