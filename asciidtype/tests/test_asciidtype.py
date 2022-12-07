import numpy as np

from asciidtype import ASCIIDType, ASCIIScalar


def test_dtype_creation():
    dtype = ASCIIDType(4)
    assert str(dtype) == "ASCIIDType(4)"


def test_scalar_creation():
    dtype = ASCIIDType(7)
    ASCIIScalar('string', dtype)


def test_creation_with_explicit_dtype():
    dtype = ASCIIDType(7)
    arr = np.array(["hello", "this", "is", "an", "array"], dtype=dtype)
    assert repr(arr) == (
        "array(['hello', 'this', 'is', 'an', 'array'], dtype=ASCIIDType(7))")
