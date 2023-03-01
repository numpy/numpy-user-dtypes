import numpy as np

from quaddtype import QuadDType, QuadScalar


def test_dtype_creation():
    assert str(QuadDType()) == "This is a quad (128-bit float) dtype."


def test_scalar_creation():
    assert str(QuadScalar(3.1)) == "3.1"


def test_create_with_explicit_dtype():
    assert (
        repr(np.array([3.0, 3.1, 3.2], dtype=QuadDType()))
        == "array([3.0, 3.1, 3.2], dtype=This is a quad (128-bit float) dtype.)"
    )


def test_multiply():
    x = np.array([3, 8.0], dtype=QuadDType())
    assert str(x * x) == "[9.0 64.0]"


def test_bytes():
    """Check that each quad is 16 bytes."""
    x = np.array([3, 8.0, 1.4], dtype=QuadDType())
    assert len(x.tobytes()) == x.size * 16


def test_is_numeric():
    assert QuadDType._is_numeric
