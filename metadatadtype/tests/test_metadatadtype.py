import numpy as np

from metadatadtype import MetadataDType, MetadataScalar


def test_dtype_creation():
    dtype = MetadataDType("some metadata")
    assert str(dtype) == "MetadataDType('some metadata')"


def test_creation_from_zeros():
    dtype = MetadataDType("test")
    arr = np.zeros(3, dtype=dtype)
    assert str(arr) == "[0.0 test 0.0 test 0.0 test]"


def test_creation_from_list():
    dtype = MetadataDType("test")
    arr = np.array([0, 0, 0], dtype=dtype)
    assert str(arr) == "[0.0 test 0.0 test 0.0 test]"


def test_creation_from_scalar():
    dtype = MetadataDType("test")
    scalar = MetadataScalar(1, dtype)
    arr = np.array([scalar, scalar, scalar])
    assert str(arr) == "[1.0 test 1.0 test 1.0 test]"


def test_multiplication():
    dtype = MetadataDType("test")
    scalar = MetadataScalar(1, dtype)
    arr = np.array([scalar, scalar, scalar])
    scalar = MetadataScalar(2, dtype)
    arr2 = np.array([scalar, scalar, scalar])
    res = arr * arr2
    assert str(res) == "[2.0 test 2.0 test 2.0 test]"


def test_isnan():
    dtype = MetadataDType("test")
    num_scalar = MetadataScalar(1, dtype)
    nan_scalar = MetadataScalar(np.nan, dtype)
    arr = np.array([num_scalar, nan_scalar, nan_scalar])
    np.testing.assert_array_equal(np.isnan(arr), np.array([False, True, True]))


def test_cast_to_different_metadata():
    dtype = MetadataDType("test")
    scalar = MetadataScalar(1, dtype)
    arr = np.array([scalar, scalar, scalar])
    dtype2 = MetadataDType("test2")
    conv = arr.astype(dtype2)
    assert str(conv) == "[1.0 test2 1.0 test2 1.0 test2]"


def test_cast_to_float64():
    dtype = MetadataDType("test")
    scalar = MetadataScalar(1, dtype)
    arr = np.array([scalar, scalar, scalar])
    conv = arr.astype("float64")
    assert str(conv) == "[1. 1. 1.]"


def test_is_numeric():
    assert MetadataDType._is_numeric
