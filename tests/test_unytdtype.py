import numpy as np
import pytest
import unyt
from unytdtype import UnytDType, UnytScalar


def test_dtype_creation():
    dtype = UnytDType("m")
    assert str(dtype) == "UnytDType('m')"


def test_creation_from_zeros():
    dtype = UnytDType("m")
    arr = np.zeros(3, dtype=dtype)
    assert str(arr) == "[0.0 m 0.0 m 0.0 m]"


def test_creation_from_list():
    dtype = UnytDType("m")
    arr = np.array([0, 0, 0], dtype=dtype)
    assert str(arr) == "[0.0 m 0.0 m 0.0 m]"


def test_creation_from_scalar():
    meter = UnytScalar(1, unyt.m)
    arr = np.array([meter, meter, meter])
    assert str(arr) == "[1.0 m 1.0 m 1.0 m]"


def test_multiplication():
    meter = UnytScalar(1, unyt.m)
    arr = np.array([meter, meter, meter])
    arr2 = np.array([2 * meter, 2 * meter, 2 * meter])
    res = arr * arr2
    assert str(res) == "[2.0 m**2 2.0 m**2 2.0 m**2]"


def test_cast_to_different_unit():
    meter = UnytScalar(1, unyt.m)
    arr = np.array([meter, meter, meter])
    conv = arr.astype(UnytDType("cm"))
    assert str(conv) == "[100.0 cm 100.0 cm 100.0 cm]"


def test_insert_with_different_unit():
    meter = UnytScalar(1, unyt.m)
    cm = UnytScalar(1, unyt.cm)
    arr = np.array([meter, meter, meter])
    arr[0] = cm
    assert str(arr) == "[0.01 m 1.0 m 1.0 m]"
