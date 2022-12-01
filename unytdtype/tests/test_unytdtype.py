import numpy as np
import unyt

from unytdtype import UnytDType, UnytScalar


def test_dtype_creation():
    dtype = UnytDType("m")
    assert str(dtype) == "UnytDType('m')"

    dtype2 = UnytDType(unyt.Unit('m'))
    assert str(dtype2) == "UnytDType('m')"
    assert dtype == dtype2


def test_scalar_creation():
    dtype = UnytDType("m")
    unit = unyt.Unit("m")
    unit_s = "m"

    s_1 = UnytScalar(1, dtype)
    s_2 = UnytScalar(1, unit)
    s_3 = UnytScalar(1, unit_s)

    assert s_1 == s_2 == s_3


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


def test_cast_to_float64():
    meter = UnytScalar(1, unyt.m)
    arr = np.array([meter, meter, meter])
    conv = arr.astype('float64')
    assert str(conv) == "[1. 1. 1.]"
