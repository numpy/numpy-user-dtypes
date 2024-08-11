import pytest
import sys
import numpy as np
import operator

from quaddtype import QuadPrecDType, QuadPrecision


def test_create_scalar_simple():
    assert isinstance(QuadPrecision("12.0"), QuadPrecision)
    assert isinstance(QuadPrecision(1.63), QuadPrecision)
    assert isinstance(QuadPrecision(1), QuadPrecision)


def test_basic_equality():
    assert QuadPrecision("12") == QuadPrecision(
        "12.0") == QuadPrecision("12.00")


@pytest.mark.parametrize("val", ["123532.543", "12893283.5"])
def test_scalar_repr(val):
    expected = f"QuadPrecision('{str(QuadPrecision(val))}')"
    assert repr(QuadPrecision(val)) == expected


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow"])
@pytest.mark.parametrize("other", ["3.0", "12.5", "100.0"])
def test_binary_ops(op, other):
    op_func = getattr(operator, op)
    quad_a = QuadPrecision("12.5")
    quad_b = QuadPrecision(other)
    float_a = 12.5
    float_b = float(other)

    quad_result = op_func(quad_a, quad_b)
    float_result = op_func(float_a, float_b)

    assert np.abs(np.float64(quad_result) - float_result) < 1e-10


@pytest.mark.parametrize("op", ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("other", ["3.0", "12.5", "100.0"])
def test_comparisons(op, other):
    op_func = getattr(operator, op)
    quad_a = QuadPrecision("12.5")
    quad_b = QuadPrecision(other)
    float_a = 12.5
    float_b = float(other)

    assert op_func(quad_a, quad_b) == op_func(float_a, float_b)


@pytest.mark.parametrize("op, val, expected", [
    ("neg", "3.0", "-3.0"),
    ("neg", "-3.0", "3.0"),
    ("pos", "3.0", "3.0"),
    ("pos", "-3.0", "-3.0"),
    ("abs", "3.0", "3.0"),
    ("abs", "-3.0", "3.0"),
    ("neg", "12.5", "-12.5"),
    ("pos", "100.0", "100.0"),
    ("abs", "-25.5", "25.5"),
])
def test_unary_ops(op, val, expected):
    quad_val = QuadPrecision(val)
    expected_val = QuadPrecision(expected)

    if op == "neg":
        result = -quad_val
    elif op == "pos":
        result = +quad_val
    elif op == "abs":
        result = abs(quad_val)
    else:
        raise ValueError(f"Unsupported operation: {op}")

    assert result == expected_val, f"{op}({val}) should be {expected}, but got {result}"


def test_nan_and_inf():
    assert (QuadPrecision("nan") != QuadPrecision("nan")) == (
        QuadPrecision("nan") == QuadPrecision("nan"))
    assert QuadPrecision("inf") > QuadPrecision("1e1000")
    assert QuadPrecision("-inf") < QuadPrecision("-1e1000")


def test_dtype_creation():
    dtype = QuadPrecDType()
    assert isinstance(dtype, np.dtype)
    assert dtype.name == 'QuadPrecDType128'


def test_array_creation():
    arr = np.array([1, 2, 3], dtype=QuadPrecDType())
    assert arr.dtype.name == 'QuadPrecDType128'
    assert all(isinstance(x, QuadPrecision) for x in arr)


def test_array_operations():
    arr1 = np.array(
        [QuadPrecision("1.5"), QuadPrecision("2.5"), QuadPrecision("3.5")])
    arr2 = np.array(
        [QuadPrecision("0.5"), QuadPrecision("1.0"), QuadPrecision("1.5")])

    result = arr1 + arr2
    expected = np.array(
        [QuadPrecision("2.0"), QuadPrecision("3.5"), QuadPrecision("5.0")])
    assert np.all(result == expected)
