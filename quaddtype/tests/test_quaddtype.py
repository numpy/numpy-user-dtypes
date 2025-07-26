import pytest
import sys
import numpy as np
import operator

import numpy_quaddtype
from numpy_quaddtype import QuadPrecDType, QuadPrecision


def test_create_scalar_simple():
    assert isinstance(QuadPrecision("12.0"), QuadPrecision)
    assert isinstance(QuadPrecision(1.63), QuadPrecision)
    assert isinstance(QuadPrecision(1), QuadPrecision)

@pytest.mark.parametrize("name,expected", [("pi", np.pi), ("e", np.e), ("log2e", np.log2(np.e)), ("log10e", np.log10(np.e)), ("ln2", np.log(2.0)), ("ln10", np.log(10.0))])
def test_math_constant(name, expected):
    assert isinstance(getattr(numpy_quaddtype, name), QuadPrecision)

    assert np.float64(getattr(numpy_quaddtype, name)) == expected


@pytest.mark.parametrize("name", ["max_value", "epsilon", "smallest_normal", "smallest_subnormal"])
def test_finfo_constant(name):
    assert isinstance(getattr(numpy_quaddtype, name), QuadPrecision)


def test_basic_equality():
    assert QuadPrecision("12") == QuadPrecision(
        "12.0") == QuadPrecision("12.00")


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow", "copysign"])
@pytest.mark.parametrize("other", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_binary_ops(op, other):
    if op == "truediv" and float(other) == 0:
        pytest.xfail("float division by zero")

    op_func = getattr(operator, op, None) or getattr(np, op)
    quad_a = QuadPrecision("12.5")
    quad_b = QuadPrecision(other)
    float_a = 12.5
    float_b = float(other)

    quad_result = op_func(quad_a, quad_b)
    float_result = op_func(float_a, float_b)

    np.testing.assert_allclose(np.float64(quad_result), float_result, atol=1e-10, rtol=0, equal_nan=True)


@pytest.mark.parametrize("op", ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_comparisons(op, a, b):
    op_func = getattr(operator, op)
    quad_a = QuadPrecision(a)
    quad_b = QuadPrecision(b)
    float_a = float(a)
    float_b = float(b)

    assert op_func(quad_a, quad_b) == op_func(float_a, float_b)


@pytest.mark.parametrize("op", ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_comparisons(op, a, b):
    op_func = getattr(operator, op)
    quad_a = np.array(QuadPrecision(a))
    quad_b = np.array(QuadPrecision(b))
    float_a = np.array(float(a))
    float_b = np.array(float(b))

    assert np.array_equal(op_func(quad_a, quad_b), op_func(float_a, float_b))


@pytest.mark.parametrize("op", ["minimum", "maximum", "fmin", "fmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_minmax(op, a, b):
    op_func = getattr(np, op)
    quad_a = np.array([QuadPrecision(a)])
    quad_b = np.array([QuadPrecision(b)])
    float_a = np.array([float(a)])
    float_b = np.array([float(b)])

    quad_res = op_func(quad_a, quad_b)
    float_res = op_func(float_a, float_b)

    np.testing.assert_array_equal(quad_res.astype(float), float_res)


@pytest.mark.parametrize("op", ["amin", "amax", "nanmin", "nanmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_aminmax(op, a, b):
    op_func = getattr(np, op)
    quad_ab = np.array([QuadPrecision(a), QuadPrecision(b)])
    float_ab = np.array([float(a), float(b)])

    quad_res = op_func(quad_ab)
    float_res = op_func(float_ab)

    np.testing.assert_array_equal(np.array(quad_res).astype(float), float_res)


@pytest.mark.parametrize("op", ["negative", "positive", "absolute", "sign", "signbit", "isfinite", "isinf", "isnan"])
@pytest.mark.parametrize("val", ["3.0", "-3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_unary_ops(op, val):
    op_func = dict(negative=operator.neg, positive=operator.pos, absolute=operator.abs).get(op, None)
    nop_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    for of in [op_func, nop_func]:
        if of is None:
            continue

        quad_result = of(quad_val)
        float_result = of(float_val)

        np.testing.assert_array_equal(np.array(quad_result).astype(float), float_result)
        assert np.signbit(float_result) == np.signbit(quad_result)


def test_inf():
    assert QuadPrecision("inf") > QuadPrecision("1e1000")
    assert QuadPrecision("-inf") < QuadPrecision("-1e1000")


def test_dtype_creation():
    dtype = QuadPrecDType()
    assert isinstance(dtype, np.dtype)
    assert dtype.name == "QuadPrecDType128"


def test_array_creation():
    arr = np.array([1, 2, 3], dtype=QuadPrecDType())
    assert arr.dtype.name == "QuadPrecDType128"
    assert all(isinstance(x, QuadPrecision) for x in arr)


def test_array_operations():
    arr1 = np.array(
        [QuadPrecision("1.5"), QuadPrecision("2.5"), QuadPrecision("3.5")])
    arr2 = np.array(
        [QuadPrecision("0.5"), QuadPrecision("1.0"), QuadPrecision("1.5")])

    result = arr1 + arr2
    expected = np.array(
        [QuadPrecision("2.0"), QuadPrecision("3.5"), QuadPrecision("5.0")])
    assert all(x == y for x, y in zip(result, expected))
