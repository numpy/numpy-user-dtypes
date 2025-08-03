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


@pytest.mark.parametrize("op", ["negative", "positive", "absolute", "sign", "signbit", "isfinite", "isinf", "isnan", "sqrt", "square", "reciprocal"])
@pytest.mark.parametrize("val", ["3.0", "-3.0", "12.5", "100.0", "1e100", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
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

        if op in ["negative", "positive", "absolute", "sign"]:
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


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
@pytest.mark.parametrize("op", [np.mod, np.remainder])
@pytest.mark.parametrize("a,b", [
    # Basic cases - positive/negative combinations
    (7.0, 3.0), (-7.0, 3.0), (7.0, -3.0), (-7.0, -3.0),

    # Zero dividend cases
    (0.0, 3.0), (-0.0, 3.0), (0.0, -3.0), (-0.0, -3.0),

    # Cases that result in zero (sign testing)
    (6.0, 3.0), (-6.0, 3.0), (6.0, -3.0), (-6.0, -3.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),

    # Fractional cases
    (7.5, 2.5), (-7.5, 2.5), (7.5, -2.5), (-7.5, -2.5),
    (0.75, 0.25), (-0.1, 0.3), (0.9, -1.0), (-1.1, -1.0),

    # Large/small numbers
    (1e10, 1e5), (-1e10, 1e5), (1e-10, 1e-5), (-1e-10, 1e-5),

    # Finite % infinity cases
    (5.0, float('inf')), (-5.0, float('inf')),
    (5.0, float('-inf')), (-5.0, float('-inf')),
    (0.0, float('inf')), (-0.0, float('-inf')),

    # NaN cases (should return NaN)
    (float('nan'), 3.0), (3.0, float('nan')), (float('nan'), float('nan')),

    # Division by zero cases (should return NaN)
    (5.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (-0.0, 0.0),

    # Infinity dividend cases (should return NaN)
    (float('inf'), 3.0), (float('-inf'), 3.0),
    (float('inf'), float('inf')), (float('-inf'), float('-inf')),
])
def test_mod(a, b, backend, op):
    """Comprehensive test for mod operation against NumPy behavior"""
    if backend == "sleef":
        quad_a = QuadPrecision(str(a))
        quad_b = QuadPrecision(str(b))
    elif backend == "longdouble":
        quad_a = QuadPrecision(a, backend='longdouble')
        quad_b = QuadPrecision(b, backend='longdouble')
    float_a = np.float64(a)
    float_b = np.float64(b)

    quad_result = op(quad_a, quad_b)
    numpy_result = op(float_a, float_b)

    # Handle NaN cases
    if np.isnan(numpy_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {a} % {b}, got {float(quad_result)}"
        return

    if np.isinf(numpy_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {a} % {b}, got {float(quad_result)}"
        assert np.sign(numpy_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {a} % {b}"
        return

    np.testing.assert_allclose(float(quad_result), numpy_result, rtol=1e-10, atol=1e-15,
                               err_msg=f"Value mismatch for {a} % {b}")

    if numpy_result == 0.0:
        numpy_sign = np.signbit(numpy_result)
        quad_sign = np.signbit(quad_result)
        assert numpy_sign == quad_sign, f"Zero sign mismatch for {a} % {b}: numpy={numpy_sign}, quad={quad_sign}"

    # Check that non-zero results have correct sign relative to divisor
    if numpy_result != 0.0 and not np.isnan(b) and not np.isinf(b) and b != 0.0:
        # In Python mod, non-zero result should have same sign as divisor (or be zero)
        result_negative = float(quad_result) < 0
        divisor_negative = b < 0
        numpy_negative = numpy_result < 0

        assert result_negative == numpy_negative, f"Sign mismatch for {a} % {b}: quad={result_negative}, numpy={numpy_negative}"
