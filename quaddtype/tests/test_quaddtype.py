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


@pytest.mark.parametrize("name", ["max_value", "epsilon", "smallest_normal", "smallest_subnormal", "resolution"])
def test_finfo_constant(name):
    assert isinstance(getattr(numpy_quaddtype, name), QuadPrecision)


@pytest.mark.parametrize("name,value", [("bits", 128), ("precision", 33)])
def test_finfo_int_constant(name, value):
    assert getattr(numpy_quaddtype, name) == value


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


@pytest.mark.parametrize("op", ["floor", "ceil", "trunc", "rint"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "3.2", "-3.2", "3.8", "-3.8", "0.1", "-0.1",
    # Edge cases around integers
    "3.0", "-3.0", "0.0", "-0.0", "1.0", "-1.0",
    # Halfway cases (important for rint)
    "2.5", "-2.5", "3.5", "-3.5", "0.5", "-0.5",
    # Large numbers
    "1e10", "-1e10", "1e15", "-1e15",
    # Small fractional numbers
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_rounding_functions(op, val):
    """Comprehensive test for rounding functions: floor, ceil, trunc, rint"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite results, check value and sign
    np.testing.assert_allclose(float(quad_result), float_result, rtol=1e-15, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


@pytest.mark.parametrize("op", ["exp", "exp2"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "0.0", "-0.0", "1.0", "-1.0", "2.0", "-2.0",
    # Small values (should be close to 1)
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Medium values
    "10.0", "-10.0", "20.0", "-20.0",
    # Values that might cause overflow
    "100.0", "200.0", "700.0", "1000.0",
    # Values that might cause underflow
    "-100.0", "-200.0", "-700.0", "-1000.0",
    # Fractional values
    "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_exponential_functions(op, val):
    """Comprehensive test for exponential functions: exp, exp2"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # Handle underflow to zero
    if float_result == 0.0:
        assert float(
            quad_result) == 0.0, f"Expected 0 for {op}({val}), got {float(quad_result)}"
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"
        return

    # For finite non-zero results
    # Use relative tolerance for exponential functions due to their rapid growth
    rtol = 1e-14 if abs(float_result) < 1e100 else 1e-10
    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")


@pytest.mark.parametrize("op", ["log", "log2", "log10"])
@pytest.mark.parametrize("val", [
    # Basic positive cases
    "1.0", "2.0", "10.0", "100.0", "1000.0",
    # Values close to 1 (important for log accuracy)
    "1.01", "0.99", "1.001", "0.999", "1.0001", "0.9999",
    # Small positive values
    "1e-10", "1e-15", "1e-100", "1e-300",
    # Large positive values
    "1e10", "1e15", "1e100", "1e300",
    # Fractional values
    "0.5", "0.1", "0.01", "2.5", "5.5", "25.0",
    # Edge cases
    "0.0", "-0.0",  # Should give -inf
    # Invalid domain (negative values) - should give NaN
    "-1.0", "-2.0", "-0.5", "-10.0",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_logarithmic_functions(op, val):
    """Comprehensive test for logarithmic functions: log, log2, log10"""
    op_func = getattr(np, op)

    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = op_func(quad_val)
    float_result = op_func(float_val)

    # Handle NaN cases (negative values, NaN input)
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for {op}({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for {op}({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for {op}({val})"
        return

    # For finite results
    # Use higher tolerance for values very close to 1 where log is close to 0
    if abs(float(val) - 1.0) < 1e-10:
        rtol = 1e-10
        atol = 1e-15
    else:
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for {op}({val})")


@pytest.mark.parametrize("val", [
    # Basic cases around -1 (critical point for log1p)
    "-0.5", "-0.1", "-0.01", "-0.001", "-0.0001",
    # Cases close to 0 (where log1p is most accurate)
    "1e-10", "-1e-10", "1e-15", "-1e-15", "1e-20", "-1e-20",
    # Larger positive values
    "0.1", "0.5", "1.0", "2.0", "10.0", "100.0",
    # Edge case at -1 (should give -inf)
    "-1.0",
    # Invalid domain (< -1) - should give NaN
    "-1.1", "-2.0", "-10.0",
    # Large positive values
    "1e10", "1e15", "1e100",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_log1p(val):
    """Comprehensive test for log1p function"""
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.log1p(quad_val)
    float_result = np.log1p(float_val)

    # Handle NaN cases (values < -1, NaN input)
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for log1p({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for log1p({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for log1p({val})"
        return

    # For finite results
    # log1p is designed for high accuracy near 0, so use tight tolerances
    if abs(float(val)) < 1e-10:
        rtol = 1e-15
        atol = 1e-20
    else:
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for log1p({val})")

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
