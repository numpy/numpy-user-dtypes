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


def test_smallest_subnormal_value():
    """Test that smallest_subnormal has the correct value across all platforms."""
    smallest_sub = numpy_quaddtype.smallest_subnormal
    repr_str = repr(smallest_sub)
    
    # The repr should show QuadPrecision('6.0e-4966', backend='sleef')
    assert "6.0e-4966" in repr_str, f"Expected '6.0e-4966' in repr, got {repr_str}"
    
    assert smallest_sub > 0, "smallest_subnormal should be positive"

@pytest.mark.parametrize("name,value", [("bits", 128), ("precision", 33)])
def test_finfo_int_constant(name, value):
    assert getattr(numpy_quaddtype, name) == value


@pytest.mark.parametrize("dtype", [
    "bool",
    "byte", "int8", "ubyte", "uint8",
    "short", "int16", "ushort", "uint16",
    "int", "int32", "uint", "uint32",
    "long", "ulong",
    "longlong", "int64", "ulonglong", "uint64",
    "half", "float16",
    "float", "float32",
    "double", "float64",
    "longdouble", "float96", "float128",
])
def test_supported_astype(dtype):
    if dtype in ("float96", "float128") and getattr(np, dtype, None) is None:
        pytest.skip(f"{dtype} is unsupported on the current platform")

    orig = np.array(1, dtype=dtype)
    quad = orig.astype(QuadPrecDType, casting="safe")
    back = quad.astype(dtype, casting="unsafe")

    assert quad == 1
    assert back == orig


@pytest.mark.parametrize("dtype", ["S10", "U10", "T", "V10", "datetime64[ms]", "timedelta64[ms]"])
def test_unsupported_astype(dtype):
    if dtype == "V10":
        with pytest.raises(TypeError, match="cast"):
          np.ones((3, 3), dtype="V10").astype(QuadPrecDType, casting="unsafe")
    else:
      with pytest.raises(TypeError, match="cast"):
          np.array(1, dtype=dtype).astype(QuadPrecDType, casting="unsafe")

      with pytest.raises(TypeError, match="cast"):
          np.array(QuadPrecision(1)).astype(dtype, casting="unsafe")


def test_basic_equality():
    assert QuadPrecision("12") == QuadPrecision(
        "12.0") == QuadPrecision("12.00")


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "pow", "copysign"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_binary_ops(op, a, b):
    if op == "truediv" and float(b) == 0:
        pytest.xfail("float division by zero")

    op_func = getattr(operator, op, None) or getattr(np, op)
    quad_a = QuadPrecision(a)
    quad_b = QuadPrecision(b)
    float_a = float(a)
    float_b = float(b)

    quad_result = op_func(quad_a, quad_b)
    float_result = op_func(float_a, float_b)

    np.testing.assert_allclose(np.float64(quad_result), float_result, atol=1e-10, rtol=0, equal_nan=True)

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({a}, {b})"


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

    # native implementation may not be sensitive to zero signs
    #  but we want to enforce it for the quad dtype
    # e.g. min(+0.0, -0.0) = -0.0
    if float_a == 0.0 and float_b == 0.0:
        assert float_res == 0.0
        float_res = np.copysign(0.0, op_func(np.copysign(1.0, float_a), np.copysign(1.0, float_b)))

    np.testing.assert_array_equal(quad_res.astype(float), float_res)

    # Check sign for zero results
    if float_res == 0.0:
        assert np.signbit(float_res) == np.signbit(
            quad_res), f"Zero sign mismatch for {op}({a}, {b})"


@pytest.mark.parametrize("op", ["amin", "amax", "nanmin", "nanmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_aminmax(op, a, b):
    op_func = getattr(np, op)
    quad_ab = np.array([QuadPrecision(a), QuadPrecision(b)])
    float_ab = np.array([float(a), float(b)])

    quad_res = op_func(quad_ab)
    float_res = op_func(float_ab)

    # native implementation may not be sensitive to zero signs
    #  but we want to enforce it for the quad dtype
    # e.g. min(+0.0, -0.0) = -0.0
    if float(a) == 0.0 and float(b) == 0.0:
        assert float_res == 0.0
        float_res = np.copysign(0.0, op_func(np.array([np.copysign(1.0, float(a)), np.copysign(1.0, float(b))])))

    np.testing.assert_array_equal(np.array(quad_res).astype(float), float_res)

    # Check sign for zero results
    if float_res == 0.0:
        assert np.signbit(float_res) == np.signbit(
            quad_res), f"Zero sign mismatch for {op}({a}, {b})"


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

        if (float_result == 0.0) and (op not in ["signbit", "isfinite", "isinf", "isnan"]):
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


def test_rint_near_halfway():
    assert np.rint(QuadPrecision("7.4999999999999999")) == 7
    assert np.rint(QuadPrecision("7.49999999999999999")) == 7
    assert np.rint(QuadPrecision("7.5")) == 8


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

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({a}, {b})"


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
    # Edge cases
    "0.0", "-0.0",
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

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"

def test_inf():
    assert QuadPrecision("inf") > QuadPrecision("1e1000")
    assert np.signbit(QuadPrecision("inf")) == 0
    assert QuadPrecision("-inf") < QuadPrecision("-1e1000")
    assert np.signbit(QuadPrecision("-inf")) == 1


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


@pytest.mark.parametrize("op", ["sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"])
@pytest.mark.parametrize("val", [
    # Basic cases
    "0.0", "-0.0", "1.0", "-1.0", "2.0", "-2.0",
    # Small values
    "1e-10", "-1e-10", "1e-15", "-1e-15",
    # Values near one
    "0.9", "-0.9", "0.9999", "-0.9999",
    "1.1", "-1.1", "1.0001", "-1.0001",
    # Medium values
    "10.0", "-10.0", "20.0", "-20.0",
    # Large values
    "100.0", "200.0", "700.0", "1000.0", "1e100", "1e308",
    "-100.0", "-200.0", "-700.0", "-1000.0", "-1e100", "-1e308",
    # Fractional values
    "0.5", "-0.5", "1.5", "-1.5", "2.5", "-2.5",
    # Special values
    "inf", "-inf", "nan", "-nan"
])
def test_hyperbolic_functions(op, val):
    """Comprehensive test for hyperbolic functions: sinh, cosh, tanh, arcsinh, arccosh, arctanh"""
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

    # For finite non-zero results
    # Use relative tolerance for exponential functions due to their rapid growth
    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
@pytest.mark.parametrize("val", [
    # Basic test values
    1.0, 2.0, 4.0, 8.0, 16.0,
    # Fractional values  
    0.5, 0.25, 0.125, 0.0625,
    # Negative values
    -1.0, -2.0, -0.5, -0.25,
    # Powers of 2 near boundaries
    1024.0, 2048.0, 0.00048828125,  # 2^10, 2^11, 2^-11
    # Large values
    1e10, 1e20, 1e30,
    # Small values  
    1e-10, 1e-20, 1e-30,
    # Edge cases
    np.finfo(np.float64).max, np.finfo(np.float64).min,
    np.finfo(np.float64).smallest_normal, np.finfo(np.float64).smallest_subnormal,
])
def test_frexp_finite_values(backend, val):
    """Test frexp for finite values comparing against numpy's float64 frexp."""
    quad_dtype = QuadPrecDType(backend=backend)
    quad_arr = np.array([val], dtype=quad_dtype)
    
    # Get frexp results for quad precision
    quad_mantissa, quad_exponent = np.frexp(quad_arr)
    
    # Get frexp results for float64 for comparison
    float64_mantissa, float64_exponent = np.frexp(np.array([val], dtype=np.float64))
    
    # Convert results for comparison
    quad_mantissa_float = float(quad_mantissa[0])
    quad_exponent_int = int(quad_exponent[0])
    
    # For finite values, mantissa should be in [0.5, 1) and val = mantissa * 2^exponent
    if val != 0.0:
        # Convert results for comparison (do this early)
        quad_mantissa_float = float(quad_mantissa[0]) if abs(val) <= 1e300 else 0.0
        
        # For very large values that might overflow Python float, check using quad precision
        if abs(val) > 1e300:  # Too large for safe Python float conversion
            # Use quad precision comparisons
            half_quad = np.array([0.5], dtype=quad_dtype)[0]
            one_quad = np.array([1.0], dtype=quad_dtype)[0]
            abs_mantissa = np.abs(quad_mantissa[0])
            assert np.greater_equal(abs_mantissa, half_quad), f"Mantissa magnitude too small for {val}"
            assert np.less(abs_mantissa, one_quad), f"Mantissa magnitude too large for {val}"
        else:
            # Safe to convert to Python float for smaller values
            assert 0.5 <= abs(quad_mantissa_float) < 1.0, f"Mantissa {quad_mantissa_float} not in [0.5, 1) for {val}"
        
        # For very large values, avoid overflow in reconstruction test
        # Just check that the relationship holds conceptually
        if abs(quad_exponent_int) < 1000 and abs(val) <= 1e300:  # Avoid overflow in 2**exponent
            # Reconstruct the original value
            reconstructed = quad_mantissa_float * (2.0 ** quad_exponent_int)
            np.testing.assert_allclose(reconstructed, val, rtol=1e-14, atol=1e-300,
                                     err_msg=f"frexp reconstruction failed for {val}")
        
        if abs(val) <= np.finfo(np.float64).max and abs(val) >= np.finfo(np.float64).smallest_normal and abs(val) <= 1e300:
            np.testing.assert_allclose(quad_mantissa_float, float64_mantissa[0], rtol=1e-14,
                                     err_msg=f"Mantissa mismatch for {val}")
            assert quad_exponent_int == float64_exponent[0], f"Exponent mismatch for {val}"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
def test_frexp_special_values(backend):
    """Test frexp for special values (zero, inf, nan)."""
    quad_dtype = QuadPrecDType(backend=backend)
    
    # Test zero
    zero_arr = np.array([0.0], dtype=quad_dtype)
    mantissa, exponent = np.frexp(zero_arr)
    assert float(mantissa[0]) == 0.0, "frexp(0.0) mantissa should be 0.0"
    assert int(exponent[0]) == 0, "frexp(0.0) exponent should be 0"
    
    # Test negative zero
    neg_zero_arr = np.array([-0.0], dtype=quad_dtype)
    mantissa, exponent = np.frexp(neg_zero_arr)
    mantissa_val = float(mantissa[0])
    assert mantissa_val == 0.0, "frexp(-0.0) mantissa should be 0.0"
    assert int(exponent[0]) == 0, "frexp(-0.0) exponent should be 0"
    
    # Test positive infinity
    inf_arr = np.array([np.inf], dtype=quad_dtype)
    mantissa, exponent = np.frexp(inf_arr)
    assert np.isinf(float(mantissa[0])), "frexp(inf) mantissa should be inf"
    assert float(mantissa[0]) > 0, "frexp(inf) mantissa should be positive"
    
    # Test negative infinity
    neg_inf_arr = np.array([-np.inf], dtype=quad_dtype)
    mantissa, exponent = np.frexp(neg_inf_arr)
    assert np.isinf(float(mantissa[0])), "frexp(-inf) mantissa should be inf"
    assert float(mantissa[0]) < 0, "frexp(-inf) mantissa should be negative"
    
    # Test NaN
    nan_arr = np.array([np.nan], dtype=quad_dtype)
    mantissa, exponent = np.frexp(nan_arr)
    assert np.isnan(float(mantissa[0])), "frexp(nan) mantissa should be nan"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
def test_frexp_array_operations(backend):
    """Test frexp on arrays of different sizes and shapes."""
    quad_dtype = QuadPrecDType(backend=backend)
    
    # Test 1D array
    values_1d = [0.5, 1.0, 2.0, 4.0, 8.0, -1.0, -0.25]
    quad_arr_1d = np.array(values_1d, dtype=quad_dtype)
    mantissa_1d, exponent_1d = np.frexp(quad_arr_1d)
    
    assert mantissa_1d.shape == quad_arr_1d.shape, "Mantissa shape should match input"
    assert exponent_1d.shape == quad_arr_1d.shape, "Exponent shape should match input"
    assert mantissa_1d.dtype == quad_dtype, "Mantissa should have quad precision dtype"
    assert exponent_1d.dtype == np.int32, "Exponent should have int32 dtype"
    
    # Verify each element
    for i, val in enumerate(values_1d):
        if val != 0.0:
            mantissa_float = float(mantissa_1d[i])
            exponent_int = int(exponent_1d[i])
            reconstructed = mantissa_float * (2.0 ** exponent_int)
            np.testing.assert_allclose(reconstructed, val, rtol=1e-14)
    
    # Test 2D array
    values_2d = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]], dtype=float)
    quad_arr_2d = values_2d.astype(quad_dtype)
    mantissa_2d, exponent_2d = np.frexp(quad_arr_2d)
    
    assert mantissa_2d.shape == quad_arr_2d.shape, "2D mantissa shape should match input"
    assert exponent_2d.shape == quad_arr_2d.shape, "2D exponent shape should match input"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
def test_frexp_dtype_consistency(backend):
    """Test that frexp output dtypes are consistent."""
    quad_dtype = QuadPrecDType(backend=backend)
    
    # Single value
    arr = np.array([3.14159], dtype=quad_dtype)
    mantissa, exponent = np.frexp(arr)
    
    # Check mantissa dtype matches input
    assert mantissa.dtype == quad_dtype, f"Mantissa dtype {mantissa.dtype} should match input {quad_dtype}"
    
    # Check exponent dtype is int32
    assert exponent.dtype == np.int32, f"Exponent dtype should be int32, got {exponent.dtype}"
    
    # Test with different input shapes
    for shape in [(5,), (2, 3), (2, 2, 2)]:
        arr = np.ones(shape, dtype=quad_dtype)
        mantissa, exponent = np.frexp(arr)
        assert mantissa.dtype == quad_dtype
        assert exponent.dtype == np.int32
        assert mantissa.shape == shape
        assert exponent.shape == shape


def test_frexp_finfo_compatibility():
    """Test that frexp works correctly with np.finfo to enable machep/negep calculation."""
    # This is the key test that was failing before implementing frexp
    quad_dtype = QuadPrecDType()
    
    # This should work without raising _UFuncNoLoopError
    try:
        finfo = np.finfo(quad_dtype)
        
        # Try to access basic properties that should work
        assert hasattr(finfo, 'dtype'), "finfo should have dtype attribute"
        
        # For custom dtypes, some properties may not be available
        # The key test is that frexp ufunc is registered and callable
        quad_arr = np.array([1.0], dtype=quad_dtype)
        mantissa, exponent = np.frexp(quad_arr)
        
        assert len(mantissa) == 1, "frexp should return mantissa array"
        assert len(exponent) == 1, "frexp should return exponent array"
        
    except AttributeError as e:
        # Some finfo properties may not be available for custom dtypes
        # The important thing is that frexp works
        quad_arr = np.array([1.0], dtype=quad_dtype)
        mantissa, exponent = np.frexp(quad_arr)
        
        assert len(mantissa) == 1, "frexp should return mantissa array"
        assert len(exponent) == 1, "frexp should return exponent array"


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
def test_frexp_edge_cases_quad_precision(backend):
    """Test frexp with values specific to quad precision range."""
    quad_dtype = QuadPrecDType(backend=backend)
    
    # Test with more reasonable values that won't overflow/underflow
    test_values = [
        1.0,  # Basic case
        2.0**100,  # Large but manageable 
        2.0**(-100),  # Small but manageable
        2.0**1000,  # Very large
        2.0**(-1000),  # Very small
    ]
    
    for val in test_values:
        arr = np.array([val], dtype=quad_dtype)
        mantissa, exponent = np.frexp(arr)
        
        # Test that mantissa is in quad precision format
        assert mantissa.dtype == quad_dtype, f"Mantissa should have quad dtype, got {mantissa.dtype}"
        assert exponent.dtype == np.int32, f"Exponent should be int32, got {exponent.dtype}"
        
        # Test reconstruction using quad precision arithmetic
        # mantissa * 2^exponent should equal original value
        pow2_exp = np.array([2.0], dtype=quad_dtype) ** exponent[0]
        reconstructed = mantissa[0] * pow2_exp
        
        # Convert both to the same dtype for comparison
        original_quad = np.array([val], dtype=quad_dtype)
        
        # For reasonable values, reconstruction should be exact
        if abs(val) >= 1e-300 and abs(val) <= 1e300:
            np.testing.assert_array_equal(reconstructed, original_quad[0], 
                                        err_msg=f"Reconstruction failed for {val}")
        
        # Test that mantissa is in correct range (using quad precision comparisons)
        if not np.equal(mantissa[0], 0.0):  # Skip zero case
            half_quad = np.array([0.5], dtype=quad_dtype)[0]
            one_quad = np.array([1.0], dtype=quad_dtype)[0]
            
            # |mantissa| should be in [0.5, 1.0) range
            abs_mantissa = np.abs(mantissa[0])
            assert np.greater_equal(abs_mantissa, half_quad), f"Mantissa magnitude too small for {val}"
            assert np.less(abs_mantissa, one_quad), f"Mantissa magnitude too large for {val}"
