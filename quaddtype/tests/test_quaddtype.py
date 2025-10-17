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


def test_smallest_subnormal_value():
    """Test that smallest_subnormal has the correct value across all platforms."""
    smallest_sub = numpy_quaddtype.smallest_subnormal
    repr_str = repr(smallest_sub)
    
    # The repr should show QuadPrecision('6.0e-4966', backend='sleef')
    assert "6.0e-4966" in repr_str, f"Expected '6.0e-4966' in repr, got {repr_str}"
    
    assert smallest_sub > 0, "smallest_subnormal should be positive"


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
            quad_result), f"Zero sign mismatch"


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
    op = "log1p"
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


@pytest.mark.parametrize("val", [
    # Cases close to 0 (where expm1 is most accurate and important)
    "0.0", "-0.0",
    "1e-10", "-1e-10", "1e-15", "-1e-15", "1e-20", "-1e-20",
    "1e-100", "-1e-100", "1e-300", "-1e-300",
    # Small values
    "0.001", "-0.001", "0.01", "-0.01", "0.1", "-0.1",
    # Moderate values
    "0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0",
    # Larger values
    "5.0", "-5.0", "10.0", "-10.0", "20.0", "-20.0",
    # Values that test exp behavior
    "50.0", "-50.0", "100.0", "-100.0",
    # Large positive values (exp(x) grows rapidly)
    "200.0", "500.0", "700.0",
    # Large negative values (should approach -1)
    "-200.0", "-500.0", "-700.0", "-1000.0",
    # Special values
    "inf",   # Should give inf
    "-inf",  # Should give -1
    "nan", "-nan"
])
def test_expm1(val):
    """Comprehensive test for expm1 function: exp(x) - 1
    
    This function provides greater precision than exp(x) - 1 for small values of x.
    """
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.expm1(quad_val)
    float_result = np.expm1(float_val)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(
            float(quad_result)), f"Expected NaN for expm1({val}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(
            float(quad_result)), f"Expected inf for expm1({val}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(
            float(quad_result)), f"Infinity sign mismatch for expm1({val})"
        return

    # For finite results
    # expm1 is designed for high accuracy near 0, so use tight tolerances for small inputs
    if abs(float(val)) < 1e-10:
        rtol = 1e-15
        atol = 1e-20
    elif abs(float_result) < 1:
        rtol = 1e-14
        atol = 1e-15
    else:
        # For larger results, use relative tolerance
        rtol = 1e-14
        atol = 1e-15

    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=atol,
                               err_msg=f"Value mismatch for expm1({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for expm1({val})"


@pytest.mark.parametrize("x", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values (test numerical stability)
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
@pytest.mark.parametrize("y", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
def test_logaddexp(x, y):
    """Comprehensive test for logaddexp function: log(exp(x) + exp(y))"""
    quad_x = QuadPrecision(x)
    quad_y = QuadPrecision(y)
    float_x = float(x)
    float_y = float(y)
    
    quad_result = np.logaddexp(quad_x, quad_y)
    float_result = np.logaddexp(float_x, float_y)
    
    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for logaddexp({x}, {y}), got {float(quad_result)}"
        return
    
    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for logaddexp({x}, {y}), got {float(quad_result)}"
        if not np.isnan(float_result):
            assert np.sign(float_result) == np.sign(float(quad_result)), \
                f"Infinity sign mismatch for logaddexp({x}, {y})"
        return
    
    # For finite results, check with appropriate tolerance
    # logaddexp is numerically sensitive, especially for large differences
    if abs(float_x - float_y) > 50:
        # When values differ greatly, result should be close to max(x, y)
        rtol = 1e-10
        atol = 1e-10
    else:
        rtol = 1e-13
        atol = 1e-15
    
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=rtol, atol=atol,
        err_msg=f"Value mismatch for logaddexp({x}, {y})"
    )


def test_logaddexp_special_properties():
    """Test special mathematical properties of logaddexp"""
    # logaddexp(x, x) = x + log(2)
    x = QuadPrecision("2.0")
    result = np.logaddexp(x, x)
    expected = float(x) + np.log(2.0)
    np.testing.assert_allclose(float(result), expected, rtol=1e-14)
    
    # logaddexp(x, -inf) = x
    x = QuadPrecision("5.0")
    result = np.logaddexp(x, QuadPrecision("-inf"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp(-inf, x) = x
    result = np.logaddexp(QuadPrecision("-inf"), x)
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp(-inf, -inf) = -inf
    result = np.logaddexp(QuadPrecision("-inf"), QuadPrecision("-inf"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # logaddexp(inf, anything) = inf
    result = np.logaddexp(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # logaddexp(anything, inf) = inf
    result = np.logaddexp(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # Commutativity: logaddexp(x, y) = logaddexp(y, x)
    x = QuadPrecision("3.0")
    y = QuadPrecision("5.0")
    result1 = np.logaddexp(x, y)
    result2 = np.logaddexp(y, x)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-14)


@pytest.mark.parametrize("x", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values (test numerical stability)
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
@pytest.mark.parametrize("y", [
    # Regular values
    "0.0", "1.0", "2.0", "-1.0", "-2.0", "0.5", "-0.5",
    # Large values
    "100.0", "1000.0", "-100.0", "-1000.0",
    # Small values
    "1e-10", "-1e-10", "1e-20", "-1e-20",
    # Special values
    "inf", "-inf", "nan", "-nan", "-0.0"
])
def test_logaddexp2(x, y):
    """Comprehensive test for logaddexp2 function: log2(2^x + 2^y)"""
    quad_x = QuadPrecision(x)
    quad_y = QuadPrecision(y)
    float_x = float(x)
    float_y = float(y)
    
    quad_result = np.logaddexp2(quad_x, quad_y)
    float_result = np.logaddexp2(float_x, float_y)
    
    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for logaddexp2({x}, {y}), got {float(quad_result)}"
        return
    
    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for logaddexp2({x}, {y}), got {float(quad_result)}"
        if not np.isnan(float_result):
            assert np.sign(float_result) == np.sign(float(quad_result)), \
                f"Infinity sign mismatch for logaddexp2({x}, {y})"
        return
    
    # For finite results, check with appropriate tolerance
    # logaddexp2 is numerically sensitive, especially for large differences
    if abs(float_x - float_y) > 50:
        # When values differ greatly, result should be close to max(x, y)
        rtol = 1e-10
        atol = 1e-10
    else:
        rtol = 1e-13
        atol = 1e-15
    
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=rtol, atol=atol,
        err_msg=f"Value mismatch for logaddexp2({x}, {y})"
    )


def test_logaddexp2_special_properties():
    """Test special mathematical properties of logaddexp2"""
    # logaddexp2(x, x) = x + 1 (since log2(2^x + 2^x) = log2(2 * 2^x) = log2(2) + log2(2^x) = 1 + x)
    x = QuadPrecision("2.0")
    result = np.logaddexp2(x, x)
    expected = float(x) + 1.0
    np.testing.assert_allclose(float(result), expected, rtol=1e-14)
    
    # logaddexp2(x, -inf) = x
    x = QuadPrecision("5.0")
    result = np.logaddexp2(x, QuadPrecision("-inf"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp2(-inf, x) = x
    result = np.logaddexp2(QuadPrecision("-inf"), x)
    np.testing.assert_allclose(float(result), float(x), rtol=1e-14)
    
    # logaddexp2(-inf, -inf) = -inf
    result = np.logaddexp2(QuadPrecision("-inf"), QuadPrecision("-inf"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # logaddexp2(inf, anything) = inf
    result = np.logaddexp2(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # logaddexp2(anything, inf) = inf
    result = np.logaddexp2(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # Commutativity: logaddexp2(x, y) = logaddexp2(y, x)
    x = QuadPrecision("3.0")
    y = QuadPrecision("5.0")
    result1 = np.logaddexp2(x, y)
    result2 = np.logaddexp2(y, x)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-14)
    
    # Relationship with logaddexp: logaddexp2(x, y) = logaddexp(x*ln2, y*ln2) / ln2
    x = QuadPrecision("2.0")
    y = QuadPrecision("3.0")
    result_logaddexp2 = np.logaddexp2(x, y)
    ln2 = np.log(2.0)
    result_logaddexp = np.logaddexp(float(x) * ln2, float(y) * ln2) / ln2
    np.testing.assert_allclose(float(result_logaddexp2), result_logaddexp, rtol=1e-13)


@pytest.mark.parametrize(
    "x_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
@pytest.mark.parametrize(
    "y_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
def test_true_divide(x_val, y_val):
    """Test true_divide ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.true_divide(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.true_divide(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for true_divide({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for true_divide({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for true_divide({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-14,
            err_msg=f"Mismatch for true_divide({x_val}, {y_val})"
        )


def test_true_divide_special_properties():
    """Test special mathematical properties of true_divide"""
    # Division by 1 returns the original value
    x = QuadPrecision("42.123456789")
    result = np.true_divide(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), float(x), rtol=1e-30)
    
    # Division of 0 by any non-zero number is 0
    result = np.true_divide(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0
    
    # Division by 0 gives inf (with appropriate sign)
    result = np.true_divide(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    result = np.true_divide(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # 0 / 0 = NaN
    result = np.true_divide(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf / inf = NaN
    result = np.true_divide(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # inf / finite = inf
    result = np.true_divide(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    # finite / inf = 0
    result = np.true_divide(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert float(result) == 0.0
    
    # Self-division (x / x) = 1 for finite non-zero x
    x = QuadPrecision("7.123456789")
    result = np.true_divide(x, x)
    np.testing.assert_allclose(float(result), 1.0, rtol=1e-30)
    
    # Sign preservation: (-x) / y = -(x / y)
    x = QuadPrecision("5.5")
    y = QuadPrecision("2.2")
    result1 = np.true_divide(-x, y)
    result2 = -np.true_divide(x, y)
    np.testing.assert_allclose(float(result1), float(result2), rtol=1e-30)
    
    # Sign rule: negative / negative = positive
    result = np.true_divide(QuadPrecision("-6.0"), QuadPrecision("-2.0"))
    assert float(result) > 0
    np.testing.assert_allclose(float(result), 3.0, rtol=1e-30)


@pytest.mark.parametrize(
    "x_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
@pytest.mark.parametrize(
    "y_val",
    [
        0.0, 1.0, 2.0, -1.0, -2.0,
        0.5, -0.5,
        100.0, 1000.0, -100.0, -1000.0,
        1e-10, -1e-10, 1e-20, -1e-20,
        float("inf"), float("-inf"), float("nan"), float("-nan"), -0.0
    ]
)
def test_floor_divide(x_val, y_val):
    """Test floor_divide ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.floor_divide(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.floor_divide(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for floor_divide({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for floor_divide({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for floor_divide({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        # Use absolute tolerance for large numbers due to float64 precision limits
        atol = max(1e-10, abs(result_float64) * 1e-9) if abs(result_float64) > 1e6 else 1e-10
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-12, atol=atol,
            err_msg=f"Mismatch for floor_divide({x_val}, {y_val})"
        )
def test_floor_divide_special_properties():
    """Test special mathematical properties of floor_divide"""
    # floor_divide(x, 1) = floor(x)
    x = QuadPrecision("42.7")
    result = np.floor_divide(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), 42.0, rtol=1e-30)
    
    # floor_divide(0, non-zero) = 0
    result = np.floor_divide(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0
    
    # floor_divide by 0 gives inf (with appropriate sign)
    result = np.floor_divide(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) > 0
    
    result = np.floor_divide(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(result)) and float(result) < 0
    
    # 0 / 0 = NaN
    result = np.floor_divide(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf / inf = NaN
    result = np.floor_divide(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # inf / finite_nonzero = NaN (NumPy behavior)
    result = np.floor_divide(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    # finite / inf = 0
    result = np.floor_divide(QuadPrecision("100.0"), QuadPrecision("inf"))
    assert float(result) == 0.0
    
    # floor_divide rounds toward negative infinity
    result = np.floor_divide(QuadPrecision("7.0"), QuadPrecision("3.0"))
    assert float(result) == 2.0  # floor(7/3) = floor(2.333...) = 2
    
    result = np.floor_divide(QuadPrecision("-7.0"), QuadPrecision("3.0"))
    assert float(result) == -3.0  # floor(-7/3) = floor(-2.333...) = -3
    
    result = np.floor_divide(QuadPrecision("7.0"), QuadPrecision("-3.0"))
    assert float(result) == -3.0  # floor(7/-3) = floor(-2.333...) = -3
    
    result = np.floor_divide(QuadPrecision("-7.0"), QuadPrecision("-3.0"))
    assert float(result) == 2.0  # floor(-7/-3) = floor(2.333...) = 2
    
    # floor_divide(x, x) = 1 for positive finite non-zero x
    x = QuadPrecision("7.123456789")
    result = np.floor_divide(x, x)
    np.testing.assert_allclose(float(result), 1.0, rtol=1e-30)
    
    # Relationship with floor and true_divide
    x = QuadPrecision("10.5")
    y = QuadPrecision("3.2")
    result_floor_divide = np.floor_divide(x, y)
    result_floor_true_divide = np.floor(np.true_divide(x, y))
    np.testing.assert_allclose(float(result_floor_divide), float(result_floor_true_divide), rtol=1e-30)


@pytest.mark.parametrize("x_val,y_val", [
    (x, y) for x in [-1e10, -100.0, -7.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 7.0, 100.0, 1e10, 
                      float('inf'), float('-inf'), float('nan'),
                      -6.0, 6.0, -0.1, 0.1, -3.14159, 3.14159]
    for y in [-1e10, -100.0, -3.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 3.0, 100.0, 1e10,
              float('inf'), float('-inf'), float('nan'),
              -2.0, 2.0, -0.25, 0.25, -1.5, 1.5]
])
def test_fmod(x_val, y_val):
    """Test fmod ufunc with comprehensive edge cases"""
    x_quad = QuadPrecision(str(x_val))
    y_quad = QuadPrecision(str(y_val))
    
    # Compute using QuadPrecision
    result_quad = np.fmod(x_quad, y_quad)
    
    # Compute using float64 for comparison
    result_float64 = np.fmod(np.float64(x_val), np.float64(y_val))
    
    # Compare results
    if np.isnan(result_float64):
        assert np.isnan(float(result_quad)), f"Expected NaN for fmod({x_val}, {y_val})"
    elif np.isinf(result_float64):
        assert np.isinf(float(result_quad)), f"Expected inf for fmod({x_val}, {y_val})"
        assert np.sign(float(result_quad)) == np.sign(result_float64), f"Sign mismatch for fmod({x_val}, {y_val})"
    else:
        # For finite results, check relative tolerance
        atol = max(1e-10, abs(result_float64) * 1e-9) if abs(result_float64) > 1e6 else 1e-10
        np.testing.assert_allclose(
            float(result_quad), result_float64, rtol=1e-12, atol=atol,
            err_msg=f"Mismatch for fmod({x_val}, {y_val})"
        )
        
        # Critical: Check sign preservation for zero results
        if result_float64 == 0.0:
            assert np.signbit(result_quad) == np.signbit(result_float64), \
                f"Sign mismatch for zero result: fmod({x_val}, {y_val}), " \
                f"expected signbit={np.signbit(result_float64)}, got signbit={np.signbit(result_quad)}"


def test_fmod_special_properties():
    """Test special mathematical properties of fmod"""
    # fmod(x, 1) gives fractional part of x (with sign preserved)
    x = QuadPrecision("42.7")
    result = np.fmod(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(result), 0.7, rtol=1e-15, atol=1e-15)
    
    # fmod(0, non-zero) = 0 with correct sign
    result = np.fmod(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-0.0"), QuadPrecision("5.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    # fmod by 0 gives NaN
    result = np.fmod(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    result = np.fmod(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # 0 fmod 0 = NaN
    result = np.fmod(QuadPrecision("0.0"), QuadPrecision("0.0"))
    assert np.isnan(float(result))
    
    # inf fmod x = NaN
    result = np.fmod(QuadPrecision("inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    result = np.fmod(QuadPrecision("-inf"), QuadPrecision("100.0"))
    assert np.isnan(float(result))
    
    # x fmod inf = x (for finite x)
    result = np.fmod(QuadPrecision("100.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(result), 100.0, rtol=1e-30)
    
    result = np.fmod(QuadPrecision("-100.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(result), -100.0, rtol=1e-30)
    
    # inf fmod inf = NaN
    result = np.fmod(QuadPrecision("inf"), QuadPrecision("inf"))
    assert np.isnan(float(result))
    
    # fmod uses truncated division (rounds toward zero)
    # Result has same sign as dividend (first argument)
    result = np.fmod(QuadPrecision("7.0"), QuadPrecision("3.0"))
    assert float(result) == 1.0  # 7 - trunc(7/3)*3 = 7 - 2*3 = 1
    
    result = np.fmod(QuadPrecision("-7.0"), QuadPrecision("3.0"))
    assert float(result) == -1.0  # -7 - trunc(-7/3)*3 = -7 - (-2)*3 = -1
    
    result = np.fmod(QuadPrecision("7.0"), QuadPrecision("-3.0"))
    assert float(result) == 1.0  # 7 - trunc(7/-3)*(-3) = 7 - (-2)*(-3) = 1
    
    result = np.fmod(QuadPrecision("-7.0"), QuadPrecision("-3.0"))
    assert float(result) == -1.0  # -7 - trunc(-7/-3)*(-3) = -7 - 2*(-3) = -1
    
    # Sign preservation when result is exactly zero
    result = np.fmod(QuadPrecision("6.0"), QuadPrecision("3.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-6.0"), QuadPrecision("3.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    result = np.fmod(QuadPrecision("6.0"), QuadPrecision("-3.0"))
    assert float(result) == 0.0 and not np.signbit(result)
    
    result = np.fmod(QuadPrecision("-6.0"), QuadPrecision("-3.0"))
    assert float(result) == 0.0 and np.signbit(result)
    
    # Difference from mod/remainder (which uses floor division)
    # fmod result has sign of dividend, mod result has sign of divisor
    x = QuadPrecision("-7.0")
    y = QuadPrecision("3.0")
    fmod_result = np.fmod(x, y)
    mod_result = np.remainder(x, y)
    
    assert float(fmod_result) == -1.0  # sign of dividend (negative)
    assert float(mod_result) == 2.0    # sign of divisor (positive)
    
    # Relationship: x = trunc(x/y) * y + fmod(x, y)
    x = QuadPrecision("10.5")
    y = QuadPrecision("3.2")
    quotient = np.trunc(np.true_divide(x, y))
    remainder = np.fmod(x, y)
    reconstructed = np.add(np.multiply(quotient, y), remainder)
    np.testing.assert_allclose(float(reconstructed), float(x), rtol=1e-30)


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


@pytest.mark.parametrize("backend", ["sleef", "longdouble"])
@pytest.mark.parametrize("a,b", [
    # Basic cases - positive/positive
    (7.0, 3.0), (10.5, 3.2), (21.0, 4.0),
    
    # Positive/negative combinations
    (-7.0, 3.0), (7.0, -3.0), (-7.0, -3.0),
    (-10.5, 3.2), (10.5, -3.2), (-10.5, -3.2),

    # Zero dividend cases
    (0.0, 3.0), (-0.0, 3.0), (0.0, -3.0), (-0.0, -3.0),

    # Cases that result in zero remainder (exact division)
    (6.0, 3.0), (-6.0, 3.0), (6.0, -3.0), (-6.0, -3.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    (10.0, 2.0), (-10.0, 2.0), (10.0, -2.0), (-10.0, -2.0),

    # Fractional cases
    (7.5, 2.5), (-7.5, 2.5), (7.5, -2.5), (-7.5, -2.5),
    (0.75, 0.25), (-0.1, 0.3), (0.9, -1.0), (-1.1, -1.0),
    (3.14159, 1.0), (-3.14159, 1.0), (3.14159, -1.0), (-3.14159, -1.0),

    # Large/small numbers
    (1e10, 1e5), (-1e10, 1e5), (1e-10, 1e-5), (-1e-10, 1e-5),
    (1e15, 1e10), (1e-15, 1e-10),

    # Finite % infinity cases
    (5.0, float('inf')), (-5.0, float('inf')),
    (5.0, float('-inf')), (-5.0, float('-inf')),
    (0.0, float('inf')), (-0.0, float('-inf')),

    # NaN cases (should return NaN for both quotient and remainder)
    (float('nan'), 3.0), (3.0, float('nan')), (float('nan'), float('nan')),

    # Division by zero cases (should return inf/NaN)
    (5.0, 0.0), (-5.0, 0.0), (0.0, 0.0), (-0.0, 0.0),

    # Infinity dividend cases (should return NaN for both)
    (float('inf'), 3.0), (float('-inf'), 3.0),
    (float('inf'), float('inf')), (float('-inf'), float('-inf')),
    
    # Cases with dividend < divisor
    (1.0, 10.0), (-1.0, 10.0), (1.0, -10.0), (-1.0, -10.0),
    (0.5, 1.0), (0.1, 1.0), (0.001, 0.01),
])
def test_divmod(a, b, backend):
    """Comprehensive test for divmod operation against NumPy behavior"""
    if backend == "sleef":
        quad_a = QuadPrecision(str(a))
        quad_b = QuadPrecision(str(b))
    elif backend == "longdouble":
        quad_a = QuadPrecision(a, backend='longdouble')
        quad_b = QuadPrecision(b, backend='longdouble')
    
    float_a = np.float64(a)
    float_b = np.float64(b)

    # Compute divmod
    quad_quotient, quad_remainder = np.divmod(quad_a, quad_b)
    numpy_quotient, numpy_remainder = np.divmod(float_a, float_b)

    # Verify quotient
    if np.isnan(numpy_quotient):
        assert np.isnan(float(quad_quotient)), \
            f"Expected NaN quotient for divmod({a}, {b})"
    elif np.isinf(numpy_quotient):
        assert np.isinf(float(quad_quotient)) and \
               np.sign(numpy_quotient) == np.sign(float(quad_quotient)), \
            f"Expected inf quotient with matching sign for divmod({a}, {b})"
    else:
        # Adaptive tolerance for large quotients due to float64 conversion precision loss
        atol_q = abs(numpy_quotient) * 1e-8 if abs(numpy_quotient) > 1e6 else 1e-15
        np.testing.assert_allclose(
            float(quad_quotient), numpy_quotient, rtol=1e-9, atol=atol_q,
            err_msg=f"Quotient mismatch for divmod({a}, {b})"
        )
        if numpy_quotient == 0.0:
            assert np.signbit(numpy_quotient) == np.signbit(quad_quotient), \
                f"Zero quotient sign mismatch for divmod({a}, {b})"

    # Verify remainder
    if np.isnan(numpy_remainder):
        assert np.isnan(float(quad_remainder)), \
            f"Expected NaN remainder for divmod({a}, {b})"
    elif np.isinf(numpy_remainder):
        assert np.isinf(float(quad_remainder)) and \
               np.sign(numpy_remainder) == np.sign(float(quad_remainder)), \
            f"Expected inf remainder with matching sign for divmod({a}, {b})"
    else:
        # Standard tolerance for remainder comparison
        np.testing.assert_allclose(
            float(quad_remainder), numpy_remainder, rtol=1e-9, atol=1e-15,
            err_msg=f"Remainder mismatch for divmod({a}, {b})"
        )
        if numpy_remainder == 0.0:
            assert np.signbit(numpy_remainder) == np.signbit(quad_remainder), \
                f"Zero remainder sign mismatch for divmod({a}, {b})"
        elif not np.isnan(b) and not np.isinf(b) and b != 0.0:
            assert (float(quad_remainder) < 0) == (numpy_remainder < 0), \
                f"Remainder sign mismatch for divmod({a}, {b})"

    # Verify the fundamental property: a = quotient * b + remainder (for finite values)
    if not np.isnan(numpy_quotient) and not np.isinf(numpy_quotient) and \
       not np.isnan(numpy_remainder) and not np.isinf(numpy_remainder) and \
       not np.isnan(b) and not np.isinf(b) and b != 0.0:
        reconstructed = float(quad_quotient) * float(quad_b) + float(quad_remainder)
        np.testing.assert_allclose(
            reconstructed, float(quad_a), rtol=1e-10, atol=1e-15,
            err_msg=f"Property a = q*b + r failed for divmod({a}, {b})"
        )


def test_divmod_special_properties():
    """Test special mathematical properties of divmod"""
    # divmod(x, 1) should give (floor(x), 0)
    x = QuadPrecision("42.7")
    quotient, remainder = np.divmod(x, QuadPrecision("1.0"))
    np.testing.assert_allclose(float(quotient), 42.0, rtol=1e-30)
    np.testing.assert_allclose(float(remainder), 0.7, rtol=1e-14)
    
    # divmod(0, non-zero) should give (0, 0)
    quotient, remainder = np.divmod(QuadPrecision("0.0"), QuadPrecision("5.0"))
    assert float(quotient) == 0.0
    assert float(remainder) == 0.0
    
    # divmod by 0 gives (inf, NaN) for positive dividend
    quotient, remainder = np.divmod(QuadPrecision("1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(quotient)) and float(quotient) > 0
    assert np.isnan(float(remainder))
    
    quotient, remainder = np.divmod(QuadPrecision("-1.0"), QuadPrecision("0.0"))
    assert np.isinf(float(quotient)) and float(quotient) < 0
    assert np.isnan(float(remainder))
    
    # divmod(inf, finite) gives (NaN, NaN)
    quotient, remainder = np.divmod(QuadPrecision("inf"), QuadPrecision("5.0"))
    assert np.isnan(float(quotient))
    assert np.isnan(float(remainder))
    
    # divmod(finite, inf) gives (0, dividend)
    quotient, remainder = np.divmod(QuadPrecision("5.0"), QuadPrecision("inf"))
    np.testing.assert_allclose(float(quotient), 0.0, rtol=1e-30)
    np.testing.assert_allclose(float(remainder), 5.0, rtol=1e-30)
    
    # Verify equivalence with floor_divide and mod
    a = QuadPrecision("10.5")
    b = QuadPrecision("3.2")
    quotient, remainder = np.divmod(a, b)
    expected_quotient = np.floor_divide(a, b)
    expected_remainder = np.mod(a, b)
    np.testing.assert_allclose(float(quotient), float(expected_quotient), rtol=1e-30)
    np.testing.assert_allclose(float(remainder), float(expected_remainder), rtol=1e-30)


def test_divmod_array():
    """Test divmod with arrays"""
    a = np.array([10.5, 21.0, -7.5, 0.0], dtype=QuadPrecDType())
    b = np.array([3.2, 4.0, 2.5, 5.0], dtype=QuadPrecDType())
    
    quotients, remainders = np.divmod(a, b)
    
    # Check dtype
    assert quotients.dtype.name == "QuadPrecDType128"
    assert remainders.dtype.name == "QuadPrecDType128"
    
    # Check against NumPy float64
    a_float = np.array([10.5, 21.0, -7.5, 0.0], dtype=np.float64)
    b_float = np.array([3.2, 4.0, 2.5, 5.0], dtype=np.float64)
    expected_quotients, expected_remainders = np.divmod(a_float, b_float)
    
    for i in range(len(a)):
        np.testing.assert_allclose(
            float(quotients[i]), expected_quotients[i], rtol=1e-10, atol=1e-15,
            err_msg=f"Quotient mismatch at index {i}"
        )
        np.testing.assert_allclose(
            float(remainders[i]), expected_remainders[i], rtol=1e-10, atol=1e-15,
            err_msg=f"Remainder mismatch at index {i}"
        )


def test_divmod_broadcasting():
    """Test divmod with broadcasting"""
    # Scalar with array
    a = np.array([10.5, 21.0, 31.5], dtype=QuadPrecDType())
    b = QuadPrecision("3.0")
    
    quotients, remainders = np.divmod(a, b)
    
    assert quotients.dtype.name == "QuadPrecDType128"
    assert remainders.dtype.name == "QuadPrecDType128"
    assert len(quotients) == 3
    assert len(remainders) == 3
    
    # Check values
    expected_quotients = [3.0, 7.0, 10.0]
    expected_remainders = [1.5, 0.0, 1.5]
    
    for i in range(3):
        np.testing.assert_allclose(float(quotients[i]), expected_quotients[i], rtol=1e-14)
        np.testing.assert_allclose(float(remainders[i]), expected_remainders[i], rtol=1e-14)


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
    rtol = 1e-13 if abs(float_result) < 1e100 else 1e-10
    np.testing.assert_allclose(float(quad_result), float_result, rtol=rtol, atol=1e-15,
                               err_msg=f"Value mismatch for {op}({val})")

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(
            quad_result), f"Zero sign mismatch for {op}({val})"


class TestTypePomotionWithPythonAbstractTypes:
    """Tests for common_dtype handling of Python abstract dtypes (PyLongDType, PyFloatDType)"""
    
    def test_promotion_with_python_int(self):
        """Test that Python int promotes to QuadPrecDType"""
        # Create array from Python int
        arr = np.array([1, 2, 3], dtype=QuadPrecDType)
        assert arr.dtype.name == "QuadPrecDType128"
        assert len(arr) == 3
        assert float(arr[0]) == 1.0
        assert float(arr[1]) == 2.0
        assert float(arr[2]) == 3.0
    
    def test_promotion_with_python_float(self):
        """Test that Python float promotes to QuadPrecDType"""
        # Create array from Python float
        arr = np.array([1.5, 2.7, 3.14], dtype=QuadPrecDType)
        assert arr.dtype.name == "QuadPrecDType128"
        assert len(arr) == 3
        np.testing.assert_allclose(float(arr[0]), 1.5, rtol=1e-15)
        np.testing.assert_allclose(float(arr[1]), 2.7, rtol=1e-15)
        np.testing.assert_allclose(float(arr[2]), 3.14, rtol=1e-15)
    
    def test_result_dtype_binary_ops_with_python_types(self):
        """Test that binary operations between QuadPrecDType and Python scalars return QuadPrecDType"""
        quad_arr = np.array([QuadPrecision("1.0"), QuadPrecision("2.0")])
        
        # Addition with Python int
        result = quad_arr + 5
        assert result.dtype.name == "QuadPrecDType128"
        assert float(result[0]) == 6.0
        assert float(result[1]) == 7.0
        
        # Multiplication with Python float
        result = quad_arr * 2.5
        assert result.dtype.name == "QuadPrecDType128"
        np.testing.assert_allclose(float(result[0]), 2.5, rtol=1e-15)
        np.testing.assert_allclose(float(result[1]), 5.0, rtol=1e-15)
    
    def test_concatenate_with_python_types(self):
        """Test concatenation handles Python numeric types correctly"""
        quad_arr = np.array([QuadPrecision("1.0")])
        # This should work if promotion is correct
        int_arr = np.array([2], dtype=np.int64)
        
        # The result dtype should be QuadPrecDType
        result = np.concatenate([quad_arr, int_arr.astype(QuadPrecDType)])
        assert result.dtype.name == "QuadPrecDType128"
        assert len(result) == 2


@pytest.mark.parametrize("func,args,expected", [
    # arange tests
    (np.arange, (0, 10), list(range(10))),
    (np.arange, (0, 10, 2), [0, 2, 4, 6, 8]),
    (np.arange, (0.0, 5.0, 0.5), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]),
    (np.arange, (10, 0, -1), [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
    (np.arange, (-5, 5), list(range(-5, 5))),
    # linspace tests
    (np.linspace, (0, 10, 11), list(range(11))),
    (np.linspace, (0, 1, 5), [0.0, 0.25, 0.5, 0.75, 1.0]),
])
def test_fill_function(func, args, expected):
    """Test quadprec_fill function with arange and linspace"""
    arr = func(*args, dtype=QuadPrecDType())
    assert arr.dtype.name == "QuadPrecDType128"
    assert len(arr) == len(expected)
    for i, exp_val in enumerate(expected):
        np.testing.assert_allclose(float(arr[i]), float(exp_val), rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("base,exponent", [
    # Basic integer powers
    (2.0, 3.0), (3.0, 2.0), (10.0, 5.0), (5.0, 10.0),
    
    # Fractional powers
    (4.0, 0.5), (9.0, 0.5), (27.0, 1.0/3.0), (16.0, 0.25),
    (8.0, 2.0/3.0), (100.0, 0.5),
    
    # Negative bases with integer exponents
    (-2.0, 3.0), (-3.0, 2.0), (-2.0, 4.0), (-5.0, 3.0),
    
    # Negative bases with fractional exponents (should return NaN)
    (-1.0, 0.5), (-4.0, 0.5), (-1.0, 1.5), (-4.0, 1.5),
    (-2.0, 0.25), (-8.0, 1.0/3.0), (-5.0, 2.5), (-10.0, 0.75),
    (-1.0, -0.5), (-4.0, -1.5), (-2.0, -2.5),
    
    # Zero base cases
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 10.0),
    (0.0, 0.5), (0.0, -0.0),
    
    # Negative zero base
    (-0.0, 0.0), (-0.0, 1.0), (-0.0, 2.0), (-0.0, 3.0),
    
    # Base of 1
    (1.0, 0.0), (1.0, 1.0), (1.0, 100.0), (1.0, -100.0),
    (1.0, float('inf')), (1.0, float('-inf')), (1.0, float('nan')),
    
    # Base of -1
    (-1.0, 0.0), (-1.0, 1.0), (-1.0, 2.0), (-1.0, 3.0),
    (-1.0, float('inf')), (-1.0, float('-inf')),
    
    # Exponent of 0
    (2.0, 0.0), (100.0, 0.0), (-5.0, 0.0), (0.5, 0.0),
    (float('inf'), 0.0), (float('-inf'), 0.0), (float('nan'), 0.0),
    
    # Exponent of 1
    (2.0, 1.0), (100.0, 1.0), (-5.0, 1.0), (0.5, 1.0),
    (float('inf'), 1.0), (float('-inf'), 1.0),
    
    # Negative exponents
    (2.0, -1.0), (2.0, -2.0), (10.0, -3.0), (0.5, -1.0),
    (4.0, -0.5), (9.0, -0.5),
    
    # Infinity base
    (float('inf'), 0.0), (float('inf'), 1.0), (float('inf'), 2.0),
    (float('inf'), -1.0), (float('inf'), -2.0), (float('inf'), 0.5),
    (float('inf'), float('inf')), (float('inf'), float('-inf')),
    
    # Negative infinity base
    (float('-inf'), 0.0), (float('-inf'), 1.0), (float('-inf'), 2.0),
    (float('-inf'), 3.0), (float('-inf'), -1.0), (float('-inf'), -2.0),
    (float('-inf'), float('inf')), (float('-inf'), float('-inf')),
    
    # Infinity exponent
    (2.0, float('inf')), (0.5, float('inf')), (1.5, float('inf')),
    (2.0, float('-inf')), (0.5, float('-inf')), (1.5, float('-inf')),
    (0.0, float('inf')), (0.0, float('-inf')),
    
    # NaN cases
    (float('nan'), 0.0), (float('nan'), 1.0), (float('nan'), 2.0),
    (2.0, float('nan')), (0.0, float('nan')),
    (float('nan'), float('nan')), (float('nan'), float('inf')),
    (float('inf'), float('nan')),
    
    # Small and large values
    (1e-10, 2.0), (1e10, 2.0), (1e-10, 0.5), (1e10, 0.5),
    (2.0, 100.0), (2.0, -100.0), (0.5, 100.0), (0.5, -100.0),
])
def test_float_power(base, exponent):
    """
    Comprehensive test for float_power ufunc.
    
    float_power differs from power in that it always promotes to floating point.
    For floating-point dtypes like QuadPrecDType, it should behave identically to power.
    """
    quad_base = QuadPrecision(str(base)) if not (np.isnan(base) or np.isinf(base)) else QuadPrecision(base)
    quad_exp = QuadPrecision(str(exponent)) if not (np.isnan(exponent) or np.isinf(exponent)) else QuadPrecision(exponent)

    float_base = np.float64(base)
    float_exp = np.float64(exponent)

    quad_result = np.float_power(quad_base, quad_exp)
    float_result = np.float_power(float_base, float_exp)

    # Handle NaN cases
    if np.isnan(float_result):
        assert np.isnan(float(quad_result)), \
            f"Expected NaN for float_power({base}, {exponent}), got {float(quad_result)}"
        return

    # Handle infinity cases
    if np.isinf(float_result):
        assert np.isinf(float(quad_result)), \
            f"Expected inf for float_power({base}, {exponent}), got {float(quad_result)}"
        assert np.sign(float_result) == np.sign(float(quad_result)), \
            f"Infinity sign mismatch for float_power({base}, {exponent})"
        return

    # For finite results
    np.testing.assert_allclose(
        float(quad_result), float_result, 
        rtol=1e-13, atol=1e-15,
        err_msg=f"Value mismatch for float_power({base}, {exponent})"
    )

    # Check sign for zero results
    if float_result == 0.0:
        assert np.signbit(float_result) == np.signbit(quad_result), \
            f"Zero sign mismatch for float_power({base}, {exponent})"


@pytest.mark.parametrize("base,exponent", [
    # Test that float_power works with integer inputs (promotes to float)
    (2, 3),
    (4, 2),
    (10, 5),
    (-2, 3),
])
def test_float_power_integer_promotion(base, exponent):
    """
    Test that float_power works with integer inputs and promotes them to QuadPrecDType.
    This is the key difference from power - float_power always returns float types.
    """
    # Create arrays with integer inputs
    base_arr = np.array([base], dtype=QuadPrecDType())
    exp_arr = np.array([exponent], dtype=QuadPrecDType())

    result = np.float_power(base_arr, exp_arr)

    # Result should be QuadPrecDType
    assert result.dtype.name == "QuadPrecDType128"

    # Check the value
    expected = float(base) ** float(exponent)
    np.testing.assert_allclose(float(result[0]), expected, rtol=1e-13)


def test_float_power_array():
    """Test float_power with arrays"""
    bases = np.array([2.0, 4.0, 9.0, 16.0], dtype=QuadPrecDType())
    exponents = np.array([3.0, 0.5, 2.0, 0.25], dtype=QuadPrecDType())

    result = np.float_power(bases, exponents)
    expected = np.array([8.0, 2.0, 81.0, 2.0], dtype=np.float64)

    assert result.dtype.name == "QuadPrecDType128"
    for i in range(len(result)):
        np.testing.assert_allclose(float(result[i]), expected[i], rtol=1e-13)


@pytest.mark.parametrize("val", [
    # Positive values
    "3.0", "12.5", "100.0", "1e100", "0.0",
    # Negative values
    "-3.0", "-12.5", "-100.0", "-1e100", "-0.0",
    # Special values
    "inf", "-inf", "nan", "-nan",
    # Small values
    "1e-100", "-1e-100"
])
def test_fabs(val):
    """
    Test np.fabs ufunc for QuadPrecision dtype.
    fabs computes absolute values (positive magnitude) for floating-point numbers.
    It should behave identically to np.absolute for real (non-complex) types.
    """
    quad_val = QuadPrecision(val)
    float_val = float(val)

    quad_result = np.fabs(quad_val)
    float_result = np.fabs(float_val)

    # Test with both scalar and array
    quad_arr = np.array([quad_val], dtype=QuadPrecDType())
    quad_arr_result = np.fabs(quad_arr)

    # Check scalar result
    np.testing.assert_array_equal(np.array(quad_result).astype(float), float_result)

    # Check array result
    np.testing.assert_array_equal(quad_arr_result.astype(float)[0], float_result)

    # For zero results, check sign (should always be positive after fabs)
    if float_result == 0.0:
        assert not np.signbit(quad_result), f"fabs({val}) should not have negative sign"
        assert not np.signbit(quad_arr_result[0]), f"fabs({val}) should not have negative sign"
