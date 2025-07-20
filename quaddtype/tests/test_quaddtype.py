import pytest
import sys
import numpy as np
import operator

from numpy_quaddtype import QuadPrecDType, QuadPrecision


def test_create_scalar_simple():
    assert isinstance(QuadPrecision("12.0"), QuadPrecision)
    assert isinstance(QuadPrecision(1.63), QuadPrecision)
    assert isinstance(QuadPrecision(1), QuadPrecision)


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

    # FIXME: @juntyr: replace with array_equal once isnan is supported
    with np.errstate(invalid="ignore"):
        assert (
            (np.float64(quad_result) == float_result) or
            (np.abs(np.float64(quad_result) - float_result) < 1e-10) or
            ((float_result != float_result) and (quad_result != quad_result))
        )


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
    if op in ["fmin", "fmax"]:
        pytest.skip("fmin and fmax ufuncs are not yet supported")

    op_func = getattr(np, op)
    quad_a = np.array([QuadPrecision(a)])
    quad_b = np.array([QuadPrecision(b)])
    float_a = np.array([float(a)])
    float_b = np.array([float(b)])

    quad_res = op_func(quad_a, quad_b)
    float_res = op_func(float_a, float_b)

    # FIXME: @juntyr: replace with array_equal once isnan is supported
    with np.errstate(invalid="ignore"):
        assert np.all((quad_res == float_res) | ((quad_res != quad_res) & (float_res != float_res)))


@pytest.mark.parametrize("op", ["amin", "amax", "nanmin", "nanmax"])
@pytest.mark.parametrize("a", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
@pytest.mark.parametrize("b", ["3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_array_aminmax(op, a, b):
    if op in ["nanmin", "nanmax"]:
        pytest.skip("fmin and fmax ufuncs are not yet supported")

    op_func = getattr(np, op)
    quad_ab = np.array([QuadPrecision(a), QuadPrecision(b)])
    float_ab = np.array([float(a), float(b)])

    quad_res = op_func(quad_ab)
    float_res = op_func(float_ab)

    # FIXME: @juntyr: replace with array_equal once isnan is supported
    with np.errstate(invalid="ignore"):
        assert np.all((quad_res == float_res) | ((quad_res != quad_res) & (float_res != float_res)))


@pytest.mark.parametrize("op,nop", [("neg", "negative"), ("pos", "positive"), ("abs", "absolute"), (None, "sign")])
@pytest.mark.parametrize("val", ["3.0", "-3.0", "12.5", "100.0", "0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
def test_unary_ops(op, nop, val):
    op_func = None if op is None else getattr(operator, op)
    nop_func = getattr(np, nop)
    
    quad_val = QuadPrecision(val)
    float_val = float(val)

    for op_func in [op_func, nop_func]:
        if op_func is None:
            continue

        quad_result = op_func(quad_val)
        float_result = op_func(float_val)

        # FIXME: @juntyr: replace with array_equal once isnan is supported
        # FIXME: @juntyr: also check the signbit once that is supported
        with np.errstate(invalid="ignore"):
            assert (
                (np.float64(quad_result) == float_result) or
                ((float_result != float_result) and (quad_result != quad_result))
            ), f"{op}({val}) should be {float_result}, but got {quad_result}"


def test_inf():
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
    assert all(x == y for x, y in zip(result, expected))
