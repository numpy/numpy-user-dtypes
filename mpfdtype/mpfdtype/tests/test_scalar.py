import pytest

import sys
import numpy as np
import operator

from mpfdtype import MPFDType, MPFloat


def test_create_scalar_simple():
    # currently inferring 53bit precision from float:
    assert MPFloat(12.).prec == 53
    # currently infers 64bit or 32bit depending on system:
    assert MPFloat(1).prec == sys.maxsize.bit_count() + 1

    assert MPFloat(MPFloat(12.)).prec == 53
    assert MPFloat(MPFloat(1)).prec == sys.maxsize.bit_count() + 1


def test_create_scalar_prec():
    assert MPFloat(1, prec=100).prec == 100
    assert MPFloat(12., prec=123).prec == 123
    assert MPFloat("12.234", prec=1000).prec == 1000

    mpf1 = MPFloat("12.4325", prec=120)
    mpf2 = MPFloat(mpf1, prec=150)
    assert mpf1 == mpf2
    assert mpf2.prec == 150


def test_basic_equality():
    assert MPFloat(12) == MPFloat(12.) == MPFloat("12.00", prec=10)


@pytest.mark.parametrize("val", [123532.543, 12893283.5])
def test_scalar_repr(val):
    # For non exponentials at least, the repr matches:
    val_repr = f"{val:e}".upper()
    expected = f"MPFloat('{val_repr}', prec=20)"
    assert repr(MPFloat(val, prec=20)) == expected

@pytest.mark.parametrize("op",
        ["add", "sub", "mul", "pow"])
@pytest.mark.parametrize("other", [3., 12.5, 100., np.nan, np.inf])
def test_binary_ops(op, other):
    # Generally, the math ops should behave the same as double math if they
    # use double precision (which they currently do).
    # (double could have errors, but not for these simple ops)
    op = getattr(operator, op)
    try:
        expected = op(12.5, other)
    except Exception as e:
        with pytest.raises(type(e)):
            op(MPFloat(12.5), other)
        with pytest.raises(type(e)):
            op(12.5, MPFloat(other))
        with pytest.raises(type(e)):
            op(MPFloat(12.5), MPFloat(other))
    else:
        if np.isnan(expected):
            # Avoiding isnan (which was also not implemented when written)
            res = op(MPFloat(12.5), other)
            assert res != res
            res = op(12.5, MPFloat(other))
            assert res != res
            res = op(MPFloat(12.5), MPFloat(other))
            assert res != res
        else:
            assert op(MPFloat(12.5), other) == expected
            assert op(12.5, MPFloat(other)) == expected
            assert op(MPFloat(12.5), MPFloat(other)) == expected


@pytest.mark.parametrize("op",
        ["eq", "ne", "le", "lt", "ge", "gt"])
@pytest.mark.parametrize("other", [3., 12.5, 100., np.nan, np.inf])
def test_comparisons(op, other):
    op = getattr(operator, op)
    expected = op(12.5, other)
    assert op(MPFloat(12.5), other) is expected
    assert op(12.5, MPFloat(other)) is expected
    assert op(MPFloat(12.5), MPFloat(other)) is expected


@pytest.mark.parametrize("op",
        ["neg", "pos", "abs"])
@pytest.mark.parametrize("val", [3., 12.5, 100., np.nan, np.inf])
def test_comparisons(op, val):
    op = getattr(operator, op)
    expected = op(val)
    if np.isnan(expected):
        assert op(MPFloat(val)) != op(MPFloat(val))
    else:
        assert op(MPFloat(val)) == expected
