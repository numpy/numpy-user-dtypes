from ._quaddtype_main import (
    QuadPrecision,
    QuadPrecDType
)

__all__ = ['QuadPrecision', 'QuadPrecDType', 'SleefQuadPrecision', 'LongDoubleQuadPrecision',
           'SleefQuadPrecDType', 'LongDoubleQuadPrecDType']


def SleefQuadPrecision(value):
    return QuadPrecision(value, backend='sleef')


def LongDoubleQuadPrecision(value):
    return QuadPrecision(value, backend='longdouble')


def SleefQuadPrecDType():
    return QuadPrecDType(backend='sleef')


def LongDoubleQuadPrecDType():
    return QuadPrecDType(backend='longdouble')
