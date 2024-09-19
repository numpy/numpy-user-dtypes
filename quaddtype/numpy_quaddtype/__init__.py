from ._quaddtype_main import (
    QuadPrecision,
    QuadPrecDType,
    is_longdouble_128
)

__all__ = ['QuadPrecision', 'QuadPrecDType', 'SleefQuadPrecision', 'LongDoubleQuadPrecision',
           'SleefQuadPrecDType', 'LongDoubleQuadPrecDType', 'is_longdouble_128', 'pi', 'e']


pi = QuadPrecision.pi
e = QuadPrecision.e


def SleefQuadPrecision(value):
    return QuadPrecision(value, backend='sleef')


def LongDoubleQuadPrecision(value):
    return QuadPrecision(value, backend='longdouble')


def SleefQuadPrecDType():
    return QuadPrecDType(backend='sleef')


def LongDoubleQuadPrecDType():
    return QuadPrecDType(backend='longdouble')
