from ._quaddtype_main import (
    QuadPrecision,
    QuadPrecDType,
    is_longdouble_128,
    pi, e, log2e, log10e, ln2, ln10,
    sqrt2, sqrt3, egamma, phi, quad_max, quad_min, quad_epsilon, quad_denorm_min
)

__all__ = [
    'QuadPrecision', 'QuadPrecDType', 'SleefQuadPrecision', 'LongDoubleQuadPrecision',
    'SleefQuadPrecDType', 'LongDoubleQuadPrecDType', 'is_longdouble_128',
    'pi', 'e', 'log2e', 'log10e', 'ln2', 'ln10',
    'sqrt2', 'sqrt3', 'egamma', 'phi',
    'quad_max', 'quad_min', 'quad_epsilon', 'quad_denorm_min'
]

def SleefQuadPrecision(value):
    return QuadPrecision(value, backend='sleef')

def LongDoubleQuadPrecision(value):
    return QuadPrecision(value, backend='longdouble')

def SleefQuadPrecDType():
    return QuadPrecDType(backend='sleef')

def LongDoubleQuadPrecDType():
    return QuadPrecDType(backend='longdouble')