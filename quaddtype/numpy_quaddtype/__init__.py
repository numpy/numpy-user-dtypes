from ._quaddtype_main import (
    QuadPrecision,
    QuadPrecDType,
    is_longdouble_128,
    get_sleef_constant,
    set_num_threads,
    get_num_threads,
    get_quadblas_version
)

__all__ = [
    'QuadPrecision', 'QuadPrecDType', 'SleefQuadPrecision', 'LongDoubleQuadPrecision',
    'SleefQuadPrecDType', 'LongDoubleQuadPrecDType', 'is_longdouble_128', 
    # Constants
    'pi', 'e', 'log2e', 'log10e', 'ln2', 'ln10', 'max_value', 'min_value', 'epsilon', 
    # QuadBLAS related functions
    'set_num_threads', 'get_num_threads', 'get_quadblas_version'
]

def SleefQuadPrecision(value):
    return QuadPrecision(value, backend='sleef')

def LongDoubleQuadPrecision(value):
    return QuadPrecision(value, backend='longdouble')

def SleefQuadPrecDType():
    return QuadPrecDType(backend='sleef')

def LongDoubleQuadPrecDType():
    return QuadPrecDType(backend='longdouble')

pi = get_sleef_constant("pi")
e = get_sleef_constant("e")
log2e = get_sleef_constant("log2e")
log10e = get_sleef_constant("log10e")
ln2 = get_sleef_constant("ln2")
ln10 = get_sleef_constant("ln10")
max_value = get_sleef_constant("quad_max")
min_value = get_sleef_constant("quad_min")
epsilon = get_sleef_constant("epsilon")