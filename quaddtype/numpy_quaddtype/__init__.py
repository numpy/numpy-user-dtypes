from ._quaddtype_main import (
    QuadPrecision,
    QuadPrecDType,
    is_longdouble_128,
    get_sleef_constant,
    set_num_threads,
    get_num_threads,
    get_quadblas_version
)

from dataclasses import dataclass, field

__all__ = [
    'QuadPrecision', 'QuadPrecDType', 'SleefQuadPrecision', 'LongDoubleQuadPrecision',
    'SleefQuadPrecDType', 'LongDoubleQuadPrecDType', 'is_longdouble_128', 
    # Constants
    'pi', 'e', 'log2e', 'log10e', 'ln2', 'ln10', 'max_value', 'epsilon',
    'smallest_normal', 'smallest_subnormal', 'bits', 'precision', 'resolution',
    # QuadBLAS related functions
    'set_num_threads', 'get_num_threads', 'get_quadblas_version',
    # finfo class
    'QuadPrecFinfo'
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
max_value = get_sleef_constant("max_value")
epsilon = get_sleef_constant("epsilon")
smallest_normal = get_sleef_constant("smallest_normal")
smallest_subnormal = get_sleef_constant("smallest_subnormal")
bits = get_sleef_constant("bits")
precision = get_sleef_constant("precision")
resolution = get_sleef_constant("resolution")

@dataclass
class QuadPrecFinfo:
    """Floating-point information for quadruple precision dtype.
    
    This class provides information about the floating-point representation
    used by the QuadPrecDType, similar to numpy.finfo but customized for
    quad precision arithmetic.
    """
    bits: int = field(default_factory=lambda: bits)
    eps: float = field(default_factory=lambda: epsilon)
    epsneg: float = field(default_factory=lambda: epsilon)
    iexp: int = field(default_factory=lambda: precision)
    machar: object = None
    machep: float = field(default_factory=lambda: epsilon)
    max: float = field(default_factory=lambda: max_value)
    maxexp: float = field(default_factory=lambda: max_value)
    min: float = field(default_factory=lambda: smallest_normal)
    minexp: float = field(default_factory=lambda: smallest_normal)
    negep: float = field(default_factory=lambda: epsilon)
    nexp: int = field(default_factory=lambda: bits - precision - 1)
    nmant: int = field(default_factory=lambda: precision)
    precision: int = field(default_factory=lambda: precision)
    resolution: float = field(default_factory=lambda: resolution)
    tiny: float = field(default_factory=lambda: smallest_normal)
    smallest_normal: float = field(default_factory=lambda: smallest_normal)
    smallest_subnormal: float = field(default_factory=lambda: smallest_subnormal)

    def get(self, attr):
        return getattr(self, attr, None)
    
    def __str__(self):
        return f"QuadPrecFinfo(precision={self.precision}, resolution={self.resolution})"
    
    def __repr__(self):
        return f"QuadPrecFinfo(max={self.max}, min={self.min}, eps={self.eps}, bits={self.bits})"