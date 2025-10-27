from typing import Final

from ._quaddtype_main import (
    QuadPrecDType,
    QuadPrecision,
    _IntoQuad,  # type-check only  # pyright: ignore[reportPrivateUsage]
    get_num_threads,
    get_quadblas_version,
    is_longdouble_128,
    set_num_threads,
)

__all__ = [
    "QuadPrecision",
    "QuadPrecDType",
    "SleefQuadPrecision",
    "LongDoubleQuadPrecision",
    "SleefQuadPrecDType",
    "LongDoubleQuadPrecDType",
    "is_longdouble_128",
    "pi",
    "e",
    "log2e",
    "log10e",
    "ln2",
    "ln10",
    "max_value",
    "epsilon",
    "smallest_normal",
    "smallest_subnormal",
    "bits",
    "precision",
    "resolution",
    "set_num_threads",
    "get_num_threads",
    "get_quadblas_version",
]

__version__: Final[str] = ...

def SleefQuadPrecision(value: _IntoQuad) -> QuadPrecision: ...
def LongDoubleQuadPrecision(value: _IntoQuad) -> QuadPrecision: ...
def SleefQuadPrecDType() -> QuadPrecDType: ...
def LongDoubleQuadPrecDType() -> QuadPrecDType: ...

pi: Final[QuadPrecision] = ...
e: Final[QuadPrecision] = ...
log2e: Final[QuadPrecision] = ...
log10e: Final[QuadPrecision] = ...
ln2: Final[QuadPrecision] = ...
ln10: Final[QuadPrecision] = ...
max_value: Final[QuadPrecision] = ...
epsilon: Final[QuadPrecision] = ...
smallest_normal: Final[QuadPrecision] = ...
smallest_subnormal: Final[QuadPrecision] = ...
resolution: Final[QuadPrecision] = ...
bits: Final = 128
precision: Final = 33
