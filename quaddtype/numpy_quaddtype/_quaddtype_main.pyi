from typing import Any, Literal, TypeAlias, final, overload

import numpy as np
from numpy._typing import _128Bit  # pyright: ignore[reportPrivateUsage]
from typing_extensions import Never, Self, override

_Backend: TypeAlias = Literal["sleef", "longdouble"]
_IntoQuad: TypeAlias = (
    QuadPrecision
    | float
    | str
    | np.floating[Any]
    | np.integer[Any]
    | np.bool_
)  # fmt: skip
_ScalarItemArg: TypeAlias = Literal[0, -1] | tuple[Literal[0, -1]] | tuple[()]

@final
class QuadPrecDType(np.dtype[QuadPrecision]):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    def __new__(cls, /, backend: _Backend = "sleef") -> Self: ...

    # `numpy.dtype` overrides
    names: None  # pyright: ignore[reportIncompatibleVariableOverride]
    @property
    @override
    def alignment(self) -> Literal[16]: ...
    @property
    @override
    def itemsize(self) -> Literal[16]: ...
    @property
    @override
    def name(self) -> Literal["QuadPrecDType128"]: ...
    @property
    @override
    def byteorder(self) -> Literal["|"]: ...
    @property
    @override
    def char(self) -> Literal["\x00"]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @property
    @override
    def kind(self) -> Literal["\x00"]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @property
    @override
    def num(self) -> Literal[-1]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @property
    @override
    def shape(self) -> tuple[()]: ...
    @property
    @override
    def ndim(self) -> Literal[0]: ...
    @property
    @override
    def fields(self) -> None: ...
    @property
    @override
    def base(self) -> Self: ...
    @property
    @override
    def subdtype(self) -> None: ...
    @property
    @override
    def hasobject(self) -> Literal[False]: ...
    @property
    @override
    def isbuiltin(self) -> Literal[0]: ...
    @property
    @override
    def isnative(self) -> Literal[True]: ...
    @property
    @override
    def isalignedstruct(self) -> Literal[False]: ...
    @override
    def __getitem__(self, key: Never, /) -> Self: ...  # type: ignore[override]

@final
class QuadPrecision(np.floating[_128Bit]):
    # NOTE: At runtime this constructor also accepts array-likes, for which it returns
    # `np.ndarray` instances with `dtype=QuadPrecDType()`.
    # But because of mypy limitations, it is currently impossible to annotate
    # constructors that do not return instances of their class (or a subclass thereof).
    # See https://github.com/python/mypy/issues/18343#issuecomment-2571784915
    @override
    def __new__(cls, /, value: _IntoQuad, backend: _Backend = "sleef") -> Self: ...

    # numpy.floating property overrides

    @property
    @override
    def dtype(self) -> QuadPrecDType: ...
    @property
    @override
    def real(self) -> Self: ...
    @property
    @override
    def imag(self) -> Self: ...

    # numpy.floating method overrides

    @override
    def item(self, arg0: _ScalarItemArg = ..., /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def tolist(self, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    # Equality operators
    @override
    def __eq__(self, other: object, /) -> bool: ...
    @override
    def __ne__(self, other: object, /) -> bool: ...

    # Rich comparison operators
    # NOTE: Unlike other numpy scalars, these return `builtins.bool`, not `np.bool`.
    @override
    def __lt__(self, other: _IntoQuad, /) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __le__(self, other: _IntoQuad, /) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __gt__(self, other: _IntoQuad, /) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ge__(self, other: _IntoQuad, /) -> bool: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    # Binary arithmetic operators
    @override
    def __add__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __radd__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __sub__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rsub__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __mul__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rmul__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __pow__(self, other: _IntoQuad, mod: None = None, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rpow__(self, other: _IntoQuad, mod: None = None, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __truediv__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rtruediv__(self, other: _IntoQuad, /) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

    # Binary modulo operators
    @override
    def __floordiv__(self, other: _IntoQuad, /) -> Self: ...
    @override
    def __rfloordiv__(self, other: _IntoQuad, /) -> Self: ...
    @override
    def __mod__(self, other: _IntoQuad, /) -> Self: ...
    @override
    def __rmod__(self, other: _IntoQuad, /) -> Self: ...
    @override
    def __divmod__(self, other: _IntoQuad, /) -> tuple[Self, Self]: ...
    @override
    def __rdivmod__(self, other: _IntoQuad, /) -> tuple[Self, Self]: ...

    # NOTE: is_integer() and as_integer_ratio() are defined on numpy.floating in the
    # stubs, but don't exist at runtime. And because QuadPrecision does not implement
    # them, we use this hacky workaround to emulate their absence.
    # TODO: Remove after https://github.com/numpy/numpy-user-dtypes/issues/216
    is_integer: Never  # pyright: ignore[reportIncompatibleMethodOverride]
    as_integer_ratio: Never  # pyright: ignore[reportIncompatibleMethodOverride]

#
def is_longdouble_128() -> bool: ...

@overload
def get_sleef_constant(constant_name: Literal["bits", "precision"], /) -> int: ...
@overload
def get_sleef_constant(
    constant_name: Literal[
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
        "resolution",
    ],
    /,
) -> QuadPrecision: ...

def set_num_threads(num_threads: int, /) -> None: ...
def get_num_threads() -> int: ...
def get_quadblas_version() -> str: ...
