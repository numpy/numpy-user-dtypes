"""A scalar type needed by the dtype machinery."""

from unyt import Unit


class UnytScalar:
    def __init__(self, value, unit):
        from . import UnytDType
        self.value = value
        if isinstance(unit, (str, Unit)):
            self.dtype = UnytDType(unit)
        elif isinstance(unit, UnytDType):
            self.dtype = unit
        else:
            raise RuntimeError

    @property
    def unit(self):
        return self.dtype.unit

    def __repr__(self):
        return f"{self.value} {self.dtype.unit}"

    def __rmul__(self, other):
        return UnytScalar(self.value * other, self.dtype.unit)

    def __eq__(self, other):
        return self.value == other.value and self.dtype == other.dtype
