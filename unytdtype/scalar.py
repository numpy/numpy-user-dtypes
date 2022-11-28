"""A scalar type needed by the dtype machinery."""

from unyt import Unit

class UnytScalar:
    def __init__(self, value, unit):
        self.value = value
        if isinstance(unit, str):
            self.unit = Unit(unit)
        elif isinstance(unit, Unit):
            self.unit = unit
        else:
            raise RuntimeError

    def __repr__(self):
        return f"{self.value} {self.unit}"

    def __rmul__(self, other):
        return UnytScalar(self.value * other, self.unit)
