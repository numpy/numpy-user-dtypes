

class QuantityScalar:
    # Very basic scalar class, this would probably be done in C, or we would
    # create it even automatically (not yet).
    # Alternatively, there may be no scalar at all.
    def __init__(self, value, unit=None):
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"{self.value}*{self.unit}"

    def __mul__(self, other):
        # Hack a multiply for simpler examples (no checks...)
        return QuantityScalar(self.value * other.value, self.unit * other.unit)

    def __rmul__(self, other):
        if not isinstance(other, float) and not isinstance(other, int):
            raise ValueError("QuantityScalar only supports int and float rmul")
        # Hack a multiply for simpler examples (no checks...)
        return QuantityScalar(self.value * other, self.unit)
