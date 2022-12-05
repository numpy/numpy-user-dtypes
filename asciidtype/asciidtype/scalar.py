"""A scalar type needed by the dtype machinery."""


class ASCIIScalar:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def __repr__(self):
        return f"{self.value}"
