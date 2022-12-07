"""A scalar type needed by the dtype machinery."""


class ASCIIScalar:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)
