"""A scalar type needed by the dtype machinery."""


class ASCIIScalar(str):
    def __new__(cls, value, dtype):
        instance = super().__new__(cls, value)
        instance.dtype = dtype
        return instance
