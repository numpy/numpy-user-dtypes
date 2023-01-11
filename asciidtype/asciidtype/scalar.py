"""A scalar type needed by the dtype machinery."""


class ASCIIScalar(str):
    def __new__(cls, value, dtype):
        instance = super().__new__(cls, value)
        instance.dtype = dtype
        return instance

    def partition(self, sep):
        ret = super().partition(sep)
        return (str(ret[0]), str(ret[1]), str(ret[2]))

    def rpartition(self, sep):
        ret = super().rpartition(sep)
        return (str(ret[0]), str(ret[1]), str(ret[2]))
