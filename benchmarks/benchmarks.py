# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import uuid

import numpy as np

from asciidtype import ASCIIDType
from strptrdtype import StrPtrDType


def generate_data():
    n = 100000
    strings_list = [str(uuid.uuid4()) + '\n' for i in range(n)]

    with open('strings', 'w') as f:
        f.writelines(strings_list)


class TimeASCIIDType:
    def setup(self):
        self.ascii_dtype_object = ASCIIDType(36)
        with open('strings', 'r') as f:
            self.strings = f.readlines()

    def time_allocate(self):
        _ = np.array(self.strings, dtype=self.ascii_dtype_object)


class TimeStrPtrDType:
    def setup(self):
        self.strptr_dtype_object = StrPtrDType()
        with open('strings', 'rb') as f:
            self.bytestrings = f.readlines()

    def time_allocate(self):
        _ = np.array(self.bytestrings, dtype=self.strptr_dtype_object)


if __name__ == "__main__":
    strptr_instance = TimeStrPtrDType()
    strptr_instance.setup()

    # ascii_instance = TimeASCIIDType()
    # ascii_instance.setup()

    strptr_instance.time_allocate()
    # ascii_instance.time_allocate()
