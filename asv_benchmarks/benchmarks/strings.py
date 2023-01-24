# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import uuid

import numpy as np

from asciidtype import ASCIIDType
from stringdtype import StringDType


def generate_data(n=100000):
    """Generate data for the benchmarks.

    The vast majority of the time spent benchmarking is generating data; generate it
    once and store it to avoid having to do this every run.
    """
    strings_list = [str(uuid.uuid4()) + "\n" for i in range(n)]

    with open("strings", "w") as f:
        f.writelines(strings_list)


class TimeASCIIDType:
    def setup(self):
        self.ascii_dtype_object = ASCIIDType(36)
        with open("strings", "r") as f:
            self.strings = f.readlines()

    def time_allocate(self):
        _ = np.array(self.strings, dtype=self.ascii_dtype_object)


class TimeStringDType:
    def setup(self):
        self.string_dtype_object = StringDType()
        with open("strings", "r") as f:
            self.strings = f.readlines()

    def time_allocate(self):
        _ = np.array(self.strings, dtype=self.string_dtype_object)


class TimeObjectDType:
    def setup(self):
        with open("strings", "r") as f:
            self.strings = f.readlines()

    def time_allocate(self):
        _ = np.array(self.strings, dtype=object)
