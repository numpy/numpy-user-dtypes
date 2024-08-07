import pytest

import numpy as np
from quaddtype import QuadPrecDType, QuadPrecision


def test():
    a = QuadPrecision("1.63")
    assert f"{np.array([a], dtype=QuadPrecDType).dtype}" == "QuadPrecDType()"
