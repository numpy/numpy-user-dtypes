import numpy as np
from numpy.testing import assert_array_equal

from mpfdtype import MPFDType


def test_advanced_indexing():
    # As of writing the test, this relies on copyswap
    arr = np.arange(100).astype(MPFDType(100))
    orig = np.arange(100).astype(MPFDType(100))  # second one, not a copy

    b = arr[[1, 2, 3, 4]]
    b[...] = 5  # does not mutate arr (internal references not broken)
    assert_array_equal(arr, orig)


def test_is_numeric():
    assert MPFDType._is_numeric
