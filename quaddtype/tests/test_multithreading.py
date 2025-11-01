import concurrent.futures
import threading

import pytest

import numpy as np
from numpy._core import _rational_tests
from numpy._core.tests.test_stringdtype import random_unicode_string_list
from numpy.testing import IS_64BIT, IS_WASM
from numpy.testing._private.utils import run_threaded

if IS_WASM:
    pytest.skip(allow_module_level=True, reason="no threading support in wasm")

from numpy_quaddtype import *


def test_as_integer_ratio_reconstruction():
    """Multi-threaded test that as_integer_ratio() can reconstruct the original value."""

    def test(barrier):
      barrier.wait() # All threads start simultaneously
      values = ["3.14", "0.1", "1.414213562373095", "2.718281828459045",
      "-1.23456789", "1000.001", "0.0001", "1e20", "1.23e15", "1e-30", pi]
      for val in values:
        quad_val = QuadPrecision(val)
        num, denom = quad_val.as_integer_ratio()
        # todo: can remove str converstion after merging PR #213
        reconstructed = QuadPrecision(str(num)) / QuadPrecision(str(denom))
        assert reconstructed == quad_val
    
    run_threaded(test, pass_barrier=True, max_workers=64, outer_iterations=100)