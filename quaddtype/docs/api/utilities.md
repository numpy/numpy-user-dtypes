# Utility Functions

Helper functions for platform precision detection and threading control.

## Platform Precision Detection

```{eval-rst}
.. function:: numpy_quaddtype.is_longdouble_128()

   Check if the platform's ``long double`` type is 128-bit.

   This is useful for determining whether the longdouble backend provides
   true quad precision on the current platform.

   :return: ``True`` if ``long double`` is 128-bit, ``False`` otherwise.
   :rtype: bool

   **Platform behavior:**

   - Linux x86_64: Returns ``False`` (80-bit extended precision)
   - Linux aarch64: Returns ``True`` (128-bit quad precision)
   - macOS (all): Returns ``False`` (64-bit double precision)
   - Windows (all): Returns ``False`` (64-bit double precision)

   **Example**

   ::

       >>> from numpy_quaddtype import is_longdouble_128
       >>> if is_longdouble_128():
       ...     print("Native quad precision available via longdouble")
       ... else:
       ...     print("Use SLEEF backend for quad precision")
```

## Threading Control

These functions control the number of threads used by QuadBLAS for parallel operations.

```{eval-rst}
.. function:: numpy_quaddtype.set_num_threads(n)

   Set the number of threads used by QuadBLAS operations.

   :param n: Number of threads to use. Must be a positive integer.
   :type n: int
   :raises ValueError: If n is not a positive integer.

   **Example**

   ::

       >>> from numpy_quaddtype import set_num_threads, get_num_threads
       >>> set_num_threads(4)
       >>> get_num_threads()
       4

   .. note::

      This function has no effect if QuadBLAS is disabled (e.g., on Windows).

.. function:: numpy_quaddtype.get_num_threads()

   Get the current number of threads used by QuadBLAS.

   :return: Current thread count for QuadBLAS operations.
   :rtype: int

   **Example**

   ::

       >>> from numpy_quaddtype import get_num_threads
       >>> get_num_threads()
       4

.. function:: numpy_quaddtype.get_quadblas_version()

   Get the QuadBLAS library version string.

   :return: Version string if QuadBLAS is available, ``None`` otherwise.
   :rtype: str or None

   **Example**

   ::

       >>> from numpy_quaddtype import get_quadblas_version
       >>> version = get_quadblas_version()
       >>> if version:
       ...     print(f"QuadBLAS version: {version}")
       ... else:
       ...     print("QuadBLAS not available")

   .. note::

      QuadBLAS is automatically disabled on Windows builds due to MSVC
      compatibility issues. In this case, the function returns ``None``.
```

## Example: Optimizing Thread Usage

```python
import numpy as np
from numpy_quaddtype import (
    QuadPrecDType, 
    set_num_threads, 
    get_num_threads,
    get_quadblas_version
)

# Check QuadBLAS availability
version = get_quadblas_version()
if version:
    print(f"QuadBLAS {version} available")

    # Get current threads
    print(f"Default threads: {get_num_threads()}")

    # Create test array
    arr = np.random.randn(100000).astype(QuadPrecDType())

    # Benchmark with different thread counts
    import time

    for threads in [1, 2, 4, 8]:
        set_num_threads(threads)

        start = time.time()
        for _ in range(10):
            result = np.dot(arr, arr)
        elapsed = time.time() - start

        print(f"  {threads} threads: {elapsed:.3f}s")
else:
    print("QuadBLAS not available - single-threaded operations only")
```
