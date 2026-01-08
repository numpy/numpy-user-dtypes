# NumPy QuadDType

```{image} https://img.shields.io/pypi/v/numpy-quaddtype.svg
:target: https://pypi.org/project/numpy-quaddtype/
:alt: PyPI version
```
```{image} https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg
:alt: Python versions
```

**A cross-platform 128-bit (quadruple precision) floating-point data type for NumPy.**

NumPy QuadDType provides IEEE 754 quadruple-precision (binary128) floating-point arithmetic as a first-class NumPy dtype, enabling high-precision numerical computations that go beyond the standard 64-bit double precision.

## Key Features

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} 🎯 True Quad Precision
:link: user_guide/precision
:link-type: doc

128-bit floating point with ~34 decimal digits of precision
:::

:::{grid-item-card} 🔌 NumPy Integration
:link: user_guide/arrays
:link-type: doc

Works seamlessly with NumPy arrays, ufuncs, and broadcasting.
:::

:::{grid-item-card} ⚡ SIMD Optimized
:link: user_guide/performance
:link-type: doc

Vectorization-friendly design that can leverage SIMD acceleration where supported.
:::

:::{grid-item-card} 🧮 Mathematical Functions
:link: api/functions
:link-type: doc

Full suite of math functions: trigonometric, exponential, logarithmic, and more.
:::

:::{grid-item-card} 🔀 Dual Backend
:link: user_guide/backends
:link-type: doc

Choose between SLEEF (default) or longdouble backends.
:::

:::{grid-item-card} 🧵 Thread-Safe
:link: user_guide/threading
:link-type: doc

Full support for Python's free-threading (GIL-free) mode.
:::

::::

## Quick Start

### Installation

```bash
pip install numpy-quaddtype
```

### Basic Usage

```python
import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType

# Create a quad-precision scalar
x = QuadPrecision("3.14159265358979323846264338327950288")

# Create a quad-precision array
arr = np.array([1, 2, 3], dtype=QuadPrecDType())

# Use NumPy functions
result = np.sin(arr)
print(result)
```

### Why Quad Precision?

Standard double precision (float64) provides approximately 15-16 significant decimal digits. While sufficient for most applications, some scenarios require higher precision:

- **Numerical Analysis**: Ill-conditioned problems, iterative algorithms
- **Scientific Computing**: Astronomy, physics simulations requiring extreme accuracy
- **Financial Calculations**: High-precision arithmetic for regulatory compliance
- **Validation**: Checking accuracy of lower-precision implementations

```{toctree}
:maxdepth: 2
:hidden:

installation
user_guide/index
api/index
contributing
changelog
```

## Indices and tables

- {ref}`genindex`
- {ref}`search`
