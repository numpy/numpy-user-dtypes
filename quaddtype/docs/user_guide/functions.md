# Mathematical Functions

NumPy QuadDType provides a comprehensive set of mathematical functions through NumPy's universal function (ufunc) system. All functions work seamlessly with both scalars and arrays.

## Basic Arithmetic

### Binary Operations

| Operation | Operator | NumPy Function |
|-----------|----------|----------------|
| Addition | `a + b` | `np.add(a, b)` |
| Subtraction | `a - b` | `np.subtract(a, b)` |
| Multiplication | `a * b` | `np.multiply(a, b)` |
| Division | `a / b` | `np.divide(a, b)` |
| Floor Division | `a // b` | `np.floor_divide(a, b)` |
| Modulo | `a % b` | `np.mod(a, b)` |
| Power | `a ** b` | `np.power(a, b)` |

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

a = np.array([1, 2, 3], dtype=QuadPrecDType())
b = np.array([4, 5, 6], dtype=QuadPrecDType())

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
```

### Unary Operations

| Operation | Operator | NumPy Function |
|-----------|----------|----------------|
| Negation | `-a` | `np.negative(a)` |
| Absolute | `abs(a)` | `np.abs(a)` |
| Positive | `+a` | `np.positive(a)` |

## Trigonometric Functions

### Standard Trigonometric

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType, pi

x = np.linspace(0, float(pi)/2, 5, dtype=QuadPrecDType())

# Basic trig functions
print(f"sin(x): {np.sin(x)}")
print(f"cos(x): {np.cos(x)}")
print(f"tan(x): {np.tan(x)}")
```

### Inverse Trigonometric

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([0, 0.5, 1.0], dtype=QuadPrecDType())

print(f"arcsin(x): {np.arcsin(x)}")
print(f"arccos(x): {np.arccos(x)}")

y = np.array([0, 1, 10], dtype=QuadPrecDType())
print(f"arctan(y): {np.arctan(y)}")
```

### Two-Argument Arctangent

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

y = np.array([1, 1, -1, -1], dtype=QuadPrecDType())
x = np.array([1, -1, 1, -1], dtype=QuadPrecDType())

# atan2 gives the angle in the correct quadrant
angles = np.arctan2(y, x)
print(f"arctan2(y, x): {angles}")
```

## Hyperbolic Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([0, 0.5, 1.0, 2.0], dtype=QuadPrecDType())

# Hyperbolic functions
print(f"sinh(x): {np.sinh(x)}")
print(f"cosh(x): {np.cosh(x)}")
print(f"tanh(x): {np.tanh(x)}")

# Inverse hyperbolic
print(f"arcsinh(x): {np.arcsinh(x)}")
print(f"arccosh(x+1): {np.arccosh(x + 1)}")  # arccosh requires x >= 1
print(f"arctanh(x/3): {np.arctanh(x / 3)}")  # arctanh requires |x| < 1
```

## Exponential and Logarithmic

### Exponential Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([0, 1, 2, 3], dtype=QuadPrecDType())

print(f"exp(x): {np.exp(x)}")
print(f"exp2(x): {np.exp2(x)}")        # 2^x
print(f"expm1(x): {np.expm1(x)}")      # exp(x) - 1, accurate for small x
```

### Logarithmic Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([1, 2, 10, 100], dtype=QuadPrecDType())

print(f"log(x): {np.log(x)}")          # Natural log
print(f"log2(x): {np.log2(x)}")        # Base-2 log
print(f"log10(x): {np.log10(x)}")      # Base-10 log
print(f"log1p(x-1): {np.log1p(x - 1)}")  # log(1+x), accurate for small x
```

## Power and Root Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([1, 4, 9, 16], dtype=QuadPrecDType())

print(f"sqrt(x): {np.sqrt(x)}")
print(f"cbrt(x): {np.cbrt(x)}")        # Cube root

# Hypotenuse (sqrt(a^2 + b^2))
a = np.array([3, 5, 8], dtype=QuadPrecDType())
b = np.array([4, 12, 15], dtype=QuadPrecDType())
print(f"hypot(a, b): {np.hypot(a, b)}")
```

## Rounding Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([1.2, 2.5, 3.7, -1.5], dtype=QuadPrecDType())

print(f"floor(x): {np.floor(x)}")
print(f"ceil(x): {np.ceil(x)}")
print(f"trunc(x): {np.trunc(x)}")
print(f"rint(x): {np.rint(x)}")  # Round to nearest integer
```

## Comparison Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

a = np.array([1, 5, 3], dtype=QuadPrecDType())
b = np.array([2, 4, 3], dtype=QuadPrecDType())

print(f"minimum(a, b): {np.minimum(a, b)}")
print(f"maximum(a, b): {np.maximum(a, b)}")

# Comparison operators
print(f"a < b: {a < b}")
print(f"a == b: {a == b}")
print(f"a >= b: {a >= b}")
```

## Special Value Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType

# Create array with special values
arr = np.array([
    QuadPrecision(1.0),
    QuadPrecision("inf"),
    QuadPrecision("-inf"),
    QuadPrecision("nan")
])

print(f"isfinite: {np.isfinite(arr)}")
print(f"isinf: {np.isinf(arr)}")
print(f"isnan: {np.isnan(arr)}")
```

## Sign and Absolute Value

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([-3, -1, 0, 1, 3], dtype=QuadPrecDType())

print(f"abs(x): {np.abs(x)}")
print(f"sign(x): {np.sign(x)}")
print(f"copysign(1, x): {np.copysign(1, x)}")
```

## Reduction Functions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.array([1, 2, 3, 4, 5], dtype=QuadPrecDType())

print(f"sum: {np.sum(arr)}")
print(f"prod: {np.prod(arr)}")
print(f"mean: {np.mean(arr)}")
print(f"min: {np.min(arr)}")
print(f"max: {np.max(arr)}")
```

## Precision Demonstration

The advantage of quad precision is evident in calculations that lose precision in float64:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# Computing 1 - cos(x) for small x loses precision in float64
x_small = 1e-8

# Float64
result_f64 = 1 - np.cos(np.float64(x_small))
print(f"1 - cos(1e-8) [float64]: {result_f64}")

# Quad precision
x_quad = np.array([x_small], dtype=QuadPrecDType())
result_quad = 1 - np.cos(x_quad)
print(f"1 - cos(1e-8) [quad]:    {result_quad[0]}")

# Theoretical value: x^2/2 ≈ 5e-17
print(f"Theoretical (x²/2):      5e-17")
```
