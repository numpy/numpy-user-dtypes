# Working with Arrays

NumPy QuadDType integrates seamlessly with NumPy arrays, providing the full power of NumPy's array operations with quad precision arithmetic.

## Creating Arrays

### From Python Lists

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# Create an array from a list
arr = np.array([1.0, 2.0, 3.0], dtype=QuadPrecDType())
print(arr)
print(f"dtype: {arr.dtype}")
```

### From String Values (High Precision)

For maximum precision, create arrays from string representations:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# String input preserves all significant digits
high_precision = np.array([
    "3.14159265358979323846264338327950288",
    "2.71828182845904523536028747135266249",
    "1.41421356237309504880168872420969807"
], dtype=QuadPrecDType())

print(high_precision)
```

### Using `zeros`, `ones`, `empty`

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# Create arrays with standard NumPy functions
zeros = np.zeros(5, dtype=QuadPrecDType())
ones = np.ones((3, 3), dtype=QuadPrecDType())
empty = np.empty(10, dtype=QuadPrecDType())

print(f"Zeros shape: {zeros.shape}")
print(f"Ones shape: {ones.shape}")
```

### Using `arange` and `linspace`

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# Create ranges
arr = np.arange(0, 10, dtype=QuadPrecDType())
print(f"arange: {arr}")

# Linear spacing
lin = np.linspace(0, 1, 11, dtype=QuadPrecDType())
print(f"linspace: {lin}")
```

## Array Operations

### Element-wise Arithmetic

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

a = np.array([1, 2, 3], dtype=QuadPrecDType())
b = np.array([4, 5, 6], dtype=QuadPrecDType())

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
```

### Reductions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.array([1, 2, 3, 4, 5], dtype=QuadPrecDType())

print(f"Sum: {np.sum(arr)}")
print(f"Product: {np.prod(arr)}")
print(f"Mean: {np.mean(arr)}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
```

### Broadcasting

QuadPrecDType fully supports NumPy broadcasting:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# 2D array
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=QuadPrecDType())

# 1D array - broadcasts across rows
row_scale = np.array([10, 100, 1000], dtype=QuadPrecDType())

result = matrix * row_scale
print(result)
```

## Mathematical Functions

All standard NumPy ufuncs work with QuadPrecDType arrays:

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.linspace(0, 2 * np.pi, 5, dtype=QuadPrecDType())

# Trigonometric functions
print(f"sin(x): {np.sin(x)}")
print(f"cos(x): {np.cos(x)}")

# Exponential and logarithmic
y = np.array([1, 2, 3], dtype=QuadPrecDType())
print(f"exp(y): {np.exp(y)}")
print(f"log(y): {np.log(y)}")

# Square root
print(f"sqrt(y): {np.sqrt(y)}")
```

## Indexing and Slicing

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.arange(10, dtype=QuadPrecDType())

# Basic indexing
print(f"arr[0]: {arr[0]}")
print(f"arr[-1]: {arr[-1]}")

# Slicing
print(f"arr[2:5]: {arr[2:5]}")
print(f"arr[::2]: {arr[::2]}")

# Boolean indexing
mask = arr > 5
print(f"arr[arr > 5]: {arr[mask]}")
```

## Reshaping and Stacking

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.arange(12, dtype=QuadPrecDType())

# Reshape
reshaped = arr.reshape(3, 4)
print(f"Reshaped:\n{reshaped}")

# Stack arrays
a = np.array([1, 2, 3], dtype=QuadPrecDType())
b = np.array([4, 5, 6], dtype=QuadPrecDType())

stacked = np.stack([a, b])
print(f"Stacked:\n{stacked}")

concatenated = np.concatenate([a, b])
print(f"Concatenated: {concatenated}")
```

## Type Conversion

### Converting to QuadPrecDType

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# From float64
float64_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
quad_arr = float64_arr.astype(QuadPrecDType())

# From integer
int_arr = np.array([1, 2, 3], dtype=np.int64)
quad_from_int = int_arr.astype(QuadPrecDType())
```

### Converting from QuadPrecDType

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

quad_arr = np.array([1.5, 2.5, 3.5], dtype=QuadPrecDType())

# To float64 (loses precision)
float64_arr = quad_arr.astype(np.float64)
print(f"As float64: {float64_arr}")
```

## Memory Considerations

QuadPrecDType arrays use 16 bytes per element (compared to 8 bytes for float64):

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

n = 1000000

float64_arr = np.zeros(n, dtype=np.float64)
quad_arr = np.zeros(n, dtype=QuadPrecDType())

print(f"float64 size: {float64_arr.nbytes / 1e6:.1f} MB")
print(f"quad size: {quad_arr.nbytes / 1e6:.1f} MB")
```
