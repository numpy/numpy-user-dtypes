# Supported NumPy Functions

NumPy QuadDType supports a comprehensive set of NumPy universal functions (ufuncs) and array functions.

## Arithmetic Operations

### Binary Arithmetic

| Function | Operator | Description |
|----------|----------|-------------|
| `np.add` | `+` | Element-wise addition |
| `np.subtract` | `-` | Element-wise subtraction |
| `np.multiply` | `*` | Element-wise multiplication |
| `np.divide` | `/` | Element-wise division |
| `np.true_divide` | `/` | Element-wise true division |
| `np.floor_divide` | `//` | Element-wise floor division |
| `np.mod` | `%` | Element-wise modulo |
| `np.power` | `**` | Element-wise power |

### Unary Arithmetic

| Function | Operator | Description |
|----------|----------|-------------|
| `np.negative` | `-x` | Numerical negative |
| `np.positive` | `+x` | Numerical positive |
| `np.absolute` | `abs(x)` | Absolute value |
| `np.sign` | - | Sign indicator |

## Trigonometric Functions

### Standard Trigonometric

| Function | Description |
|----------|-------------|
| `np.sin` | Sine |
| `np.cos` | Cosine |
| `np.tan` | Tangent |

### Inverse Trigonometric

| Function | Description |
|----------|-------------|
| `np.arcsin` | Inverse sine |
| `np.arccos` | Inverse cosine |
| `np.arctan` | Inverse tangent |
| `np.arctan2` | Two-argument inverse tangent |

### Hyperbolic Functions

| Function | Description |
|----------|-------------|
| `np.sinh` | Hyperbolic sine |
| `np.cosh` | Hyperbolic cosine |
| `np.tanh` | Hyperbolic tangent |
| `np.arcsinh` | Inverse hyperbolic sine |
| `np.arccosh` | Inverse hyperbolic cosine |
| `np.arctanh` | Inverse hyperbolic tangent |

## Exponential and Logarithmic

### Exponential

| Function | Description |
|----------|-------------|
| `np.exp` | Exponential (e^x) |
| `np.exp2` | Base-2 exponential (2^x) |
| `np.expm1` | exp(x) - 1 (accurate for small x) |

### Logarithmic

| Function | Description |
|----------|-------------|
| `np.log` | Natural logarithm |
| `np.log2` | Base-2 logarithm |
| `np.log10` | Base-10 logarithm |
| `np.log1p` | log(1 + x) (accurate for small x) |

## Power and Root Functions

| Function | Description |
|----------|-------------|
| `np.sqrt` | Square root |
| `np.cbrt` | Cube root |
| `np.square` | Square (x²) |
| `np.hypot` | Hypotenuse (√(x² + y²)) |

## Comparison Functions

### Element-wise Comparison

| Function | Operator | Description |
|----------|----------|-------------|
| `np.equal` | `==` | Equal |
| `np.not_equal` | `!=` | Not equal |
| `np.less` | `<` | Less than |
| `np.less_equal` | `<=` | Less than or equal |
| `np.greater` | `>` | Greater than |
| `np.greater_equal` | `>=` | Greater than or equal |

### Min/Max

| Function | Description |
|----------|-------------|
| `np.minimum` | Element-wise minimum |
| `np.maximum` | Element-wise maximum |
| `np.fmin` | Element-wise minimum (ignores NaN) |
| `np.fmax` | Element-wise maximum (ignores NaN) |

## Rounding Functions

| Function | Description |
|----------|-------------|
| `np.floor` | Floor (round down) |
| `np.ceil` | Ceiling (round up) |
| `np.trunc` | Truncate toward zero |
| `np.rint` | Round to nearest integer |

## Special Value Functions

| Function | Description |
|----------|-------------|
| `np.isfinite` | Test for finite values |
| `np.isinf` | Test for infinity |
| `np.isnan` | Test for NaN |
| `np.signbit` | Test for negative sign bit |
| `np.copysign` | Copy sign of second to first |

## Reduction Functions

| Function | Description |
|----------|-------------|
| `np.sum` | Sum of elements |
| `np.prod` | Product of elements |
| `np.mean` | Arithmetic mean |
| `np.min` / `np.amin` | Minimum value |
| `np.max` / `np.amax` | Maximum value |

## Array Creation and Manipulation

### Creation

| Function | Description |
|----------|-------------|
| `np.zeros` | Array of zeros |
| `np.ones` | Array of ones |
| `np.empty` | Uninitialized array |
| `np.full` | Array filled with value |
| `np.arange` | Range of values |
| `np.linspace` | Linearly spaced values |

### Manipulation

| Function | Description |
|----------|-------------|
| `np.reshape` | Reshape array |
| `np.transpose` | Transpose array |
| `np.concatenate` | Join arrays |
| `np.stack` | Stack arrays |
| `np.split` | Split array |

## Linear Algebra (via QuadBLAS)

When QuadBLAS is available (not on Windows):

| Function | Description |
|----------|-------------|
| `np.dot` | Dot product |
| `np.matmul` / `@` | Matrix multiplication |

## Type Conversion

| Function | Description |
|----------|-------------|
| `np.astype` | Convert array dtype |
| `np.array` | Create array from data |

## Usage Examples

### Trigonometric

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType, pi

x = np.array([0, float(pi)/6, float(pi)/4, float(pi)/3, float(pi)/2], 
             dtype=QuadPrecDType())

print("sin(x):", np.sin(x))
print("cos(x):", np.cos(x))
```

### Exponential and Logarithmic

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

x = np.array([1, 2, 3], dtype=QuadPrecDType())

print("exp(x):", np.exp(x))
print("log(exp(x)):", np.log(np.exp(x)))  # Should return x
```

### Reductions

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

arr = np.arange(1, 11, dtype=QuadPrecDType())

print("Sum:", np.sum(arr))        # 55
print("Product:", np.prod(arr))   # 3628800 (10!)
print("Mean:", np.mean(arr))      # 5.5
```
