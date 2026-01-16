# Supported NumPy Functions

NumPy QuadDType supports a comprehensive set of NumPy universal functions (ufuncs) and array functions.

## Element-wise Arithmetic Operations

### Unary Arithmetic

| Function | Operator | Description |
|----------|----------|-------------|
| `np.negative` | `-x` | Numerical negative |
| `np.positive` | `+x` | Numerical positive |
| `np.absolute` | `abs(x)` | Absolute value |

### Binary Arithmetic

| Function | Operator | Description |
|----------|----------|-------------|
| `np.add` | `+` | Addition |
| `np.subtract` | `-` | Subtraction |
| `np.multiply` | `*` | Multiplication |
| `np.divide` | `/` | Division |
| `np.true_divide` | `/` | True division |
| `np.floor_divide` | `//` | Floor division |
| `np.mod` | `%` | Modulo |
| `np.power` | `**` | Power |

## Element-wise Sign Functions

| Function | Description |
|----------|-------------|
| `np.sign` | Sign indicator |
| `np.signbit` | Test for negative sign bit (works with NaN) |
| `np.copysign` | Copy sign of second to first |

## Element-wise Trigonometric Functions

| Function | Description |
|----------|-------------|
| `np.sin` | Sine |
| `np.cos` | Cosine |
| `np.tan` | Tangent |
| `np.arcsin` | Inverse sine |
| `np.arccos` | Inverse cosine |
| `np.arctan` | Inverse tangent |
| `np.arctan2` | Two-argument inverse tangent |

### Element-wise Hyperbolic Functions

| Function | Description |
|----------|-------------|
| `np.sinh` | Hyperbolic sine |
| `np.cosh` | Hyperbolic cosine |
| `np.tanh` | Hyperbolic tangent |
| `np.arcsinh` | Inverse hyperbolic sine |
| `np.arccosh` | Inverse hyperbolic cosine |
| `np.arctanh` | Inverse hyperbolic tangent |

## Element-wise Exponential Functions

| Function | Description |
|----------|-------------|
| `np.exp` | Exponential ({math}`e^x`) |
| `np.exp2` | Base-2 exponential ({math}`2^x`) |
| `np.expm1` | `exp(x) - 1` (accurate for small x) |

## Element-wise Logarithmic Functions

| Function | Description |
|----------|-------------|
| `np.log` | Natural logarithm |
| `np.log2` | Base-2 logarithm |
| `np.log10` | Base-10 logarithm |
| `np.log1p` | `log(1 + x)` (accurate for small x) |

## Element-wise Power and Root Functions

| Function | Description |
|----------|-------------|
| `np.square` | Square ({math}`x^2`) |
| `np.sqrt` | Square root |
| `np.cbrt` | Cube root |
| `np.hypot` | Hypotenuse ({math}`\sqrt{x^2 + y^2}`) |

## Element-wise Rounding Functions

| Function | Description |
|----------|-------------|
| `np.floor` | Floor (round down) |
| `np.ceil` | Ceiling (round up) |
| `np.trunc` | Truncate toward zero |
| `np.rint` | Round to nearest integer (ties to even) |

## Element-wise Classification Functions

| Function | Description |
|----------|-------------|
| `np.isfinite` | Test for finite values |
| `np.isinf` | Test for infinity |
| `np.isnan` | Test for NaN |

## Element-wise Comparison Functions

| Function | Operator | Description |
|----------|----------|-------------|
| `np.equal` | `==` | Equal |
| `np.not_equal` | `!=` | Not equal |
| `np.less` | `<` | Less than |
| `np.less_equal` | `<=` | Less than or equal |
| `np.greater` | `>` | Greater than |
| `np.greater_equal` | `>=` | Greater than or equal |

### Element-wise Minimum/Maximum

| Function | Description |
|----------|-------------|
| `np.minimum` | Minimum |
| `np.maximum` | Maximum |
| `np.fmin` | Minimum (ignores NaN) |
| `np.fmax` | Maximum (ignores NaN) |

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
| `np.full` | Array filled with given value |
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

x = np.array([0, pi/6, pi/4, pi/3, pi/2, pi], dtype=QuadPrecDType())

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
