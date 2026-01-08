# Mathematical Constants

NumPy QuadDType provides pre-defined mathematical constants with full quad precision accuracy.

## Available Constants

```python
from numpy_quaddtype import (
    pi, e, log2e, log10e, ln2, ln10,
    max_value, epsilon, smallest_normal, smallest_subnormal,
    bits, precision, resolution
)
```

## Mathematical Constants

### π (Pi)

The ratio of a circle's circumference to its diameter.

```python
from numpy_quaddtype import pi
print(f"π = {pi}")
# 3.14159265358979323846264338327950288...
```

### e (Euler's Number)

The base of the natural logarithm.

```python
from numpy_quaddtype import e
print(f"e = {e}")
# 2.71828182845904523536028747135266249...
```

### Logarithmic Constants

```python
from numpy_quaddtype import log2e, log10e, ln2, ln10

print(f"log₂(e) = {log2e}")   # 1.44269504088896340735992468100189213...
print(f"log₁₀(e) = {log10e}") # 0.43429448190325182765112891891660508...
print(f"ln(2) = {ln2}")       # 0.69314718055994530941723212145817656...
print(f"ln(10) = {ln10}")     # 2.30258509299404568401799145468436420...
```

## Type Limits

### Machine Epsilon

The smallest positive number such that `1.0 + epsilon != 1.0`.

```python
from numpy_quaddtype import epsilon, QuadPrecision

print(f"ε = {epsilon}")

# Demonstration
one = QuadPrecision(1.0)
print(f"1 + ε == 1: {one + epsilon == one}")           # False
print(f"1 + ε/2 == 1: {one + epsilon/2 == one}")       # True
```

### Value Ranges

```python
from numpy_quaddtype import max_value, smallest_normal, smallest_subnormal

print(f"Maximum value:       {max_value}")
print(f"Smallest normal:     {smallest_normal}")
print(f"Smallest subnormal:  {smallest_subnormal}")
```

## Type Information

```python
from numpy_quaddtype import bits, precision, resolution

print(f"Total bits: {bits}")           # 128
print(f"Decimal precision: {precision}")  # 33-34 significant decimal digits
print(f"Resolution: {resolution}")     # Smallest distinguishable difference
```

## Using Constants in Calculations

```python
import numpy as np
from numpy_quaddtype import pi, e, QuadPrecDType

# Calculate e^(iπ) + 1 ≈ 0 (Euler's identity, real part)
# We'll compute cos(π) + 1 which should be 0
result = np.cos(np.array([pi]))[0] + 1
print(f"cos(π) + 1 = {result}")

# Area of a circle with radius 1
radius = np.array([1], dtype=QuadPrecDType())
area = pi * radius ** 2
print(f"Area of unit circle: {area[0]}")

# Natural exponential
x = np.array([1], dtype=QuadPrecDType())
exp_1 = np.exp(x)
print(f"e¹ = {exp_1[0]}")
print(f"e constant = {e}")
```

## Comparison with NumPy Constants

```python
import numpy as np
from numpy_quaddtype import pi as quad_pi, e as quad_e

print("Pi comparison:")
print(f"  NumPy float64: {np.pi}")
print(f"  QuadPrecision: {quad_pi}")

print("\ne comparison:")
print(f"  NumPy float64: {np.e}")
print(f"  QuadPrecision: {quad_e}")
```

The quad precision constants provide approximately 33-34 significant decimal digits, compared to 15-16 for float64.

## Constant Reference Table

| Constant | Symbol | Approximate Value |
|----------|--------|-------------------|
| `pi` | π | 3.14159265358979323846... |
| `e` | e | 2.71828182845904523536... |
| `log2e` | log₂(e) | 1.44269504088896340735... |
| `log10e` | log₁₀(e) | 0.43429448190325182765... |
| `ln2` | ln(2) | 0.69314718055994530941... |
| `ln10` | ln(10) | 2.30258509299404568401... |
| `epsilon` | ε | ~1.93×10⁻³⁴ |
| `max_value` | - | ~1.19×10⁴⁹³² |
| `smallest_normal` | - | ~3.36×10⁻⁴⁹³² |
| `bits` | - | 128 |
| `precision` | - | 33 |
