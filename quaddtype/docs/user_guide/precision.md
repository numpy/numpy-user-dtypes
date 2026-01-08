# Understanding Quad Precision

## What is Quad Precision?

Quad precision (also known as quadruple precision or binary128) is a floating-point format defined by the IEEE 754 standard. It provides significantly higher precision than the commonly used double precision (float64).

## Precision Comparison

| Format | Bits | Sign | Exponent | Mantissa | Decimal Digits |
|--------|------|------|----------|----------|----------------|
| Single (float32) | 32 | 1 | 8 | 23 | ~7 |
| Double (float64) | 64 | 1 | 11 | 52 | ~15-16 |
| **Quad (float128)** | **128** | **1** | **15** | **112** | **~33-34** |

## Demonstrating the Precision Difference

```python
import numpy as np
from numpy_quaddtype import QuadPrecision, pi

# Standard double precision π
pi_float64 = np.float64(np.pi)
print(f"float64 π: {pi_float64}")

# Quad precision π
print(f"quad    π: {pi}")

# The actual value of π to 50 decimal places:
# 3.14159265358979323846264338327950288419716939937510...
```

### Practical Example: Computing e

Let's compute Euler's number using the series expansion $e = \sum_{n=0}^{\infty} \frac{1}{n!}$:

```python
import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType

def compute_e_quad(terms=50):
    """Compute e using quad precision."""
    result = QuadPrecision(0)
    factorial = QuadPrecision(1)
    
    for n in range(terms):
        if n > 0:
            factorial = factorial * n
        result = result + QuadPrecision(1) / factorial
    
    return result

def compute_e_float64(terms=50):
    """Compute e using float64."""
    result = np.float64(0)
    factorial = np.float64(1)
    
    for n in range(terms):
        if n > 0:
            factorial = factorial * n
        result = result + 1.0 / factorial
    
    return result

e_quad = compute_e_quad(30)
e_float64 = compute_e_float64(30)

print(f"e (quad):    {e_quad}")
print(f"e (float64): {e_float64}")
```

## When to Use Quad Precision

### ✅ Good Use Cases

1. **Ill-conditioned Problems**: When numerical instability affects results
2. **Reference Implementations**: Validating lower-precision algorithms
3. **Financial Calculations**: When regulatory compliance requires high precision
4. **Scientific Research**: Astronomy, physics simulations
5. **Cryptographic Applications**: Where precision is critical

### ⚠️ Consider Alternatives

1. **Performance-Critical Code**: Quad precision is slower than float64
2. **Large Datasets**: Memory usage is 2x compared to float64
3. **Simple Calculations**: When float64 precision is sufficient

## Memory Layout

QuadPrecision values are stored as 128-bit (16 bytes) values in memory:

```
┌─────────┬─────────────────┬────────────────────────────────────────────────────┐
│  Sign   │    Exponent     │                     Mantissa                        │
│  1 bit  │    15 bits      │                    112 bits                         │
└─────────┴─────────────────┴────────────────────────────────────────────────────┘
```

## Special Values

QuadPrecision supports all IEEE 754 special values:

```python
from numpy_quaddtype import QuadPrecision
import numpy as np

# Infinity
pos_inf = QuadPrecision("inf")
neg_inf = QuadPrecision("-inf")

# NaN (Not a Number)
nan = QuadPrecision("nan")

# Check special values
print(f"Is inf: {np.isinf(pos_inf)}")
print(f"Is nan: {np.isnan(nan)}")
print(f"Is finite: {np.isfinite(QuadPrecision(1.0))}")
```

## Precision Limits

```python
from numpy_quaddtype import epsilon, smallest_normal, smallest_subnormal, max_value

print(f"Machine epsilon:      {epsilon}")
print(f"Smallest normal:      {smallest_normal}")
print(f"Smallest subnormal:   {smallest_subnormal}")
print(f"Maximum value:        {max_value}")
```
