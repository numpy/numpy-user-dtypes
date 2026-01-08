# Constants Reference

Pre-defined mathematical constants with quad precision accuracy.

## Mathematical Constants

```{eval-rst}
.. data:: numpy_quaddtype.pi

   The mathematical constant π (pi).
   
   Value: 3.14159265358979323846264338327950288...
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.e

   Euler's number, the base of natural logarithms.
   
   Value: 2.71828182845904523536028747135266249...
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.log2e

   The base-2 logarithm of e: log₂(e).
   
   Value: 1.44269504088896340735992468100189213...
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.log10e

   The base-10 logarithm of e: log₁₀(e).
   
   Value: 0.43429448190325182765112891891660508...
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.ln2

   The natural logarithm of 2: ln(2).
   
   Value: 0.69314718055994530941723212145817656...
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.ln10

   The natural logarithm of 10: ln(10).
   
   Value: 2.30258509299404568401799145468436420...
   
   :type: QuadPrecision
```

## Type Limits

```{eval-rst}
.. data:: numpy_quaddtype.epsilon

   Machine epsilon: the smallest positive number such that 1.0 + epsilon ≠ 1.0.
   
   Approximately 1.93 × 10⁻³⁴.
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.max_value

   The largest representable finite quad-precision value.
   
   Approximately 1.19 × 10⁴⁹³².
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.smallest_normal

   The smallest positive normalized quad-precision value.
   
   Approximately 3.36 × 10⁻⁴⁹³².
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.smallest_subnormal

   The smallest positive subnormal (denormalized) quad-precision value.
   
   :type: QuadPrecision

.. data:: numpy_quaddtype.resolution

   The approximate decimal resolution of quad precision.
   
   :type: QuadPrecision
```

## Type Information

```{eval-rst}
.. data:: numpy_quaddtype.bits

   Total number of bits in quad precision representation.
   
   :value: 128
   :type: int

.. data:: numpy_quaddtype.precision

   Approximate number of significant decimal digits.
   
   :value: 33
   :type: int
```

## Example Usage

```python
from numpy_quaddtype import (
    pi, e, log2e, log10e, ln2, ln10,
    epsilon, max_value, smallest_normal,
    bits, precision
)

# Mathematical constants
print(f"π = {pi}")
print(f"e = {e}")

# Verify relationships
import numpy as np
from numpy_quaddtype import QuadPrecDType

# e^(ln2) should equal 2
two = np.exp(np.array([ln2]))[0]
print(f"e^(ln2) = {two}")

# log2(e) * ln(2) should equal 1
one = log2e * ln2
print(f"log2(e) × ln(2) = {one}")

# Type limits
print(f"\nQuad precision uses {bits} bits")
print(f"Approximately {precision} decimal digits of precision")
print(f"Machine epsilon: {epsilon}")
```
