# Constants Reference

Pre-defined mathematical constants with quad precision accuracy.

## Mathematical Constants

```{eval-rst}
.. data:: numpy_quaddtype.pi

   The mathematical constant :math:`\pi` (pi).

   :value: 3.14159265358979323846264338327950288...

   :type: QuadPrecision

.. data:: numpy_quaddtype.e

   Euler's number :math:`e`, the base of natural logarithms.

   :value: 2.71828182845904523536028747135266249...

   :type: QuadPrecision

.. data:: numpy_quaddtype.log2e

   The base-2 logarithm of :math:`e`: :math:`\log_{2}{e}`.

   :value: 1.44269504088896340735992468100189213...

   :type: QuadPrecision

.. data:: numpy_quaddtype.log10e

   The base-10 logarithm of :math:`e`: :math:`\log_{10}{e}`.

   :value: 0.43429448190325182765112891891660508...

   :type: QuadPrecision

.. data:: numpy_quaddtype.ln2

   The natural logarithm of 2: :math:`\log_{e}{2}`.

   :value: 0.69314718055994530941723212145817656...

   :type: QuadPrecision

.. data:: numpy_quaddtype.ln10

   The natural logarithm of 10: :math:`\log_{e}{10}`.

   :value: 2.30258509299404568401799145468436420...

   :type: QuadPrecision
```

## Type Limits

```{eval-rst}
.. data:: numpy_quaddtype.epsilon

   Machine epsilon: the smallest positive number such that :math:`1.0 + \epsilon \neq 1.0`.

   :value: :math:`2^{-112}` or approximately :math:`1.93 \cdot 10^{-34}`

   :type: QuadPrecision

.. data:: numpy_quaddtype.max_value

   The largest representable finite quad-precision value.

   The largest negative representable finite quad-precision value is ``-numpy_quaddtype.max_value``.

   :value: :math:`216383 \cdot (2 - 2^{-112})` or approximately :math:`1.19 \cdot 10^{4932}`

   :type: QuadPrecision

.. data:: numpy_quaddtype.smallest_normal

   The smallest positive normal (normalized, mantissa has a leading 1 bit) quad-precision value.

   :value: :math:`2^{-16382} \cdot (1 - 2^{-112})` or approximately :math:`3.36 \cdot 10^{-4932}`

   :type: QuadPrecision

.. data:: numpy_quaddtype.smallest_subnormal

   The smallest positive subnormal (denormalized, mantissa has a leading 0 bit) quad-precision value.

   :value: :math:`2^{-16494}` or approximately :math:`6.48 \cdot 10^{-4966}`

   :type: QuadPrecision

.. data:: numpy_quaddtype.resolution

   The approximate decimal resolution of quad precision, i.e. `10 ** (-precision)`.

   :value: :math:`10^{-33}`

   :type: QuadPrecision
```

## Type Information

```{eval-rst}
.. data:: numpy_quaddtype.bits

   The total number of bits in quad precision representation.

   :value: 128
   :type: int

.. data:: numpy_quaddtype.precision

   The approximate number of significant decimal digits.

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
two = np.exp(np.array(ln2))
print(f"e^(ln2) = {two}")

# log2(e) * ln(2) should equal 1
one = log2e * ln2
print(f"log2(e) × ln(2) = {one}")

# Type limits
print(f"\nQuad precision uses {bits} bits")
print(f"Approximately {precision} decimal digits of precision")
print(f"Machine epsilon: {epsilon}")
```
