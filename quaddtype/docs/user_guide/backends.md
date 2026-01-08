# Backends

NumPy QuadDType supports two computational backends for quad-precision arithmetic. Understanding the differences helps you choose the right one for your use case.

## Available Backends

### SLEEF (Default)

**SLEEF** (SIMD Library for Evaluating Elementary Functions) is the default and recommended backend.

```python
from numpy_quaddtype import QuadPrecDType, QuadPrecision

# Explicit SLEEF backend
dtype = QuadPrecDType("sleef")
scalar = QuadPrecision(3.14, backend="sleef")

# Or simply use defaults
dtype = QuadPrecDType()  # SLEEF is default
```

**Advantages:**
- ✅ True IEEE 754 binary128 quad precision
- ✅ SIMD-optimized for performance
- ✅ Consistent behavior across all platforms
- ✅ Full suite of mathematical functions

**Considerations:**
- Uses the SLEEF library (bundled with the package)

### Long Double

The **longdouble** backend uses your platform's native `long double` type.

```python
from numpy_quaddtype import QuadPrecDType, QuadPrecision, is_longdouble_128

# Check if your platform has 128-bit long double
print(f"Is long double 128-bit? {is_longdouble_128()}")

# Use longdouble backend
dtype = QuadPrecDType("longdouble")
scalar = QuadPrecision(3.14, backend="longdouble")
```

**Advantages:**
- ✅ Uses native CPU instructions (when available)
- ✅ No external library dependency

**Considerations:**
- ⚠️ Precision varies by platform (see table below)
- ⚠️ Not true quad precision on most platforms

## Platform-Specific Long Double Precision

| Platform | Architecture | Long Double Size | Precision |
|----------|--------------|------------------|-----------|
| Linux | x86_64 | 80-bit (stored as 128) | ~18-19 decimal digits |
| Linux | aarch64 | 128-bit | ~33-34 decimal digits |
| macOS | x86_64 | 64-bit | Same as double |
| macOS | arm64 | 64-bit | Same as double |
| Windows | x64 | 64-bit | Same as double |

```{warning}
On macOS and Windows, `long double` is typically the same as `double` (64-bit), 
providing no precision benefit. Use the SLEEF backend for true quad precision 
on these platforms.
```

## Checking Backend Support

```python
from numpy_quaddtype import is_longdouble_128

if is_longdouble_128():
    print("Your platform supports 128-bit long double!")
    print("Both backends will provide similar precision.")
else:
    print("Long double is NOT 128-bit on your platform.")
    print("Use SLEEF backend for true quad precision.")
```

## Convenience Functions

For cleaner code, use the pre-defined helper functions:

```python
from numpy_quaddtype import (
    SleefQuadPrecDType, 
    SleefQuadPrecision,
    LongDoubleQuadPrecDType, 
    LongDoubleQuadPrecision
)

# SLEEF backend
sleef_dtype = SleefQuadPrecDType()
sleef_scalar = SleefQuadPrecision("3.14159265358979323846")

# Long double backend  
ld_dtype = LongDoubleQuadPrecDType()
ld_scalar = LongDoubleQuadPrecision(3.14)
```

## Checking Which Backend is in Use

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType, QuadBackend

dtype = QuadPrecDType("sleef")
print(f"Backend: {dtype.backend}")  # QuadBackend.SLEEF

# Compare backends
if dtype.backend == QuadBackend.SLEEF:
    print("Using SLEEF backend")
elif dtype.backend == QuadBackend.LONGDOUBLE:
    print("Using longdouble backend")
```

## Mixing Backends

```{warning}
Arrays with different backends cannot be directly combined in operations.
```

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType

# These have different backends
sleef_arr = np.array([1, 2, 3], dtype=QuadPrecDType("sleef"))
ld_arr = np.array([4, 5, 6], dtype=QuadPrecDType("longdouble"))

# This will raise an error:
# result = sleef_arr + ld_arr  # Error!

# Convert to same backend first:
ld_arr_converted = ld_arr.astype(QuadPrecDType("sleef"))
result = sleef_arr + ld_arr_converted  # Works!
```

## Recommendations

| Use Case | Recommended Backend |
|----------|---------------------|
| Cross-platform consistency | SLEEF |
| Maximum precision needed | SLEEF |
| Linux aarch64 with native support | Either (SLEEF preferred) |
| Performance-critical on x86_64 | SLEEF |
| Debugging/comparison | Both (for validation) |
