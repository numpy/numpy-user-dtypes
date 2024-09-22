# Numpy-QuadDType

## Installation

```
pip install numpy==2.1.0
pip install -i https://test.pypi.org/simple/ quaddtype
```

## Usage

```python
import numpy as np
from numpy_quaddtype import QuadPrecDType, QuadPrecision

# using sleef backend (default)
np.array([1,2,3], dtype=QuadPrecDType())
np.array([1,2,3], dtype=QuadPrecDType("sleef"))

# using longdouble backend
np.array([1,2,3], dtype=QuadPrecDType("longdouble"))
```
