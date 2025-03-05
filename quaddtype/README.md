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

## Install from source

The code needs the quad precision pieces of the sleef library, which
is not available on most systems by default, so we have to generate
that first.  The below assumes one has the required pieces to build
sleef (cmake and libmpfr-dev), and that one is in the package
directory locally.

```
git clone https://github.com/shibatch/sleef.git
cd sleef
cmake -S . -B build -DSLEEF_BUILD_QUAD:BOOL=ON -DSLEEF_BUILD_SHARED_LIBS:BOOL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/ --clean-first -j
cd ..
```

In principle, one can now install this system-wide, but easier would
seem to use the version that was just created, as follows:
```
export SLEEF_DIR=$PWD/sleef/build
export LIBRARY_PATH=$SLEEF_DIR/lib
export C_INCLUDE_PATH=$SLEEF_DIR/include
export CPLUS_INCLUDE_PATH=$SLEEF_DIR/include
python3 -m venv temp
source temp/bin/activate
pip install meson-python numpy pytest
pip install -e . -v --no-build-isolation
export LD_LIBRARY_PATH=$SLEEF_DIR/lib
```

Here, we created an editable install on purpose, so one can just work
from the package directory if needed, e.g., to run the tests with,
```
python -m pytest
```
