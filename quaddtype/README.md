# Numpy-QuadDType

A cross-platform Quad (128-bit) float Data-Type for NumPy.

## Installation

```bash
pip install numpy
pip install numpy-quaddtype
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

## Installation from source

#### Prerequisites

- **gcc/clang**
- **CMake** (≥3.15)
- **Python 3.10+**
- **Git**

### Linux/Unix/macOS

Building the `numpy-quaddtype` package from locally installed sleef:

```bash
# setup the virtual env
python3 -m venv temp
source temp/bin/activate

# Install the package
pip install meson-python numpy pytest

# If you see errors about a missing atomics library, you might need -latomic
export LDFLAGS="-fopenmp -lpthread"

# To build without QBLAS (default for MSVC)
# export CFLAGS="-DDISABLE_QUADBLAS"
# export CXXFLAGS="-DDISABLE_QUADBLAS"

python -m pip install . -v --no-build-isolation

# Run the tests
cd ..
python -m pytest
```

### Windows

#### Prerequisites

- **Visual Studio 2017 or later** (with MSVC compiler)
- **CMake** (≥3.15)
- **Python 3.10+**
- **Git**

#### Step-by-Step Installation

1. **Setup Development Environment**

   Open a **Developer Command Prompt for VS** or **Developer PowerShell for VS** to ensure MSVC is properly configured.

2. **Setup Python Environment**

   ```powershell
   # Create and activate virtual environment
   python -m venv numpy_quad_env
   .\numpy_quad_env\Scripts\Activate.ps1

   # Install build dependencies
   pip install -U pip
   pip install meson-python numpy pytest ninja meson
   ```

3. **Set Environment Variables**

   ```powershell
   # Note: QBLAS is disabled on Windows due to MSVC compatibility issues
   $env:CFLAGS = "/DDISABLE_QUADBLAS"
   $env:CXXFLAGS = "/DDISABLE_QUADBLAS"
   ```

4. **Build and Install numpy-quaddtype**

   ```powershell
   # Build and install the package
   python -m pip install . -v --no-build-isolation
   ```

5. **Test Installation**

   ```powershell
   # Run tests
   pytest -s tests/
   ```

6. **QBLAS Disabled**: QuadBLAS optimization is automatically disabled on Windows builds due to MSVC compatibility issues. This is handled by the `-DDISABLE_QUADBLAS` compiler flag.

7. **Visual Studio Version**: The instructions assume Visual Studio 2022. For other versions, adjust the generator string:

   - VS 2019: `"Visual Studio 16 2019"`
   - VS 2017: `"Visual Studio 15 2017"`

8. **Architecture**: The instructions are for x64. For x86 builds, change `-A x64` to `-A Win32`.
