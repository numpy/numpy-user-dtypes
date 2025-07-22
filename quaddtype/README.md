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

The code needs the quad precision pieces of the sleef library, which is not available on most systems by default, so we have to generate that first. Choose the appropriate section below based on your operating system.

### Linux/Unix/macOS

The below assumes one has the required pieces to build sleef (cmake and libmpfr-dev), and that one is in the package directory locally.

```bash
git clone --branch 3.8 https://github.com/shibatch/sleef.git
cd sleef
cmake -S . -B build -DSLEEF_BUILD_QUAD:BOOL=ON -DSLEEF_BUILD_SHARED_LIBS:BOOL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build/ --clean-first -j
cd ..
```

Building the `numpy-quaddtype` package from locally installed sleef:

```bash
export SLEEF_DIR=$PWD/sleef/build
export LIBRARY_PATH=$SLEEF_DIR/lib
export C_INCLUDE_PATH=$SLEEF_DIR/include
export CPLUS_INCLUDE_PATH=$SLEEF_DIR/include

# setup the virtual env
python3 -m venv temp
source temp/bin/activate

# Install the package
pip install meson-python numpy pytest

export LDFLAGS="-Wl,-rpath,$SLEEF_DIR/lib -fopenmp -latomic -lpthread"
export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC"

# To build without QBLAS (default for MSVC)
# export CFLAGS="-fPIC -DDISABLE_QUADBLAS"
# export CXXFLAGS="-fPIC -DDISABLE_QUADBLAS"

python -m pip install . -v --no-build-isolation -Cbuilddir=build -C'compile-args=-v'

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

2. **Clone and Build SLEEF**

   ```powershell
   # Clone SLEEF library
   git clone --branch 3.8 https://github.com/shibatch/sleef.git
   cd sleef

   # Configure with CMake for Windows
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSLEEF_BUILD_QUAD:BOOL=ON -DSLEEF_BUILD_SHARED_LIBS:BOOL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON

   # Build and install SLEEF
   cmake --build build --config Release
   cmake --install build --prefix "C:/sleef" --config Release

   cd ..
   ```

3. **Setup Python Environment**

   ```powershell
   # Create and activate virtual environment
   python -m venv numpy_quad_env
   .\numpy_quad_env\Scripts\Activate.ps1

   # Install build dependencies
   pip install -U pip
   pip install meson-python numpy pytest ninja meson
   ```

4. **Set Environment Variables**

   ```powershell
   # Set up paths and compiler flags
   $env:INCLUDE = "C:/sleef/include;$env:INCLUDE"
   $env:LIB = "C:/sleef/lib;$env:LIB"
   $env:PATH = "C:/sleef/bin;$env:PATH"

   # Note: QBLAS is disabled on Windows due to MSVC compatibility issues
   $env:CFLAGS = "/IC:/sleef/include /DDISABLE_QUADBLAS"
   $env:CXXFLAGS = "/IC:/sleef/include /DDISABLE_QUADBLAS"
   $env:LDFLAGS = "C:/sleef/lib/sleef.lib C:/sleef/lib/sleefquad.lib"
   ```

5. **Build and Install numpy-quaddtype**

   ```powershell
   # Ensure submodules are initialized
   git submodule update --init --recursive

   # Build and install the package
   python -m pip install . -v --no-build-isolation -Cbuilddir=build -C'compile-args=-v'
   ```

6. **Test Installation**

   ```powershell
   # Run tests
   pytest -s tests/
   ```

1. **QBLAS Disabled**: QuadBLAS optimization is automatically disabled on Windows builds due to MSVC compatibility issues. This is handled by the `-DDISABLE_QUADBLAS` compiler flag.

2. **Visual Studio Version**: The instructions assume Visual Studio 2022. For other versions, adjust the generator string:
   - VS 2019: `"Visual Studio 16 2019"`
   - VS 2017: `"Visual Studio 15 2017"`

3. **Architecture**: The instructions are for x64. For x86 builds, change `-A x64` to `-A Win32`.

4. **Alternative SLEEF Location**: If you prefer to install SLEEF elsewhere, update all path references accordingly.

#### Windows Troubleshooting
- **Link errors**: Verify that `sleef.lib` and `sleefquad.lib` exist in `C:/sleef/lib/`