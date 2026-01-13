# NumPy-QuadDType

[![PyPI](https://img.shields.io/pypi/v/numpy-quaddtype.svg)](https://pypi.org/project/numpy-quaddtype/)
[![PyPI Downloads](https://static.pepy.tech/badge/numpy-quaddtype/month)](https://pepy.tech/project/numpy-quaddtype)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/numpy_quaddtype.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/numpy_quaddtype)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

A cross-platform Quad (128-bit) float Data-Type for NumPy.

## Table of Contents

- [NumPy-QuadDType](#numpy-quaddtype)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Installation from source](#installation-from-source)
    - [Linux/Unix/macOS](#linuxunixmacos)
    - [Windows](#windows)
  - [Building with ThreadSanitizer (TSan)](#building-with-threadsanitizer-tsan)
  - [Building the documentation](#building-the-documentation)
    - [Serving the documentation](#serving-the-documentation)
  - [Development Tips](#development-tips)
    - [Cleaning the Build Directory](#cleaning-the-build-directory)

## Installation

```bash
pip install "numpy>=2.4"
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

### Linux/Unix/macOS

**Prerequisites:** gcc/clang, CMake (≥3.15), Python 3.11+, Git, NumPy ≥ 2.4

```bash
# setup the virtual env
python3 -m venv temp
source temp/bin/activate

# Install build and test dependencies
pip install pytest meson meson-python "numpy>=2.4"

# To build without QBLAS (default for MSVC)
# export CFLAGS="-DDISABLE_QUADBLAS"
# export CXXFLAGS="-DDISABLE_QUADBLAS"

python -m pip install ".[test]" -v

# Run the tests
python -m pytest tests
```

### Windows

**Prerequisites:** Visual Studio 2017+ (with MSVC), CMake (≥3.15), Python 3.11+, Git

1. **Setup Development Environment**

   Open a **Developer Command Prompt for VS** or **Developer PowerShell for VS** to ensure MSVC is properly configured.

2. **Setup Python Environment**

   ```powershell
   # Create and activate virtual environment
   python -m venv numpy_quad_env
   .\numpy_quad_env\Scripts\Activate.ps1

   # Install build dependencies
   pip install -U pip
   pip install numpy pytest ninja meson
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
   python -m pip install ".[test]" -v
   ```

5. **Test Installation**

   ```powershell
   # Run tests
   pytest -s tests
   ```

6. **QBLAS Disabled**: QuadBLAS optimization is automatically disabled on Windows builds due to MSVC compatibility issues. This is handled by the `-DDISABLE_QUADBLAS` compiler flag.

7. **Visual Studio Version**: The instructions assume Visual Studio 2022. For other versions, adjust the generator string:

   - VS 2019: `"Visual Studio 16 2019"`
   - VS 2017: `"Visual Studio 15 2017"`

8. **Architecture**: The instructions are for x64. For x86 builds, change `-A x64` to `-A Win32`.

## Building with ThreadSanitizer (TSan)

This is a development feature to help detect threading issues. To build `numpy-quaddtype` with TSan enabled, follow these steps:

> Use of clang is recommended with machine NOT supporting `libquadmath` (like ARM64). Set the compiler to clang/clang++ before proceeding.
>
> ```bash
> export CC=clang
> export CXX=clang++
> ```

1. Compile free-threaded CPython with TSan support. Follow the [Python Free-Threading Guide](https://py-free-threading.github.io/thread_sanitizer/#compile-free-threaded-cpython-with-tsan) for detailed instructions.
2. Create and activate a virtual environment using the TSan-enabled Python build.
3. Installing dependencies:

```bash
python -m pip install meson meson-python wheel ninja
# Need NumPy built with TSan as well
python -m pip install "numpy @ git+https://github.com/numpy/numpy" -C'setup-args=-Db_sanitize=thread'
```

4. Building SLEEF with TSan:

```bash
# clone the repository
git clone https://github.com/shibatch/sleef.git
cd sleef
git checkout 43a0252ba9331adc7fb10755021f802863678c38

# Build SLEEF with TSan
cmake \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_FLAGS="-fsanitize=thread -g -O1" \
-DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" \
-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
-DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread" \
-DSLEEF_BUILD_QUAD=ON \
-DSLEEF_BUILD_TESTS=OFF \
-DCMAKE_INSTALL_PREFIX=/usr/local
-S . -B build

cmake --build build -j --clean-first
cmake --install build
```

5. Build and install `numpy-quaddtype` with TSan:

```bash
# SLEEF is already installed with TSan, we need to provide proper flags to numpy-quaddtype's meson file
# So that it does not build SLEEF again and use the installed one.

export CFLAGS="-fsanitize=thread -g -O0"
export CXXFLAGS="-fsanitize=thread -g -O0"
export LDFLAGS="-fsanitize=thread"
python -m pip install . -vv -Csetup-args=-Db_sanitize=thread
```

## Building the documentation

The documentation for the `numpy-quaddtype` package is built using Sphinx. To build the documentation, follow these steps:

1. Install the required dependencies:

   ```bash
   pip install ."[docs]"
   ```

2. Navigate to the `docs` directory and build the documentation:

   ```bash
   cd docs/
   make html
   ```

3. The generated HTML documentation can be found in the `_build/html` directory within the `docs` folder. Open the `index.html` file in a web browser to view the documentation, or use a local server to serve the files:

   ```bash
   python3 -m http.server --directory _build/html
   ```

### Serving the documentation

The documentation is automatically built and served using GitHub Pages. Every time changes are pushed to the `main` branch, the documentation is rebuilt and deployed to the `gh-pages` branch of the repository. You can access the documentation at:

```
https://numpy.github.io/numpy-user-dtypes/quaddtype/
```

Check the `.github/workflows/build_docs.yml` file for details.

## Development Tips

### Cleaning the Build Directory

The subproject folders (`subprojects/sleef`, `subprojects/qblas`) are cloned as git repositories. To fully clean them, use double force:

```bash
git clean -ffxd
```
