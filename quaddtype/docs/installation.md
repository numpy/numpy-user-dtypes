# Installation

## Quick Install

The simplest way to install NumPy QuadDType is via pip:

```bash
pip install numpy-quaddtype
```

```{note}
NumPy QuadDType requires **NumPy 2.0 or later** and **Python 3.11+**.
```

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.11 |
| NumPy | ≥ 2.0 |

## Platform Support

NumPy QuadDType provides pre-built wheels for:

| Platform | Architectures |
|----------|---------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64, arm64 (Apple Silicon) |
| Windows | x64 |

## Installing from Source

For development or if pre-built wheels aren't available for your platform:

### Prerequisites

- **C/C++ Compiler**: GCC or Clang
- **CMake**: ≥ 3.15
- **Python**: 3.11+
- **Git**

### Linux/macOS

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install NumPy (development version required for NumPy 2.x features)
pip install "numpy @ git+https://github.com/numpy/numpy.git"

# Install build dependencies
pip install meson meson-python ninja pytest

# Clone and install
git clone https://github.com/numpy/numpy-user-dtypes.git
cd numpy-user-dtypes/quaddtype
pip install . -v --no-build-isolation
```

### Windows

```{warning}
On Windows, QuadBLAS optimization is automatically disabled due to MSVC compatibility issues.
```

1. Open **Developer Command Prompt for VS** or **Developer PowerShell for VS**

2. Setup environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   pip install -U pip
   pip install numpy pytest ninja meson meson-python
   ```

3. Set compiler flags:
   ```powershell
   $env:CFLAGS = "/DDISABLE_QUADBLAS"
   $env:CXXFLAGS = "/DDISABLE_QUADBLAS"
   ```

4. Build and install:
   ```powershell
   pip install . -v --no-build-isolation
   ```

## Verifying Installation

```python
import numpy as np
from numpy_quaddtype import QuadPrecision, QuadPrecDType

# Check version
import numpy_quaddtype
print(f"numpy-quaddtype version: {numpy_quaddtype.__version__}")

# Create a quad precision value
x = QuadPrecision("3.141592653589793238462643383279502884197")
print(f"π in quad precision: {x}")

# Create an array
arr = np.array([1, 2, 3], dtype=QuadPrecDType())
print(f"Array dtype: {arr.dtype}")
```

## Optional: Development Installation

For contributing to NumPy QuadDType:

```bash
# Clone the repository
git clone https://github.com/numpy/numpy-user-dtypes.git
cd numpy-user-dtypes/quaddtype

# Install in editable mode with test dependencies
pip install -e ".[test,docs]" -v --no-build-isolation
```

## Troubleshooting

### CMake Not Found

If you get a CMake error, install it:

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install cmake

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

### NumPy Version Error

NumPy QuadDType requires NumPy 2.0+. If you have an older version:

```bash
pip install --upgrade numpy>=2.0
```

### Compiler Issues on macOS

If you encounter compiler issues on macOS, ensure you have Xcode command-line tools:

```bash
xcode-select --install
```
