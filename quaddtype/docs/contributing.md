# Contributing

We welcome contributions to NumPy QuadDType! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- GCC or Clang compiler
- CMake ≥ 3.15
- Git

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/numpy/numpy-user-dtypes.git
cd numpy-user-dtypes/quaddtype

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install NumPy (development version)
pip install "numpy @ git+https://github.com/numpy/numpy.git"

# Install development dependencies
pip install -e ".[test,docs]" -v --no-build-isolation
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_basic.py

# Run with parallel execution
pytest -n auto tests/  # requires pytest-xdist
```

## Code Style

We follow standard Python conventions:

- **PEP 8** for Python code style
- **Type hints** for public APIs
- **Docstrings** for all public functions and classes

### Type Checking

```bash
# Run mypy
mypy numpy_quaddtype/

# Run pyright
pyright numpy_quaddtype/
```

## Building Documentation

```bash
# Install documentation dependencies
pip install ".[docs]"

# Build HTML documentation
cd docs/
make html

# View locally
python -m http.server --directory _build/html
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Your Changes

- Write code with tests
- Add docstrings
- Update documentation if needed

### 3. Run Tests

```bash
pytest tests/
```

### 4. Submit a Pull Request

- Push your branch to GitHub
- Open a pull request against `main`
- Fill out the PR template
- Wait for review

## Project Structure

```
quaddtype/
├── docs/               # Documentation (Sphinx)
├── numpy_quaddtype/    # Python package
│   ├── __init__.py     # Public API
│   ├── __init__.pyi    # Type stubs
│   ├── _quaddtype_main.pyi  # C extension stubs
│   └── src/            # C source files
├── tests/              # Test suite
├── subprojects/        # Meson subprojects (SLEEF)
├── meson.build         # Build configuration
└── pyproject.toml      # Package metadata
```

## C Extension Development

The core functionality is implemented in C. Key files:

- `numpy_quaddtype/src/quaddtype_main.c` - Main extension module
- `numpy_quaddtype/src/scalar.c` - QuadPrecision scalar implementation
- `numpy_quaddtype/src/dtype.c` - QuadPrecDType implementation
- `numpy_quaddtype/src/umath.c` - Universal function implementations

### Building the C Extension

```bash
# Rebuild after C changes
pip install . -v --no-build-isolation

# With debug symbols
CFLAGS="-g -O0" pip install . -v --no-build-isolation
```

## Reporting Issues

When reporting bugs, please include:

1. Operating system and version
2. Python version
3. NumPy version
4. NumPy-QuadDType version
5. Minimal code to reproduce the issue
6. Full error traceback

## Code of Conduct

This project follows the [NumPy Code of Conduct](https://numpy.org/code-of-conduct/).

## License

By contributing to NumPy QuadDType, you agree that your contributions will be licensed under the BSD-3-Clause License.
