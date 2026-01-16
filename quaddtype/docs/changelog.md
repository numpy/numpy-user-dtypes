# Changelog

All notable changes to NumPy QuadDType will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Full type stub support (`.pyi` files) for static type checking
- Type-safe API with mypy and pyright compatibility
- `QuadBackend` enum for backend type checking
- Pre-defined mathematical constants (`pi`, `e`, `log2e`, etc.)
- Type limit constants (`epsilon`, `max_value`, `smallest_normal`, etc.)
- QuadBLAS threading control functions
- Windows support (with QBLAS disabled)
- Free-threading (GIL-free) support for Python 3.13+
- Comprehensive test suite with thread safety tests

### Changed

- Improved string representation of QuadPrecision values
- Better error messages for invalid operations
- Enhanced documentation

### Fixed

- Memory alignment issues on certain platforms
- Thread safety in scalar operations

## [0.2.2] - 13.10.2025

### Changed

- prioritise system-wide dependencies over meson wrap fallback

## [0.2.1] - 11.10.2025

### Fixed

- multiple copies of OpenMP runtime initialization
- null pointer dereference

## [0.2.0] - 12.09.2025

### Added

- Cast for ubyte and half dtypes

### Changed

- Bundle SLEEF and submodules using meson wrap (sdist compatible)

### Fixed

- smallest_subnormal constant

## [0.1.0] - 03.09.2025

### Added

- Support for Python 3.13 and 3.14
- Support for ufuncs: copysign, sign, signbit, isfinite, isinf, isnan, fmin, fmax, reciprocal, matmul, sinh, cosh, tanh, arcsinh, arccosh, arctanh
- Constants: smallest_subnormal, bits, precision, resolution

### Fixed

- NaN comparisons
- mod ufunc
- rint ufunc for near-halfway cases

## [0.0.1] - 02.07.2025

### Added

- Initial release
- `QuadPrecision` scalar type
- `QuadPrecDType` NumPy dtype
- SLEEF backend for cross-platform quad precision
- Longdouble backend for native support
- Basic arithmetic operations
- Trigonometric functions (sin, cos, tan, etc.)
- Exponential and logarithmic functions
- Comparison operations
- Array broadcasting support
- Linux and macOS wheel builds
