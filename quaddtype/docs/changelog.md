# Changelog

All notable changes to NumPy QuadDType will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025

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

## [0.1.0] - 2024

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

## Unreleased

### Planned

- Complex quad precision support
- Additional linear algebra functions
- GPU acceleration exploration
- Improved performance for small arrays
