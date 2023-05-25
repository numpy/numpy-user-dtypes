# quaddtype

Quad (128-bit) float dtype for numpy

## Installation

To install, make sure you have `numpy` nightly installed. Then build without
isolation so that the `quaddtype` can link against the experimental dtype API
headers, which aren't in the latest releases of `numpy`:

```bash
pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy
pip install . --no-build-isolation
```

Developed with Python 3.11, but 3.9 and 3.10 will probably also work.
