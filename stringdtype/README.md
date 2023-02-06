# A dtype that stores pointers to strings

This is a simple proof-of-concept dtype using the (as of early 2023) experimental
[new dtype
implementation](https://numpy.org/neps/nep-0041-improved-dtype-support.html) in
NumPy.

## Building

Ensure Meson and NumPy are installed in the python environment you would like to use:

```
$ python3 -m pip install meson meson-python build patchelf
```

It is important to have the latest development version of numpy installed.
Nightly wheels work well for this purpose, and can be installed easily:

```bash
$ pip install -i https://pypi.anaconda.org/scipy-wheels-nightly/simple numpy
```

Build with meson, create a wheel, and install it.

```bash
$ rm -r dist/
$ meson build
$ python -m build --wheel -Cbuilddir=build
```

Or simply install directly, taking care to install without build isolation:

```bash
$ pip install -v . --no-build-isolation
```
