# A dtype that stores pointers to strings

This is a simple proof-of-concept dtype using the (as of early 2023) experimental
[new dtype
implementation](https://numpy.org/neps/nep-0041-improved-dtype-support.html) in
NumPy.

## Building

Ensure Meson and NumPy are installed in the python environment you would like to use:

```
$ python3 -m pip install meson meson-python numpy build patchelf
```

Build with meson, create a wheel, and install it

```
$ rm -r dist/
$ meson build
$ python -m build --wheel -Cbuilddir=build
$ python -m pip install dist/asciidtype*.whl
```

The `mesonpy` build backend for pip [does not currently support editable
installs](https://github.com/mesonbuild/meson-python/issues/47), so `pip install
-e .` will not work.
