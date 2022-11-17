# A dtype that stores unit metadata

This is a simple proof-of-concept dtype using the (as of late 2022) experimental
[new dtype
implementation](https://numpy.org/neps/nep-0041-improved-dtype-support.html) in
NumPy. It leverages the [unyt](https://unyt.readthedocs.org) library's `Unit`
type to store unit metadata, but relies on the dtype machinery to implement
mathematical operations rather than an ndarray subclass like `unyt_array`. This
is currently mostly for experimenting with the dtype API and is not useful for
real work.

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
$ python -m pip install --force-reinstall dist/unytdtype*.whl
```

The `mesonpy` build backend for pip [does not currently support editable
installs](https://github.com/mesonbuild/meson-python/issues/47), so `pip install
-e .` will not work.
