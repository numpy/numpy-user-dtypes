# A dtype that stores metadata

This is a simple proof-of-concept dtype using the (as of late 2022) experimental
[new dtype
implementation](https://numpy.org/neps/nep-0041-improved-dtype-support.html) in
NumPy. For now all it does it storea piece of static metadata in the dtype
itself.

## Building

Ensure Meson and NumPy are installed in the python environment you would like to use:

```
$ python3 -m pip install meson numpy build
```

Build with meson, create a wheel, and install it

```
$ rm -r dist/
$ meson build
$ python -m build --wheel -Cbuilddir=build
$ pip install dist/metadatadtype*.whl
```

The `mesonpy` build backend for pip [does not currently support editable
installs](https://github.com/mesonbuild/meson-python/issues/47), so `pip install
-e .` will not work.
