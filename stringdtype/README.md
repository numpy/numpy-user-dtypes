# A dtype that stores pointers to strings

This is the prototype implementation of the variable-width UTF-8 string DType
described in [NEP 55](https://numpy.org/neps/nep-0055-string_dtype.html).

See the NEP for implementation details and usage examples. See
`numpy.dtypes.StringDType` for the version that made it into NumPy.

## Building

Ensure Meson and NumPy are installed in the python environment you would like to use:

```
$ python3 -m pip install meson meson-python
```

It is important to have the latest development version of numpy installed.
Nightly wheels work well for this purpose, and can be installed easily:

```bash
$ pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy
```

You can install with `pip` directly, taking care to disable build isolation so
the numpy nightly gets picked up at build time:

```bash
$ pip install -v . --no-build-isolation
```

If you want to work on the `stringdtype` code, you can build with meson,
create a wheel, and install it.

```bash
$ rm -r dist/
$ meson build
$ python -m build --wheel -Cbuilddir=build
$ python -m pip install dist/path-to-wheel-file.whl
```
