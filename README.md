# A dtype that stores metadata

This is a simple proof-of-concept dtype using the (as of late 2022) experimental
[new dtype
implementation](https://numpy.org/neps/nep-0041-improved-dtype-support.html) in
NumPy. For now all it does it storea piece of static metadata in the dtype
itself.

## Building

Ensure Meson and NumPy are installed in the python environment you would like to use:

```
$ python3 -m pip install meson numpy
```

And then build with meson

```
$ meson build
```
