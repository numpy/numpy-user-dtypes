# A multi precision DType for NumPy

:warning: Note that while the DType is BSD license, MPFR and GMP which it
must be linked with are not.

A very basic example::

    import mpfdtype as mpf

    # create an array with 200 bits precision:
    arr = np.arange(10).astype(MPFDType(200))

    # Math uses the input precision:
    arr**2 + np.sqrt(arr)

    # cast to a different precision:
    arr2 = arr.astype(mpf.MPFDtype(500))

    res = arr1 + arr2
    print(res)  # uses the larger precision now

There also is an `mpf.MPFloat(value, prec=None)`.  There is no "context"
as most libraries like this (including mpfr itself) typically have.
The rounding mode is always the normal one.
