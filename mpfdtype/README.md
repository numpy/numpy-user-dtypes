# A multi precision DType for NumPy

A DType and scalar which uses [MPFR](https://www.mpfr.org/) for multi precision floating point math.  MPFR itself has an LGPL license. 

A very basic example::

    import numpy as np
    from mpfdtype import MPFDType, MPFloat

    # create an array with 200 bits precision:
    arr = np.arange(3).astype(MPFDType(200))
    print(repr(arr))
    # array(['0E+00', '1.0E+00', '2.0E+00'], dtype=MPFDType(200))

    # Math uses the input precision (wraps almost all math functions):
    res = arr**2 + np.sqrt(arr)
    print(repr(res))
    # array(['0E+00', '2.0E+00',
    #        '5.4142135623730950488016887242096980785696718753769480731766784E+00'],
    #       dtype=MPFDType(200))

    # cast to a different precision:
    arr2 = arr.astype(MPFDType(500))
    print(repr(arr2))
    # array(['0E+00', '1.0E+00', '2.0E+00'], dtype=MPFDType(500))

    res = arr + arr2
    print(repr(res))  # uses the larger precision now
    # array(['0E+00', '2.0E+00', '4.0E+00'], dtype=MPFDType(500))

There also is an `mpf.MPFloat(value, prec=None)`.  There is no "context"
as most libraries like this (including mpfr itself) typically have.
The rounding mode is always the normal one.
