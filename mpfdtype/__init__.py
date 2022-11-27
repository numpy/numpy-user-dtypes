import numpy as np

from ._mpfdtype_main import MPFDType, MPFloat



# Lets add some uglier hacks:

# NumPy uses repr as a fallback (as of writing this code), we want to
# customize the printing of MPFloats though...
def mystr(obj):
    if isinstance(obj, MPFloat):
        return f"'{obj}'"
    return repr(obj)

np.set_printoptions(formatter={"numpystr": mystr})
