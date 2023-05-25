"""A dtype for working with variable-length string data

"""

from .missing import NA  # isort: skip
from .scalar import StringScalar, PandasStringScalar  # isort: skip
from ._main import StringDType, _memory_usage

try:
    from ._main import PandasStringDType
except ImportError:
    PandasStringDType = None

__all__ = [
    "NA",
    "StringDType",
    "StringScalar",
    "_memory_usage",
]

# this happens when pandas isn't importable
if PandasStringDType is None:
    del PandasStringDType
else:
    __all__.extend("PandasStringDType")
