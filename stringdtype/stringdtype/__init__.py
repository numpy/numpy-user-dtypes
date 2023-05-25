"""A dtype for working with variable-length string data

"""

from .missing import NA  # isort: skip
from .scalar import StringScalar, PandasStringScalar  # isort: skip
from ._main import PandasStringDType, StringDType, _memory_usage

__all__ = [
    "NA",
    "StringDType",
    "StringScalar",
    "_memory_usage",
]

# this happens when pandas isn't importable
if StringDType is PandasStringDType:
    del PandasStringDType
else:
    __all__.extend("PandasStringDType")
