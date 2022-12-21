"""A dtype for working with string data

This is an example usage of the experimental new dtype API
in Numpy and is not intended for any real purpose.
"""

from .scalar import StrScalar  # isort: skip
from ._main import StrPtrDType

__all__ = ["StrPtrDType", "StrScalar"]
