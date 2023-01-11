"""A dtype for working with string data

This is an example usage of the experimental new dtype API
in Numpy and is not intended for any real purpose.
"""

from .scalar import StringScalar  # isort: skip
from ._main import StringDType

__all__ = ["StringDType", "StringScalar"]
