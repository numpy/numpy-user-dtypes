"""A dtype for working with ASCII data

This is an example usage of the experimental new dtype API
in Numpy and is not intended for any real purpose.
"""

from .scalar import ASCIIScalar  # isort: skip
from ._asciidtype_main import ASCIIDType

__all__ = ["ASCIIDType", "ASCIIScalar"]
