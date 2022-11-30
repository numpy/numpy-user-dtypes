"""A dtype that carries around unit metadata.

This is an example usage of the experimental new dtype API
in Numpy and is not yet intended for any real purpose.
"""

from .scalar import UnytScalar  # isort: skip
from ._unytdtype_main import UnytDType

__all__ = ["UnytDType", "UnytScalar"]
