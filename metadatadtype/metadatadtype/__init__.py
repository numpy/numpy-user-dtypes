"""A dtype that carries around metadata.

This is an example usage of the experimental new dtype API
in Numpy and is not intended for any real purpose.
"""

from .scalar import MetadataScalar  # isort: skip
from ._metadatadtype_main import MetadataDType

__all__ = ["MetadataDType", "MetadataScalar"]
