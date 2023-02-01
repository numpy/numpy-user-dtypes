"""A dtype for working with ASCII data

This is an example usage of the experimental new dtype API
in Numpy and is not intended for any real purpose.
"""

from .scalar import ASCIIScalar  # isort: skip
from ._asciidtype_main import ASCIIDType


def _reconstruct_ASCIIDType(*args):
    # this is needed for pickling instances because numpy overrides the pickling
    # behavior of the DTypeMeta class using copyreg. By pickling a wrapper
    # around the ASCIIDType initializer, we avoid triggering the code in numpy
    # that tries to handle pickling DTypeMeta instances. See
    # https://github.com/numpy/numpy/issues/23135#issuecomment-1410967842
    return ASCIIDType(*args)


__all__ = ["ASCIIDType", "ASCIIScalar", "_reconstruct_ASCIIDType"]
