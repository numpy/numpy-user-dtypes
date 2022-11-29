# Scalar quantity must be defined _before_ the dtype, so don't isort it.
# During initialization of _quaddtype_main, QuadScalar is imported from this (partially initialized)
# module, and therefore has to be defined first.
from .quadscalar import QuadScalar  # isort: skip
from ._quaddtype_main import QuadDType
