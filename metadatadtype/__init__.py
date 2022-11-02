import unyt

from .scalar import QuantityScalar

from ._metadatadtype_main import UnitDType


# Hack a few common units into scalars for simpler handling.
# this defines `m`, `cm`, etc.
# That is so we can write 1 * m and have it work...
for name, unit in unyt.__dict__.items():
    if type(unit) is unyt.Unit:
        globals()[name] = QuantityScalar(1, unit)
