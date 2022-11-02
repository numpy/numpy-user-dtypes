"""
These are just some hacks to reduce the overhead a bit.  Note that this does
not mean that the overhead is particularly small.  It is not.

The other nice thing about this Python file: Actually, all that is necessary
to swap out the unit provider is here right now :).
"""

from functools import lru_cache

import unyt

# Would need to add `__weakref__` to the slots and doubt it is super important
# so not using a WeakValueDictionary:
_unit_cache = {}


def get_unit(obj=""):
    """
    Small wrapper around unyt.Unit, because a cache seems necessary/worthwhile
    but unyt doesn't seem to cache as aggressively as it should (at least
    not by default).
    """
    # Lets just assume all objects are hashable in practice:
    unit = _unit_cache.get(obj, None)
    if unit is not None:
        return unit

    unit = unyt.Unit(obj)
    _unit_cache[obj] = unit

    return unit


@lru_cache(128)
def get_conversion_factor(from_unit, to_unit):
    """Same as above, this feels very worthwhile to cache, but not sure unyt
    caches as aggressive as possible/reasonable.
    (Maybe it does and the _above_ failure to cache just mades this one fail
    also.)
    """
    return from_unit.get_conversion_factor(to_unit)
