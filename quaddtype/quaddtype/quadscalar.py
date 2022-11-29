"""Quad scalar floating point type for numpy."""


class QuadScalar:
    """Quad scalar floating point type."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{self.value}"
