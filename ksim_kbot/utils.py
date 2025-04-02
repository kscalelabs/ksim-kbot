"""Utility functions.."""

from jaxtyping import Array


def cubic_bezier_interpolation(y_start: Array, y_end: Array, x: Array) -> Array:
    """Cubic bezier interpolation."""
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier