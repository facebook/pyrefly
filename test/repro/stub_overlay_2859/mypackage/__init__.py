from mypackage._core import add


def sub(a: int, b: int) -> int:
    """Subtract two integers. This function HAS type hints."""
    return add(a, -b)


__all__ = ["add", "sub"]
