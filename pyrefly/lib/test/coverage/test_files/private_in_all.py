# Private names listed in __all__ should be reported as public (issue #3578).

__all__ = ["_foo", "_X"]


def _foo(x: int) -> int:
    return x


def _hidden(x: int) -> int:
    return x


_X: int = 1
_Y: int = 2
