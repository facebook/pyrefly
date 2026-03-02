from typing import Literal

from numpy.typing import ArrayLike

def _show_skimage_info(): ...

class PytestTester:
    def __init__(self, module_name): ...
    def __call__(
        self,
        label: Literal["fast", "full"] = "fast",
        verbose: int = 1,
        extra_argv: ArrayLike | None = None,
        doctests: bool = False,
        coverage: bool = False,
        durations: int = ...,
        tests=None,
    ) -> bool: ...
