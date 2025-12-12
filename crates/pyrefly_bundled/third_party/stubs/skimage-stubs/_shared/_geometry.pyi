__all__ = ["polygon_clip", "polygon_area"]

from .version_requirements import require

@require("matplotlib", ">=3.0.3")
def polygon_clip(rp, cp, r0, c0, r1, c1): ...
def polygon_area(pr, pc) -> float: ...
