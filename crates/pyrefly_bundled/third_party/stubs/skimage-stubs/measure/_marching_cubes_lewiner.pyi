def marching_cubes(
    volume,
    level: float | None = None,
    *,
    spacing=...,
    gradient_direction: str = "descent",
    step_size: int = 1,
    allow_degenerate: bool = True,
    method: str = "lewiner",
    mask=None,
): ...
def _marching_cubes_lewiner(
    volume,
    level,
    spacing,
    gradient_direction,
    step_size,
    allow_degenerate,
    use_classic,
    mask,
): ...
def _to_array(args): ...

# Map an edge-index to two relative pixel positions. The ege index
# represents a point that lies somewhere in between these pixels.
# Linear interpolation should be used to determine where it is exactly.
#   0
# 3   1   ->  0x
#   2         xx
EDGETORELATIVEPOSX = ...
EDGETORELATIVEPOSY = ...
EDGETORELATIVEPOSZ = ...

def _get_mc_luts(): ...
