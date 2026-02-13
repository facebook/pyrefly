from numpy.typing import NDArray

from .._shared.utils import deprecate_kwarg

@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def flood_fill(
    image: NDArray,
    seed_point: tuple | int,
    new_value,
    *,
    footprint: NDArray | None = None,
    connectivity: int | None = None,
    tolerance: float | None = None,
    in_place: bool = False,
) -> NDArray: ...
@deprecate_kwarg(
    kwarg_mapping={"selem": "footprint"},
    removed_version="1.0",
    deprecated_version="0.19",
)
def flood(
    image: NDArray,
    seed_point: tuple | int,
    *,
    footprint: NDArray | None = None,
    connectivity: int | None = None,
    tolerance: float | None = None,
) -> NDArray: ...
