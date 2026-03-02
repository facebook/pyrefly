from numpy.typing import NDArray

def binary_blobs(
    length: int = 512,
    blob_size_fraction: float = 0.1,
    n_dim: int = 2,
    volume_fraction: float = 0.5,
    seed=None,
) -> NDArray: ...
