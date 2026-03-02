from typing import Literal

import numpy as np
from numpy.random import RandomState

def sample_without_replacement(
    n_population: int,
    n_samples: int,
    random_state: int | RandomState | None = None,
    method: Literal["auto", "tracking_selection", "reservoir_sampling", "pool"] = "auto",
) -> np.ndarray: ...
