from collections.abc import Hashable
import dataclasses
from typing import (
    Literal,
    TypeAlias,
)

@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int

ReductionKernelType: TypeAlias = Literal[
    "all",
    "any",
    "corrwith",
    "count",
    "first",
    "idxmax",
    "idxmin",
    "last",
    "max",
    "mean",
    "median",
    "min",
    "nunique",
    "prod",
    # as long as `quantile`'s signature accepts only
    # a single quantile value, it's a reduction.
    # GH#27526 might change that.
    "quantile",
    "sem",
    "size",
    "skew",
    "std",
    "sum",
    "var",
]

TransformationKernelType: TypeAlias = Literal[
    "bfill",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "diff",
    "ffill",
    "fillna",
    "ngroup",
    "pct_change",
    "rank",
    "shift",
]

TransformReductionListType: TypeAlias = ReductionKernelType | TransformationKernelType
