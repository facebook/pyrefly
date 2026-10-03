# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Type stubs for torch.distributions.transforms."""

from typing import Any

from shape_extensions import IntTuple, IntVar
from torch import Tensor

class Transform:
    """Base class for invertible transforms with computable log det Jacobians."""

    domain: Any
    codomain: Any
    bijective: bool
    sign: int

    def __init__(self, cache_size: int = 0) -> None: ...
    def __call__[S: IntTuple](self, x: Tensor[S]) -> Tensor[S]: ...
    def _call[S: IntTuple](self, x: Tensor[S]) -> Tensor[S]: ...
    def _inverse[S: IntTuple](self, y: Tensor[S]) -> Tensor[S]: ...
    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor: ...

class AbsTransform(Transform): ...

class ExpTransform(Transform):
    def log_abs_det_jacobian[S: IntTuple](
        self, x: Tensor[S], y: Tensor[S]
    ) -> Tensor[S]: ...

class SigmoidTransform(Transform):
    def log_abs_det_jacobian[S: IntTuple](
        self, x: Tensor[S], y: Tensor[S]
    ) -> Tensor[S]: ...

class SoftplusTransform(Transform):
    def log_abs_det_jacobian[S: IntTuple](
        self, x: Tensor[S], y: Tensor[S]
    ) -> Tensor[S]: ...

class TanhTransform(Transform):
    def log_abs_det_jacobian[S: IntTuple](
        self, x: Tensor[S], y: Tensor[S]
    ) -> Tensor[S]: ...

class LowerCholeskyTransform(Transform): ...
class PositiveDefiniteTransform(Transform): ...
class SoftmaxTransform(Transform): ...

class StickBreakingTransform(Transform):
    def __call__[S: IntTuple, N: IntVar](
        self, x: Tensor[[*S, N]]
    ) -> Tensor[[*S, N + 1]]: ...
    def log_abs_det_jacobian[S: IntTuple, N: IntVar](
        self, x: Tensor[[*S, N]], y: Tensor[[*S, N + 1]]
    ) -> Tensor[S]: ...

# TODO: Replace these availability stubs with shape-aware declarations.
AffineTransform: Any
CatTransform: Any
ComposeTransform: Any
CorrCholeskyTransform: Any
CumulativeDistributionTransform: Any
IndependentTransform: Any
PowerTransform: Any
ReshapeTransform: Any
StackTransform: Any
identity_transform: Any
