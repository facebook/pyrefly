# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.optim module.
Optimization algorithms for training neural networks.
"""

from typing import Any, Callable, Iterator

from torch import Tensor
from torch.nn import Module

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "ASGD",
    "LBFGS",
]

# Base optimizer class
class Optimizer:
    """Base class for all optimizers."""

    def __init__(
        self, params: Iterator[Tensor] | list[dict[str, Any]], defaults: dict[str, Any]
    ) -> None: ...
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients."""
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state."""
        ...

# SGD optimizer
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
    ) -> None: ...

# Adam optimizer
class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...

# AdamW optimizer
class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None: ...

# RMSprop optimizer
class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

# Adagrad optimizer
class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

# Adadelta optimizer
class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

# Adamax optimizer
class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

# ASGD optimizer
class ASGD(Optimizer):
    """Averaged Stochastic Gradient Descent."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None: ...

# LBFGS optimizer
class LBFGS(Optimizer):
    """L-BFGS optimizer."""

    def __init__(
        self,
        params: Iterator[Tensor] | list[dict[str, Any]],
        lr: float = 1,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
    ) -> None: ...
