# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

from jax._array import Array as Array

from . import lax as lax, nn as nn, numpy as numpy

tree: Any
tree_util: Any

def jit[F: Callable[..., Any]](fun: F, *args: Any, **kwargs: Any) -> F: ...
def grad(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]: ...
def vmap(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]: ...
