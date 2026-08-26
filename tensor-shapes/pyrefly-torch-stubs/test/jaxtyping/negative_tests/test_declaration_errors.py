# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative test: shape strings that a declaration cannot satisfy.

The grammar itself is unit tested in `pyrefly/lib/alt/jaxtyping.rs`. These cases
check that the resulting diagnostics reach the user, which a parser test cannot.
"""

from typing import TYPE_CHECKING

from shape_extensions import static_jaxtyping

if TYPE_CHECKING:
    from jaxtyping import Shaped
    from torch import Tensor


@static_jaxtyping("*batch")
# E: Tensor shape can have at most one variadic dimension
def two_variadics(x: Shaped[Tensor, "*batch ... 3"]) -> None:
    """A shape has one unpacked segment, so two variadics cannot be divided."""
    pass


@static_jaxtyping("*batch")
# E: `batch` is used as a dimension but declared as a variadic shape
def variadic_used_as_dimension(x: Shaped[Tensor, "batch 3"]) -> None:
    """A name keeps the arity it was declared with."""
    pass


@static_jaxtyping("n")
# E: `n` is used as a variadic shape but declared as a dimension
def dimension_used_as_variadic(x: Shaped[Tensor, "*n 3"]) -> None:
    pass
