# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

from shape_extensions import Int, IntVar

from .. import dtype, float64, ndarray

@overload
def randn[N: IntVar](d0: Int[N], /) -> ndarray[[N], dtype[float64]]: ...
@overload
def randn[N: IntVar, M: IntVar](
    d0: Int[N], d1: Int[M], /
) -> ndarray[[N, M], dtype[float64]]: ...
