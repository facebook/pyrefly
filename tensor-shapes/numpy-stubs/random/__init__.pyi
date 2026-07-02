# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

from shape_extensions import Dim

from .. import dtype, float64, ndarray

@overload
def randn[N](d0: Dim[N], /) -> ndarray[tuple[Dim[N]], dtype[float64]]: ...
@overload
def randn[N, M](
    d0: Dim[N], d1: Dim[M], /
) -> ndarray[tuple[Dim[N], Dim[M]], dtype[float64]]: ...
