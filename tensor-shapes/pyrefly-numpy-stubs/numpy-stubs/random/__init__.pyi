# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

from shape_extensions import Dim, SymVar

from .. import dtype, float64, ndarray

@overload
def randn[N: SymVar](d0: Dim[N], /) -> ndarray[[N], dtype[float64]]: ...
@overload
def randn[N: SymVar, M: SymVar](
    d0: Dim[N], d1: Dim[M], /
) -> ndarray[[N, M], dtype[float64]]: ...
