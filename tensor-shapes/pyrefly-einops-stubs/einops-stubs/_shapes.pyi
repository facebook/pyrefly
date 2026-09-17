# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, IntTuples, type_shape_dsl_function

@type_shape_dsl_function
def rearrange_shape(pattern: str, shape: IntTuple) -> IntTuple:
    return dsl.rearrange(pattern, shape)

@type_shape_dsl_function
def reduce_shape(pattern: str, shape: IntTuple) -> IntTuple:
    return dsl.reduce(pattern, shape)

@type_shape_dsl_function
def repeat_shape(pattern: str, shape: IntTuple) -> IntTuple:
    return dsl.repeat(pattern, shape)

@type_shape_dsl_function
def einsum_shape(pattern: str, shapes: IntTuples) -> IntTuple:
    return dsl.einops_einsum(pattern, shapes)
