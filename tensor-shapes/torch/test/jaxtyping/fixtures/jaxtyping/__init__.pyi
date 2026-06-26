# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Stub file for jaxtyping dtype wrappers. Real jaxtyping exposes these as
# Annotated aliases to static type checkers. Pyrefly treats the wrappers as
# shape-syntax markers only and does not model dtype refinements.

from typing import (
    Annotated as BFloat16,
    Annotated as Bool,
    Annotated as Complex,
    Annotated as Complex128,
    Annotated as Complex64,
    Annotated as Float,
    Annotated as Float16,
    Annotated as Float32,
    Annotated as Float64,
    Annotated as Inexact,
    Annotated as Int,
    Annotated as Int16,
    Annotated as Int32,
    Annotated as Int64,
    Annotated as Int8,
    Annotated as Integer,
    Annotated as Key,
    Annotated as Num,
    Annotated as Real,
    Annotated as Shaped,
    Annotated as UInt,
    Annotated as UInt16,
    Annotated as UInt32,
    Annotated as UInt64,
    Annotated as UInt8,
)
