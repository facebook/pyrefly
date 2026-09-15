# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

# Variable annotated with Any: typed for coverage, untyped for strict_coverage
x: Any = 1

# Variable with concrete type (baseline)
y: int = 2


# Function with Any params and return
def func_any(a: Any, b: Any) -> Any:
    return a


# Function with mixed Any and concrete types
def func_mixed(a: int, b: Any) -> str:
    return ""


# Unannotated variable (baseline for comparison)
z = 42
