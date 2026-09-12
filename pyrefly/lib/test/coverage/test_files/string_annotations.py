# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated

# String annotation resolved to concrete type
x: "int" = 1


# String annotation on function params and return
def func_str_ann(a: "int", b: "str") -> "bool":
    return True


# Annotated unwrapping: inner type should be extracted
y: Annotated[int, "metadata"] = 2
