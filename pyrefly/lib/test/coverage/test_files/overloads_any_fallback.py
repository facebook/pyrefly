# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Regression test for https://github.com/facebook/pyrefly/issues/3257:
# A single @overload returning `Any` must not take precedence over the concrete return
# types of other overloads in the merged return type

from typing import Any, overload


@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
@overload
def f(x: object) -> Any: ...
def f(x):
    return x
