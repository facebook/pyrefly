# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def transform(x: int, s: str) -> bool:
    return bool(x) and bool(s)


def variadic(*args: int) -> str:
    return ""


# Module-level function-valued variables: their inferred types are callables
# and must be rendered as valid `typing.Callable`, not pyrefly's internal
# `(args) -> ret` display form.
handler = transform
collect = variadic
