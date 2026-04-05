# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import overload


@overload
def process(x: int) -> int: ...


@overload
def process(x: str) -> str: ...


def process(x):
    return x


@typing.overload
def convert(x: int) -> str: ...


@typing.overload
def convert(x: str) -> int: ...


def convert(x):
    return x


def normal_function(x: int) -> int:
    return x
