# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypedDict

from typing_extensions import Unpack


class Options(TypedDict):
    name: str
    age: int


def func(**kwargs: Unpack[Options]) -> None:
    pass


checker = func
