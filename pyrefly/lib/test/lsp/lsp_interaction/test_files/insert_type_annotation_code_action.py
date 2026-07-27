# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def make_items():
    return [1, 2, 3]


items = make_items()


class Box:
    def __init__(self):
        self.items = make_items()


left, right = (items, items)


def consume(value: list[int]) -> None:
    pass


consume(items)


annotated: list[int] = make_items()
