# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class A:
    x: int | str


class B(A):
    x: int  # pyrefly: ignore[bad-override]
