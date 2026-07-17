# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Regression test for gh-4020: PEP 562 module hooks written as `def` must be
# excluded from coverage, like the assignment form already is.


def __getattr__(name):
    return None


def __dir__():
    return []


def visible() -> int:
    return 1
