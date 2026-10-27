# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from defs import greet


def farewell(message):
    return message


def call():
    return greet(name="Alice", message="Hello"), farewell(message="Goodbye")
