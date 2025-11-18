# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from third_party_lib import external_function


def local_function():
    return "local"


def main():
    local_result = local_function()
    external_result = external_function()

    return {
        "local": local_result,
        "external": external_result,
    }
