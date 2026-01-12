# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from editable_module import editable_function, EditableClass


def local_function():
    return "local"


def main():
    # This function call should be renamable because it's from an editable package
    editable_result = editable_function()

    local_result = local_function()

    obj = EditableClass()
    method_result = obj.method()

    return {
        "local": local_result,
        "editable": editable_result,
        "method": method_result,
    }
