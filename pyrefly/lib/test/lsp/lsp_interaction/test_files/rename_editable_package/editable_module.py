# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This simulates a module from an editable package
def editable_function():
    """Function from editable package that should be renamable"""
    return "editable"


class EditableClass:
    """Class from editable package that should be renamable"""

    def method(self):
        return "editable_method"
