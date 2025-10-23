# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from base_module import (
    directly_defined_function as reexported_function,  # noqa: F401
    DirectlyDefinedClass as ReexportedClass,  # noqa: F401
)


def another_direct_function() -> None:
    pass


class AnotherDirectClass:
    pass
