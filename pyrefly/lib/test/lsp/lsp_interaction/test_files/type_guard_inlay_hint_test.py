# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing_extensions import TypeGuard


def is_str(val: object) -> TypeGuard[str]:
    return isinstance(val, str)


checker = is_str
