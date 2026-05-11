# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing_extensions import deprecated


@deprecated("use new_function instead")
def old_function() -> None: ...


old_function()
