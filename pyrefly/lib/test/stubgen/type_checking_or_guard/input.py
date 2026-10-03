# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

USE_EXTENSIONS = True


class Guards:
    if TYPE_CHECKING or not USE_EXTENSIONS:
        from pkg._impl import LeftOrImport

    if not USE_EXTENSIONS or TYPE_CHECKING:
        from pkg._impl import RightOrImport

    if TYPE_CHECKING and USE_EXTENSIONS:
        from pkg._impl import AndImport
