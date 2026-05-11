# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


class Config:
    def is_skipped(self, path: Path) -> bool:
        return str(path) == ""
