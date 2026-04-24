# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Example:
    required: int
    default_none: str | None = field(default=None)
    default_factory: list[int] = field(default_factory=list)
    literal_default: int = 1
