# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import ClassVar

try:
    from pkg._impl import obj1  # noqa: F401
except ImportError:
    from pkg._fallback_impl import obj2  # noqa: F401


class Feature:
    try:
        MY_VAR: ClassVar[str] = os.environ["MY_VAR"]
    except KeyError:
        MY_VAR = ""
