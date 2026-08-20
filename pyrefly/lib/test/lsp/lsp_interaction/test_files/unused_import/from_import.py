# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List  # noqa: F401


def process(items: List[str]):
    return [item.upper() for item in items]
