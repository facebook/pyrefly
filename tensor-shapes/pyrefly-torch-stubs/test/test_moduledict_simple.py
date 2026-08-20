# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test to debug ModuleDict with TypedDict
"""

from typing import TypedDict

import torch.nn as nn


class MyModules(TypedDict):
    linear: nn.Linear


# Just test the type variable inference
modules: MyModules = dict(linear=nn.Linear(10, 5))
module_dict = nn.ModuleDict(modules)

# Try to access - should be nn.Linear, not Module
result: nn.Linear = module_dict.linear
