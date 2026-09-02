# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# `override` was added to `typing` in 3.12, so this is an error at the configured
# `python-version = "3.9"` and not an error at the 3.12 the client-provided
# interpreter reports. It is how this file tells the two versions apart.
from typing import override

# Resolvable only through the site packages that same interpreter reports, so an
# error here means the interpreter has not been applied yet.
from custom_module import CustomClass

print(CustomClass.custom_attr, override)
