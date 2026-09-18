# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests that excluded functions and classes aren't reported as variables when a
# later module-level loop ends in `break`, which turns their bindings into forwards.

from typing import no_type_check, type_check_only


@no_type_check
def unchecked(x):
    return x


@type_check_only
class Internal: ...


while True:
    break
