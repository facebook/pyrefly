# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import library
import library as alias

value = library.Target()
alias_value = alias.Target()

def shadow(library):
    return library.Target
