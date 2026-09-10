# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# `x` merges a plain assignment with a `match` capture over the imported `json`.
# `involves_import` must not follow the `PatternCapture` through to `import json`
# (it is a definition boundary), so `x` is reported as an untyped variable rather
# than skipped the way an optional import is.

import json

if bool():
    x = 1
else:
    match json:
        case x:
            pass
