# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# gh-3773: optional imports are imports, not variables, and must not be reported.

try:
    import json
except ImportError:
    json = None

try:
    from os import path
except ImportError:
    path = None

# Non-import flow merge: still untyped.
if bool():
    w = list()
else:
    w = dict()
