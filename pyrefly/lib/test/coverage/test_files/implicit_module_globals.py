# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Regression test for gh-3505: end-of-loop Phi joins wrap CPython-injected module
# globals (__file__, __name__, etc.), so the report filter must skip them by name
# rather than by binding shape.

"""Test module."""

while True:
    try:
        pass
    except KeyboardInterrupt:
        break
