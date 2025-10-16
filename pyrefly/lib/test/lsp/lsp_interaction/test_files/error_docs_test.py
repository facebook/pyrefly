# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Test file for error documentation links
# This file contains various error types to test that documentation links are properly included

# Bad assignment error
x: int = ""


# Bad context manager error
class A:
    pass


with A():
    pass

# Missing attribute error
obj = object()
obj.nonexistent_method()
