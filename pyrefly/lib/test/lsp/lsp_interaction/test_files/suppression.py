# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore
1 + ""

2 + ""  # pyrefly: ignore

3 + ""  # pyrefly: ignore[invalid-inheritance]

1 + 1  # pyrefly: ignore
1 + ""

# pyrefly: ignore
1 + 1
