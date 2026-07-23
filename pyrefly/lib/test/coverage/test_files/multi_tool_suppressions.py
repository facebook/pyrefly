# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

x = 1  # type: ignore[assignment]
y = 2  # pyright: ignore[reportGeneralClassIssue]
z = 3  # pyrefly: ignore[error-code]
w = 4  # pyre-ignore[7]
