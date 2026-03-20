#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Wrapper script for mypy_primer that runs `pyrefly init --non-interactive`
# before forwarding arguments to the real pyrefly binary.
#
# mypy_primer calls: {pyrefly} check {paths} --summary=none --output-format min-text
# This wrapper intercepts that call, runs init on the current directory first,
# then forwards the original arguments to the real pyrefly binary.

# The real pyrefly binary sits alongside this script with a "-real" suffix.
REAL_PYREFLY="$(dirname "$0")/pyrefly-real"

# Run init if no config exists yet.
if [ ! -f "pyrefly.toml" ] && ! grep -q '\[tool\.pyrefly\]' pyproject.toml 2>/dev/null; then
    "$REAL_PYREFLY" init --non-interactive . >/dev/null 2>&1 || true
fi

# Forward all original arguments to the real pyrefly binary.
exec "$REAL_PYREFLY" "$@"
