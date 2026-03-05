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

# Run init on the current working directory (mypy_primer sets cwd to the project dir).
# --non-interactive ensures no stdin prompts; exit code is ignored since init may
# legitimately fail (e.g., existing config) and we still want to proceed with check.
"$REAL_PYREFLY" init --non-interactive . >/dev/null 2>&1 || true

# Forward all original arguments to the real pyrefly binary
exec "$REAL_PYREFLY" "$@"
