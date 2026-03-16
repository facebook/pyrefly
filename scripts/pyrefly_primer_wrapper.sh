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
#
# When PYREFLY_SCRIPTS_DIR is set, the wrapper also sets up project dependencies
# via setup_primer_deps.py, giving pyrefly access to third-party type stubs
# and reducing Unknown-type noise in primer results.

# The real pyrefly binary sits alongside this script with a "-real" suffix.
REAL_PYREFLY="$(dirname "$0")/pyrefly-real"

# Run init on the current working directory (mypy_primer sets cwd to the project dir).
# --non-interactive ensures no stdin prompts; exit code is ignored since init may
# legitimately fail (e.g., existing config) and we still want to proceed with check.
"$REAL_PYREFLY" init --non-interactive . >/dev/null 2>&1 || true

# When PYREFLY_SCRIPTS_DIR is set, set up project deps and extract site-package
# paths so pyrefly can resolve third-party imports.
#
# setup_primer_deps.py outputs one --site-package-path=... flag per line.
# We use readarray to safely handle paths containing spaces.
SITE_PATH_ARGS=()
if [ -n "$PYREFLY_SCRIPTS_DIR" ]; then
    PROJECT_NAME=$(basename "$(pwd)")
    if [ ! -d ".venv" ]; then
        echo "[wrapper] Installing deps for $PROJECT_NAME" >&2
        readarray -t SITE_PATH_ARGS < <(
            PYTHONPATH="$PYREFLY_SCRIPTS_DIR" python3 \
                "$PYREFLY_SCRIPTS_DIR/setup_primer_deps.py" "$PROJECT_NAME" 2>/dev/null
        ) || echo "[wrapper] Dep install failed for $PROJECT_NAME (non-fatal)" >&2
    else
        echo "[wrapper] Reusing existing .venv for $PROJECT_NAME" >&2
        if [ -f ".venv/bin/python" ]; then
            readarray -t SITE_PATH_ARGS < <(
                .venv/bin/python -c "
import site
for p in site.getsitepackages():
    print('--site-package-path=' + p)
"
            )
        fi
    fi
fi

# Forward all original arguments to the real pyrefly binary
exec "$REAL_PYREFLY" "$@" "${SITE_PATH_ARGS[@]}"
