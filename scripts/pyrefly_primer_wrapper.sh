#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Wrapper script for mypy_primer that generates a Pyrefly config for the real
# pyrefly binary before forwarding arguments.
#
# mypy_primer calls: {pyrefly} check {paths} --summary=none --output-format min-text
# This wrapper intercepts that call, generates a config for the current
# directory, then forwards the original arguments to the real pyrefly binary.
#
# When PYREFLY_SCRIPTS_DIR is set, the wrapper also sets up project dependencies
# via setup_primer_deps.py, giving pyrefly access to third-party type stubs
# and reducing Unknown-type noise in primer results.

# The real pyrefly binary sits alongside this script with a "-real" suffix.
REAL_PYREFLY="$(dirname "$0")/pyrefly-real"

# mypy_primer runs old and new pyrefly concurrently (asyncio.gather) in the
# same project directory. Without serialization, the two wrapper instances race
# on venv creation. We use mkdir as a portable atomic lock to ensure only one
# instance performs shared dependency setup; the second waits, then reuses it.
#
# Config migration is intentionally not shared: a new pyrefly binary can emit
# config error codes that the old binary cannot parse. Each real binary writes a
# private generated config and checks with --config so primer compares checker
# behavior rather than config-parser compatibility.


# TODO/BE: cleaner solution is to run the setup/initialization steps once before running both old/new pyrefly binaries
LOCKDIR=".pyrefly_primer.lock"

if mkdir "$LOCKDIR" 2>/dev/null; then
    # We won the lock — perform shared dependency setup.
    if [ -n "$PYREFLY_SCRIPTS_DIR" ]; then
        PROJECT_NAME=$(basename "$(pwd)")
        if [ ! -d ".venv" ]; then
            PYTHONPATH="$PYREFLY_SCRIPTS_DIR" python3 \
                "$PYREFLY_SCRIPTS_DIR/setup_primer_deps.py" "$PROJECT_NAME" >/dev/null 2>&1 || true
        fi
    fi

    # Signal that setup is complete.
    touch "$LOCKDIR/done"
else
    # Another instance is running setup — wait for it to finish.
    while [ ! -f "$LOCKDIR/done" ]; do
        sleep 0.2
    done
fi

CONFIG_TAG=$(printf "%s" "$REAL_PYREFLY" | cksum | cut -d " " -f 1)
CONFIG_PATH=".pyrefly_primer_${CONFIG_TAG}.toml"
CONFIG_TMP="${CONFIG_PATH}.tmp"
if [ ! -f "$CONFIG_PATH" ]; then
    if "$REAL_PYREFLY" init --non-interactive --dry-run --print-config . >"$CONFIG_TMP" 2>/dev/null; then
        mv "$CONFIG_TMP" "$CONFIG_PATH"
    else
        rm -f "$CONFIG_TMP"
    fi
fi

# Extract site-package paths (read-only, safe to run after setup is complete).
SITE_PATH_ARGS=()
if [ -n "$PYREFLY_SCRIPTS_DIR" ] && [ -f ".venv/bin/python" ]; then
    readarray -t SITE_PATH_ARGS < <(
        .venv/bin/python -c "
import site
for p in site.getsitepackages():
    print('--site-package-path=' + p)
"
    )
fi

# Forward all original arguments to the real pyrefly binary.
# Normalize the per-binary config filename in output so that old and new
# binaries produce identical warnings and primer doesn't report spurious diffs.
if [ "$1" = "check" ] && [ -f "$CONFIG_PATH" ]; then
    "$REAL_PYREFLY" "$@" "${SITE_PATH_ARGS[@]}" --config "$CONFIG_PATH" 2>&1 \
        | sed "s|$CONFIG_PATH|pyrefly.toml|g"
    exit "${PIPESTATUS[0]}"
fi
exec "$REAL_PYREFLY" "$@" "${SITE_PATH_ARGS[@]}"
