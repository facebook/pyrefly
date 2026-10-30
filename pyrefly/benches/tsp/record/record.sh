#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Record the TSP requests that Pylance sends to pyrefly for the features that
# `benches/tsp/pytorch.rs` replays, and write them to `benches/tsp/traces/`,
# one file per feature.
#
# Usage: record.sh <pyrefly binary> <PyTorch checkout at the pinned revision>
#
# Runs the latest stable VS Code and Pylance headless on Linux. Needs node and
# internet access to download them (on a devserver, set HTTPS_PROXY to fwdproxy).

set -euo pipefail

PYREFLY=$(realpath "$1")
PYTORCH=$(realpath "$2")
HERE=$(dirname "$(realpath "$0")")
WORK=$(mktemp -d)
echo "Working in $WORK"

curl -sSL -o "$WORK/vscode.tar.gz" https://update.code.visualstudio.com/latest/linux-x64/stable
tar -xzf "$WORK/vscode.tar.gz" -C "$WORK"
DIRS=(--user-data-dir "$WORK/user" --extensions-dir "$WORK/extensions")
"$WORK/VSCode-linux-x64/bin/code" "${DIRS[@]}" --install-extension ms-python.python --install-extension ms-python.vscode-pylance
"$WORK/VSCode-linux-x64/bin/code" "${DIRS[@]}" --list-extensions --show-versions

# Pylance uses pyrefly only alongside the Python extension, which it installs
# above. Pylance launches the type server as `<pyrefly> tsp ...`. The wrapper records
# pyrefly's log and the main connection's requests, which carry every
# foreground request.
cat > "$WORK/pyrefly" <<WRAPPER
#!/bin/bash
if [ "\$1" = tsp ]; then
  tee -a "$WORK/main_in.log" | "$PYREFLY" "\$@" 2>> "$WORK/pyrefly.log"
else
  exec "$PYREFLY" "\$@"
fi
WRAPPER
chmod +x "$WORK/pyrefly"
touch "$WORK/pyrefly.log" "$WORK/main_in.log"

mkdir -p "$WORK/user/User"
cat > "$WORK/user/User/settings.json" <<SETTINGS
{
  "python.analysis.diagnosticsSource": "Pylance + Pyrefly",
  "python.analysis.pyreflyVersion": "$WORK/pyrefly",
  "python.analysis.logTypeServerMessages": true,
  "python.analysis.enablePyreflyNotification": false,
  "security.workspace.trust.enabled": false,
  "telemetry.telemetryLevel": "off",
  "extensions.autoUpdate": false,
  "update.mode": "none"
}
SETTINGS

RECORD_WORK="$WORK" RECORD_PYTORCH="$PYTORCH" "$WORK/VSCode-linux-x64/code" "${DIRS[@]}" \
  --extensionDevelopmentPath="$HERE/runner" \
  --extensionTestsPath="$HERE/runner/index.js" \
  --ozone-platform=headless --disable-gpu --no-sandbox --disable-workspace-trust \
  "$PYTORCH" > "$WORK/vscode.log" 2>&1

python3 "$HERE/extract.py" "$WORK" "$PYTORCH" "$HERE/../traces"
