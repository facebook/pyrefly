#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -u

label="${1:?Usage: $0 <test-label>}"
attempts=3
marker=
marker_for_node=
rc=
for signal in 1 2 3 13 15; do
  eval "trap 'exit $((signal + 128))' $signal"
done
trap 'rc=$?; set +e; rm -f "$marker"; exit $rc' EXIT
marker="$(mktemp)" || exit 1
marker_for_node="$marker"
if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
  marker_for_node="$(cygpath -w "$marker")" || exit 1
fi

command=(npm run test -- --label "$label")
if [[ "${RUNNER_OS:-}" == "Linux" ]]; then
  command=(xvfb-run -a "${command[@]}")
fi

for ((attempt = 1; attempt <= attempts; attempt++)); do
  rm -f "$marker"
  PYREFLY_EXTENSION_TEST_STARTED="$marker_for_node" "${command[@]}"
  status=$?
  if ((status == 0)); then
    exit 0
  fi
  if ((status != 1 || attempt == attempts)) || [[ -e "$marker" ]]; then
    exit "$status"
  fi

  message="VS Code exited before the $label test suite started (attempt $attempt of $attempts); retrying"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::warning title=VS Code startup failed::$message"
  else
    echo "$message" >&2
  fi
done
