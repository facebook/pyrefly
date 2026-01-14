#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script is used by Sandcastle (Meta's internal CI) to build the WASM for
# the test page, and it can also be useful for testing changes to the website
# code as an open-source developer.

# Fail if we have any errors
set -e
set -o pipefail

# Change to `pyrefly` directory
cd -- "$(dirname -- "$0")/.."

# shellcheck source=/dev/null
source scripts/setup_cargo.sh

cargo install wasm-pack wasm-opt
scripts/build_wasm_for_test.sh
