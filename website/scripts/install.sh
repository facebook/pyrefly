#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Scripts that install all dependencies for the Pyre website
# This includes:
# - yarn
# - cargo
# - wasm

# Fail if we have any errors
set -e

# Change to `pyrefly` directory
cd -- "$(dirname -- "$0")/.."

# Merge stdout and stderr logs because Sandcastle shows them separately.
exec >&2

# shellcheck source=/dev/null
source ../facebook/setup_cargo.sh

cargo install wasm-pack wasm-opt

yarn install
