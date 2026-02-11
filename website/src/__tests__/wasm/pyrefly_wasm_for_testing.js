/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// This is a stub to ensure that we can still resolve modules in website tests.
//
// Currently, you need to build the wasm and js binding and copy them over here to replace this file. We
// have a bash script to do this that should work; when you finish you need to revert the changes.
// ```
// cd pyrefly/website
// cp src/__tests__/wasm/pyrefly_wasm_for_testing.js src/__tests__/wasm/pyrefly_wasm_for_testing.js.bak
// bash scripts/build_wasm_for_test_for_sandcastle.sh
// yarn test
// mv src/__tests__/wasm/pyrefly_wasm_for_testing.js.bak src/__tests__/wasm/pyrefly_wasm_for_testing.js
// ```
// When you finish testing, you'll need to revert changes to this file before you commit.
module.exports = {};
