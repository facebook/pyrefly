/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

export default async function init() {
  throw new Error(
    'Pyrefly WASM bundle not found. Run pyrefly_wasm/build.sh and copy pyrefly_wasm.js into lsp/wasm.',
  );
}

export class State {
  constructor() {
    throw new Error(
      'Pyrefly WASM bundle not found. Run pyrefly_wasm/build.sh and copy pyrefly_wasm.js into lsp/wasm.',
    );
  }
}
