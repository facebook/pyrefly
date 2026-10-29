/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

const directory = path.join(__dirname, 'target');
const manifest = {};
for (const name of ['pyrefly_wasm.js', 'pyrefly_wasm_bg.wasm']) {
  manifest[name] = createHash('sha256').update(fs.readFileSync(path.join(directory, name))).digest('hex');
}
fs.writeFileSync(path.join(directory, 'web-manifest.json'), JSON.stringify(manifest));
