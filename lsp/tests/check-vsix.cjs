/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const yauzl = require('yauzl');

/** Check the actual package, including hashes of both browser server assets. */
async function checkVsix(filename) {
  const files = new Map();
  await new Promise((resolve, reject) => {
    yauzl.open(filename, {lazyEntries: true}, (error, zip) => {
      if (error) { reject(error); return; }
      zip.on('error', reject);
      zip.on('end', resolve);
      zip.on('entry', entry => {
        if (!/^extension\/(package.json|dist\/(extension-web.js|pyrefly-web-worker.js|pyrefly_wasm_bg.wasm|web-manifest.json))$/.test(entry.fileName)) {
          zip.readEntry(); return;
        }
        zip.openReadStream(entry, (error, stream) => {
          if (error) { reject(error); return; }
          const chunks = [];
          stream.on('data', data => chunks.push(data));
          stream.on('error', reject);
          stream.on('end', () => {
            files.set(entry.fileName.replace('extension/', ''), Buffer.concat(chunks));
            zip.readEntry();
          });
        });
      });
      zip.readEntry();
    });
  });
  for (const name of ['package.json', 'dist/extension-web.js', 'dist/pyrefly-web-worker.js',
    'dist/pyrefly_wasm_bg.wasm', 'dist/web-manifest.json']) {
    assert.ok(files.has(name), 'Missing browser package asset: ' + name);
  }
  assert.ok(JSON.parse(files.get('package.json')).browser);
  const manifest = JSON.parse(files.get('dist/web-manifest.json'));
  for (const name of ['pyrefly-web-worker.js', 'pyrefly_wasm_bg.wasm']) {
    assert.equal(createHash('sha256').update(files.get('dist/' + name)).digest('hex'), manifest[name]);
  }
  await WebAssembly.compile(files.get('dist/pyrefly_wasm_bg.wasm'));
}

module.exports = {checkVsix};
if (require.main === module) {
  checkVsix(process.argv[2]).catch(error => { console.error(error); process.exitCode = 1; });
}
