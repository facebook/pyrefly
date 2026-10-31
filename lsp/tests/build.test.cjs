/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {spawn, spawnSync} = require('node:child_process');
const {test} = require('node:test');

/** Build fixture inputs have different markers in each JS/WASM generation. */
function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pyrefly-web-build-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true, maxRetries: 3, retryDelay: 100}));
  const root = path.join(directory, 'lsp');
  const inputs = path.join(directory, 'pyrefly_wasm/target');
  fs.mkdirSync(path.join(root, 'src/web-test'), {recursive: true});
  fs.mkdirSync(inputs, {recursive: true});
  fs.symlinkSync(path.resolve(__dirname, '../node_modules'), path.join(root, 'node_modules'), 'junction');
  fs.copyFileSync(path.resolve(__dirname, '../esbuild.js'), path.join(root, 'esbuild.js'));
  fs.writeFileSync(path.join(root, 'src/extension-web.ts'), 'export const activate = () => {};');
  fs.writeFileSync(path.join(root, 'src/web-test/suite.ts'), 'export const run = () => {};');
  fs.writeFileSync(path.join(root, 'src/pyrefly-web-worker.ts'),
    "import {marker} from '../wasm/pyrefly_wasm'; self.postMessage(marker);");
  function generate(marker) {
    fs.writeFileSync(path.join(inputs, 'pyrefly_wasm.js'), 'export const marker = ' + JSON.stringify(marker));
    fs.writeFileSync(path.join(inputs, 'pyrefly_wasm_bg.wasm'), marker);
    const manifest = {};
    for (const name of ['pyrefly_wasm.js', 'pyrefly_wasm_bg.wasm']) {
      manifest[name] = createHash('sha256').update(fs.readFileSync(path.join(inputs, name))).digest('hex');
    }
    fs.writeFileSync(path.join(inputs, 'web-manifest.json'), JSON.stringify(manifest));
  }
  return {root, inputs, generate};
}

test('one build consumes matching regenerated bindings and rejects incomplete inputs', t => {
  const f = fixture(t);
  const build = () => spawnSync(process.execPath, ['esbuild.js', '--web-only'], {cwd: f.root, encoding: 'utf8'});
  f.generate('first-generation');
  assert.equal(build().status, 0);
  f.generate('second-generation');
  assert.equal(build().status, 0);
  assert.match(fs.readFileSync(path.join(f.root, 'dist/pyrefly-web-worker.js'), 'utf8'), /second-generation/);
  assert.equal(fs.readFileSync(path.join(f.root, 'dist/pyrefly_wasm_bg.wasm'), 'utf8'), 'second-generation');
  fs.writeFileSync(path.join(f.inputs, 'pyrefly_wasm.js'), 'export const marker = "unpaired";');
  assert.notEqual(build().status, 0);
  fs.unlinkSync(path.join(f.inputs, 'pyrefly_wasm_bg.wasm'));
  assert.notEqual(build().status, 0);
});

test('watch builds consume a regenerated pair on the first rebuild', {timeout: 15000}, async t => {
  let child;
  t.after(async () => {
    if (child && child.exitCode === null) {
      const closed = new Promise(resolve => child.once('exit', resolve));
      child.kill();
      await closed;
    }
  });
  const f = fixture(t);
  f.generate('first-generation');
  child = spawn(process.execPath, ['esbuild.js', '--web-only', '--watch'], {cwd: f.root});
  let output = '';
  child.stderr.on('data', data => { output += data; });
  child.stdout.on('data', data => { output += data; });
  async function waitFor(marker) {
    const deadline = Date.now() + 10000;
    const outputPath = path.join(f.root, 'dist/pyrefly-web-worker.js');
    while (!fs.existsSync(outputPath) || !fs.readFileSync(outputPath, 'utf8').includes(marker)) {
      assert.ok(Date.now() < deadline, output);
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  await waitFor('first-generation');
  f.generate('second-generation');
  await waitFor('second-generation');
  assert.equal(fs.readFileSync(path.join(f.root, 'dist/pyrefly_wasm_bg.wasm'), 'utf8'), 'second-generation');
});
