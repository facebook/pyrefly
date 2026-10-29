/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {Worker} = require('node:worker_threads');
const {test} = require('node:test');

/** Run the actual bundled browser transport and WASM with a minimal Worker shim. */
async function server(t, projects = []) {
  const worker = new Worker(`
    const {parentPort, workerData} = require('node:worker_threads');
    globalThis.self = globalThis;
    globalThis.addEventListener = (name, listener) => {
      if (name === 'message') parentPort.on('message', data => listener({data}));
    };
    globalThis.postMessage = message => parentPort.postMessage(message);
    parentPort.on('message', data => globalThis.onmessage?.({data}));
    require('node:vm').runInThisContext(require('node:fs').readFileSync(workerData, 'utf8'));
  `, {eval: true, workerData: path.resolve(__dirname, '../dist/pyrefly-web-worker.js')});
  t.after(() => worker.terminate());
  let nextId = 1;
  const pending = new Map();
  const disk = new Map();
  const diagnostics = new Map();
  const reads = [];
  worker.on('message', message => {
    if (message.method === 'pyrefly/readWorkspaceFile') {
      reads.push(message.params);
      let result = null;
      for (const filename of message.params.filenames) {
        const uri = message.params.rootUri.replace(/\/$/, '') + '/' +
          filename.split('/').map(encodeURIComponent).join('/');
        if (disk.has(uri)) {
          result = {filename, uri, content: disk.get(uri)};
          break;
        }
      }
      worker.postMessage({jsonrpc: '2.0', id: message.id, result});
    } else if (message.method === 'textDocument/publishDiagnostics') {
      diagnostics.set(message.params.uri, message.params.diagnostics);
    } else if (!message.method && pending.has(message.id)) {
      const {resolve, reject} = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(message.error.message)) : resolve(message.result);
    }
  });
  worker.on('error', error => {
    for (const {reject} of pending.values()) { reject(error); }
    pending.clear();
  });
  function request(method, params) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, {resolve, reject});
      worker.postMessage({jsonrpc: '2.0', id, method, params});
    });
  }
  function notify(method, params) {
    worker.postMessage({jsonrpc: '2.0', method, params});
  }
  await request('initialize', {
    capabilities: {}, initializationOptions: {
      projects, pythonVersion: '3.12',
      wasmBytes: fs.readFileSync(path.resolve(__dirname, '../dist/pyrefly_wasm_bg.wasm')),
    },
  });
  return {
    request, notify, disk, diagnostics, reads,
    open(uri, text) {
      notify('textDocument/didOpen', {textDocument: {uri, text, languageId: 'python', version: 1}});
    },
    async hover(uri, line, character) {
      return request('textDocument/hover', {textDocument: {uri}, position: {line, character}});
    },
    async change(uri, text) {
      notify('textDocument/didChange', {
        textDocument: {uri, version: 2}, contentChanges: [{text}],
      });
      await this.hover(uri, 0, 0);
    },
  };
}

const project = (uri, pythonVersion = '3.12', files = {}) => ({uri, pythonVersion, files});

test('original URIs and module state stay separate across roots and untitled documents', {timeout: 30000}, async t => {
  const s = await server(t, [project('file:///A'), project('file:///B')]);
  s.disk.set('file:///A/utils.py', 'value: int = 1\n');
  s.disk.set('file:///B/utils.py', 'value: str = "x"\n');
  s.open('file:///A/main.py', 'from utils import value\nvalue\n');
  s.open('file:///B/main.py', 'from utils import value\nvalue\n');
  assert.match(JSON.stringify(await s.hover('file:///A/main.py', 1, 2)), /int/);
  assert.match(JSON.stringify(await s.hover('file:///B/main.py', 1, 2)), /str/);
  for (const root of ['A', 'B']) {
    const definitions = await s.request('textDocument/definition', {
      textDocument: {uri: 'file:///' + root + '/main.py'}, position: {line: 1, character: 2},
    });
    assert.equal(definitions[0].uri, 'file:///' + root + '/utils.py');
  }
  for (const uri of ['untitled:Untitled-1', 'file:///outside.py', 'file:///AB/main.py']) {
    s.open(uri, 'target: int = "bad"\n');
    assert.ok(await s.hover(uri, 0, 2));
    assert.ok(s.diagnostics.get(uri).some(item => item.code === 'bad-assignment'));
  }
  assert.equal(await s.hover('file:///not-open.py', 0, 0), null);
  assert.equal(await s.hover('vscode-notebook-cell:/cell.py', 0, 0), null);
});

test('UTF-16 positions and LSP completion edit shapes survive the worker boundary', {timeout: 30000}, async t => {
  const s = await server(t, [project('file:///workspace%20name')]);
  const uri = 'file:///workspace%20name/a%20b.py';
  const text = 's = "😀"; target: int = "bad"; count = len(s); tar\ntarget\nreveal_type';
  s.open(uri, text);
  assert.ok(await s.hover(uri, 0, text.indexOf('target') + 2));
  const diagnostic = s.diagnostics.get(uri).find(item => item.code === 'bad-assignment');
  assert.ok(diagnostic);
  assert.ok(diagnostic.range.start.character >= text.indexOf('target'));
  const definitions = await s.request('textDocument/definition', {
    textDocument: {uri}, position: {line: 1, character: 2},
  });
  assert.equal(definitions[0].uri, uri);
  assert.equal(definitions[0].range.start.character, text.indexOf('target'));
  const items = await s.request('textDocument/completion', {
    textDocument: {uri}, position: {line: 2, character: 11},
  });
  const item = items.find(item => item.label === 'reveal_type');
  assert.equal(item.kind, 3); // LSP Function, not Monaco Function.
  assert.equal(item.additionalTextEdits[0].newText, 'from typing import reveal_type\n');
  assert.deepEqual(item.additionalTextEdits[0].range.start, {line: 0, character: 0});
  const localItems = await s.request('textDocument/completion', {
    textDocument: {uri}, position: {line: 0, character: text.split('\n')[0].length},
  });
  assert.ok(localItems.some(item => item.label === 'target'));
  const hints = await s.request('textDocument/inlayHint', {
    textDocument: {uri}, range: {start: {line: 0, character: 0}, end: {line: 3, character: 0}},
  });
  assert.ok(hints.some(hint => hint.position.line === 0 &&
    hint.position.character === text.indexOf('count') + 'count'.length), JSON.stringify(hints));
});

test('suppressed imports load and filesystem events invalidate positive and negative caches', {timeout: 30000}, async t => {
  const s = await server(t, [project('file:///workspace')]);
  const uri = 'file:///workspace/main.py';
  const source = 'from utils import value  # pyrefly: ignore[missing-import]\nvalue\n';
  s.open(uri, source);
  await s.hover(uri, 1, 2);
  const firstReads = s.reads.length;
  for (let i = 0; i < 3; i++) { await s.change(uri, source + '# edit ' + i); }
  assert.equal(s.reads.length, firstReads);
  const dependency = 'file:///workspace/utils.py';
  for (const [type, text] of [[1, 'value: int = 1'], [2, 'value: str = "x"'], [3, null]]) {
    text === null ? s.disk.delete(dependency) : s.disk.set(dependency, text);
    s.notify('workspace/didChangeWatchedFiles', {changes: [{uri: dependency, type}]});
    const hover = JSON.stringify(await s.hover(uri, 1, 2));
    if (type === 1) { assert.match(hover, /int/); }
    if (type === 2) { assert.match(hover, /str/); }
    if (type === 3) { assert.doesNotMatch(hover, /str/); }
  }
});

test('session and TOML Python versions survive document updates', {timeout: 30000}, async t => {
  const source = 'import sys\nif sys.version_info >= (3, 12):\n    value: int = "bad"\nelse:\n    value: int = 1\n';
  const s = await server(t, [
    project('file:///setting', '3.11'),
    project('file:///toml', '3.11', {'pyrefly.toml': 'python-version = "3.12"'}),
    project('file:///fallback'),
  ]);
  for (const root of ['setting', 'toml', 'fallback']) {
    const uri = 'file:///' + root + '/main.py';
    s.open(uri, source);
    await s.change(uri, source + '# changed\n');
    const hasError = s.diagnostics.get(uri).some(item => item.code === 'bad-assignment');
    assert.equal(hasError, root !== 'setting', root);
  }
});

test('invalid editor versions report configuration errors', {timeout: 30000}, async t => {
  await assert.rejects(server(t, [project('file:///invalid', 'invalid-version')]),
    /Invalid Pyrefly Python version/);
});

test('LSP completion kinds identify classes, functions, and methods', {timeout: 30000}, async t => {
  const s = await server(t);
  const uri = 'untitled:Kinds';
  const prefix = 'class Widget:\n    def method(self) -> int: return 1\n\ndef function() -> int: return 1\n\n';
  s.open(uri, prefix + 'Wid');
  const complete = (line, character) => s.request('textDocument/completion', {
    textDocument: {uri}, position: {line, character},
  });
  assert.equal((await complete(5, 3)).find(item => item.label === 'Widget').kind, 7);
  await s.change(uri, prefix + 'func');
  const functions = (await complete(5, 4)).filter(item => item.label === 'function');
  assert.ok(functions.some(item => item.kind === 3), JSON.stringify(functions));
  await s.change(uri, prefix + 'obj = Widget()\nobj.me');
  assert.equal((await complete(6, 6)).find(item => item.label === 'method').kind, 2);
});

test('absolute imports use the root and relative imports use the importer package', {timeout: 30000}, async t => {
  const s = await server(t, [project('file:///workspace')]);
  s.disk.set('file:///workspace/utils.py', 'value: int = 1');
  s.disk.set('file:///workspace/pkg/__init__.py', '');
  s.disk.set('file:///workspace/pkg/utils.py', 'value: str = "x"');
  const uri = 'file:///workspace/pkg/main.py';
  s.open(uri, 'from utils import value\nvalue');
  assert.match(JSON.stringify(await s.hover(uri, 1, 2)), /int/);
  await s.change(uri, 'from .utils import value\nvalue');
  assert.match(JSON.stringify(await s.hover(uri, 1, 2)), /str/);
});
