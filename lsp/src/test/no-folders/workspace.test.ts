/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {resolveExecutable} from '../../lspPath';
import {fakeEnvironment, recordingChannel} from '../helpers';

/**
 * Its own `.vscode-test.mjs` configuration, with no `workspaceFolder` at all,
 * so VS Code opens an empty window. Selection still has to produce something
 * sensible when there is no folder to resolve against.
 */
suite('workspace with no folders', () => {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  const bundled = vscode.Uri.joinPath(
    extension?.extensionUri ?? vscode.Uri.file(''),
    'bin',
    process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly',
  ).fsPath;

  async function setting(key: string, value: unknown): Promise<void> {
    await vscode.workspace
      .getConfiguration()
      .update(key, value, vscode.ConfigurationTarget.Global);
  }

  /** Resolve, reporting the scopes the interpreter was looked up for. */
  async function resolve(interpreter?: string) {
    const scopes: (vscode.Uri | undefined)[] = [];
    const spec = await resolveExecutable(
      extension!.extensionUri,
      fakeEnvironment(interpreter, scopes),
      recordingChannel([]),
    );
    return {spec, scopes};
  }

  setup(function () {
    // An empty window is the whole premise, so fail rather than pass vacuously.
    assert.ok(extension, 'extension not found');
    assert.strictEqual(
      (vscode.workspace.workspaceFolders ?? []).length,
      0,
      'expected a window with no folders open',
    );
  });

  teardown(async () => {
    await setting('pyrefly.lspPath', undefined);
    await setting('pyrefly.pyreflyExecutable', undefined);
  });

  test('a relative lspPath is left as written', async () => {
    // There is no root to resolve against, so the value is passed through
    // rather than being anchored to some arbitrary directory such as the
    // extension host's cwd.
    await setting('pyrefly.lspPath', './bin/pyrefly');
    const {spec} = await resolve();
    assert.strictEqual(spec.command, './bin/pyrefly');
  });

  test('the interpreter is still consulted, with no folder scope', async () => {
    await setting('pyrefly.pyreflyExecutable', 'from-environment');
    const {scopes} = await resolve();
    assert.strictEqual(scopes.length, 1, 'expected a single interpreter lookup');
    assert.strictEqual(scopes[0], undefined);
  });

  test('selection still yields the bundled binary', async () => {
    await setting('pyrefly.pyreflyExecutable', 'from-environment');
    const {spec} = await resolve();
    assert.strictEqual(spec.command, bundled);
    assert.deepStrictEqual(spec.args, ['lsp']);
  });
});
