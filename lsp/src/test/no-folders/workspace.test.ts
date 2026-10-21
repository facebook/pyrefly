/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import {tmpdir} from 'os';
import {join} from 'path';
import * as vscode from 'vscode';
import {resolveExecutable} from '../../lsp-path';
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

  /**
   * Resolve, reporting the scopes the interpreter was looked up for. Omitting
   * `folders` resolves against the open window, which has none.
   */
  async function resolve(
    interpreter?: string,
    folders?: readonly vscode.WorkspaceFolder[],
  ) {
    const scopes: (vscode.Uri | undefined)[] = [];
    const spec = await resolveExecutable(
      extension!.extensionUri,
      fakeEnvironment(interpreter, scopes),
      recordingChannel([]),
      folders,
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

  /**
   * The multi-root suite covers this transition from a synthesized empty
   * folder list; here the window really is empty, so it also pins down that
   * the two agree. The new shape is handed to selection directly because
   * `updateWorkspaceFolders` terminates and restarts the extension host — and
   * with it the test runner — when the first folder is added. The extension
   * does not listen for folder changes either, so this describes what the next
   * resolution picks rather than a restart.
   */
  test('adding the first folder gives the lookup a root', async () => {
    await setting('pyrefly.pyreflyExecutable', 'from-environment');
    const added: vscode.WorkspaceFolder = {
      uri: vscode.Uri.file(join(tmpdir(), 'pyrefly-first-root')),
      name: 'first',
      index: 0,
    };

    const empty = await resolve();
    assert.deepStrictEqual(empty.scopes, [undefined]);

    const opened = await resolve(undefined, [added]);
    assert.deepStrictEqual(opened.scopes.map(String), [added.uri.toString()]);
    // The folder has no interpreter behind it, so the binary is unchanged;
    // what moved is which root the next resolution asks about.
    assert.strictEqual(opened.spec.command, bundled);
  });
});
