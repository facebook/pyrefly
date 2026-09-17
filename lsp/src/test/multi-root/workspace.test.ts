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
import {resolveExecutable} from '../../lspPath';
import {PythonEnvironment} from '../../python-environment';
import {recordingChannel} from '../helpers';

/**
 * These run in their own `.vscode-test.mjs` configuration, against the
 * two-folder fixture workspace. The rest of the suite stays single-root.
 */
suite('multi-root workspace', () => {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  const folders = vscode.workspace.workspaceFolders ?? [];
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

  const setLspPath = (value: string | undefined) =>
    setting('pyrefly.lspPath', value);

  /** An environment where each folder reports its own interpreter. */
  function environmentPerFolder(
    interpreters: ReadonlyMap<string, string>,
    scopes: (vscode.Uri | undefined)[],
  ): PythonEnvironment {
    return {
      getInterpreterPath: async (uri?: vscode.Uri) => {
        scopes.push(uri);
        return uri && interpreters.get(uri.toString());
      },
    } as unknown as PythonEnvironment;
  }

  /**
   * Resolve with a per-folder interpreter map, reporting the specification,
   * what was logged, and which folders were consulted.
   */
  async function resolve(interpreters: ReadonlyMap<string, string> = new Map()) {
    const scopes: (vscode.Uri | undefined)[] = [];
    const lines: string[] = [];
    const spec = await resolveExecutable(
      extension!.extensionUri,
      environmentPerFolder(interpreters, scopes),
      recordingChannel(lines),
    );
    return {spec, scopes, log: lines.join('\n')};
  }

  /** As above, but against a workspace of `shape` rather than the open one. */
  async function resolveShape(
    shape: readonly vscode.WorkspaceFolder[],
    interpreters: ReadonlyMap<string, string> = new Map(),
  ) {
    const scopes: (vscode.Uri | undefined)[] = [];
    const lines: string[] = [];
    const spec = await resolveExecutable(
      extension!.extensionUri,
      environmentPerFolder(interpreters, scopes),
      recordingChannel(lines),
      shape,
    );
    return {spec, scopes, log: lines.join('\n')};
  }

  suiteSetup(async () => {
    // Pin the mode rather than trusting the default: the two configurations
    // share a user-data directory, so a leaked setting would quietly skip
    // the interpreter lookup these tests are about.
    await setting('pyrefly.pyreflyExecutable', 'from-environment');
  });

  suiteTeardown(async () => {
    await setting('pyrefly.pyreflyExecutable', undefined);
  });

  setup(function () {
    // Without two folders the rest of the suite proves nothing, so fail
    // rather than pass vacuously.
    assert.ok(extension, 'extension not found');
    assert.strictEqual(folders.length, 2, 'fixture workspace did not open');
  });

  teardown(async () => {
    await setLspPath(undefined);
  });

  test('opens the fixture folders in order', () => {
    assert.strictEqual(folders[0].name, 'primary');
    assert.strictEqual(folders[1].name, 'secondary');
  });

  test('consults the primary root for the interpreter, and only it', async () => {
    const {scopes} = await resolve();
    assert.strictEqual(scopes.length, 1, 'expected a single interpreter lookup');
    assert.strictEqual(scopes[0]?.toString(), folders[0].uri.toString());
    assert.notStrictEqual(scopes[0]?.toString(), folders[1].uri.toString());
  });

  test('a new interpreter in the primary root changes the selection', async () => {
    const before = join(tmpdir(), 'pyrefly-env-primary-before', 'python');
    const after = join(tmpdir(), 'pyrefly-env-primary-after', 'python');
    const key = folders[0].uri.toString();

    const first = await resolve(new Map([[key, before]]));
    assert.ok(first.log.includes(`interpreter=${before}`), first.log);

    // Re-resolving after the primary root's environment changes picks the
    // new one up; neither exists, so both fall back to the bundled binary.
    const second = await resolve(new Map([[key, after]]));
    assert.ok(second.log.includes(`interpreter=${after}`), second.log);
    assert.strictEqual(second.spec.command, bundled);
  });

  test('a new interpreter in the secondary root does not change the selection', async () => {
    const primary = join(tmpdir(), 'pyrefly-env-primary', 'python');
    const withSecondary = (suffix: string) =>
      new Map([
        [folders[0].uri.toString(), primary],
        [
          folders[1].uri.toString(),
          join(tmpdir(), `pyrefly-env-secondary-${suffix}`, 'python'),
        ],
      ]);

    const first = await resolve(withSecondary('before'));
    const second = await resolve(withSecondary('after'));

    // Changing the secondary root's environment moves nothing: the lookup is
    // scoped to the primary root, so its interpreter is the only one that
    // ever reaches the selection.
    assert.strictEqual(second.spec.command, first.spec.command);
    assert.ok(second.log.includes(`interpreter=${primary}`), second.log);
    assert.ok(!second.log.includes('pyrefly-env-secondary'), second.log);
    assert.strictEqual(second.scopes.length, 1);
  });

  test('resolves a relative lspPath against the primary root', async () => {
    await setLspPath('./bin/pyrefly');
    const {spec} = await resolve();
    assert.strictEqual(spec.command, join(folders[0].uri.fsPath, 'bin', 'pyrefly'));
    assert.notStrictEqual(
      spec.command,
      join(folders[1].uri.fsPath, 'bin', 'pyrefly'),
    );
  });

  /**
   * The remaining cases hand `resolveExecutable` the folder list directly
   * instead of rearranging the open window. `updateWorkspaceFolders` would
   * terminate and restart the extension host — and with it the test runner —
   * on every one of these except appending after the first folder.
   *
   * Note that the extension does not listen for workspace folder changes, so
   * these describe what the *next* resolution picks, not a restart. A running
   * server keeps its binary until a settings or interpreter change re-resolves.
   */
  const firstPython = join(tmpdir(), 'pyrefly-env-first', 'python');
  const secondPython = join(tmpdir(), 'pyrefly-env-second', 'python');
  const bothFolders = () =>
    new Map([
      [folders[0].uri.toString(), firstPython],
      [folders[1].uri.toString(), secondPython],
    ]);

  test('reordering the folders moves the lookup to the new first one', async () => {
    const before = await resolveShape(folders, bothFolders());
    const after = await resolveShape(
      [folders[1], folders[0]],
      bothFolders(),
    );

    assert.ok(before.log.includes(`interpreter=${firstPython}`), before.log);
    assert.ok(after.log.includes(`interpreter=${secondPython}`), after.log);
  });

  test('adding the first folder to an empty window gives the lookup a root', async () => {
    const empty = await resolveShape([], bothFolders());
    assert.deepStrictEqual(empty.scopes, [undefined]);
    assert.strictEqual(empty.spec.command, bundled);

    const added = await resolveShape([folders[0]], bothFolders());
    assert.ok(added.log.includes(`interpreter=${firstPython}`), added.log);
  });

  test('adding a folder after the first leaves the lookup alone', async () => {
    const appended: vscode.WorkspaceFolder = {
      uri: vscode.Uri.joinPath(folders[1].uri, '..', 'third'),
      name: 'third',
      index: 2,
    };
    const interpreters = new Map([
      ...bothFolders(),
      [appended.uri.toString(), join(tmpdir(), 'pyrefly-env-third', 'python')],
    ]);

    const {scopes, log} = await resolveShape(
      [...folders, appended],
      interpreters,
    );

    assert.deepStrictEqual(scopes.map(String), [folders[0].uri.toString()]);
    assert.ok(log.includes(`interpreter=${firstPython}`), log);
    assert.ok(!log.includes('pyrefly-env-third'), log);
  });

  test('removing the first folder hands the lookup to the second', async () => {
    const {scopes, log} = await resolveShape([folders[1]], bothFolders());

    assert.deepStrictEqual(scopes.map(String), [folders[1].uri.toString()]);
    assert.ok(log.includes(`interpreter=${secondPython}`), log);
    assert.ok(!log.includes(firstPython), log);
  });

  test('removing the last folder leaves no root and falls back', async () => {
    const {spec, scopes, log} = await resolveShape([], bothFolders());

    assert.deepStrictEqual(scopes, [undefined]);
    assert.strictEqual(spec.command, bundled);
    assert.ok(log.includes('interpreter=<none>'), log);
    assert.ok(log.includes('no active Python interpreter'), log);
  });
});
