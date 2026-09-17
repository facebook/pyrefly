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
import {resolveExecutable} from '../lspPath';
import {fakeEnvironment, recordingChannel} from './helpers';

suite('resolveExecutable', () => {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  const bundled = vscode.Uri.joinPath(
    extension?.extensionUri ?? vscode.Uri.file(''),
    'bin',
    process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly',
  ).fsPath;

  // User settings, not workspace: the harness opens a folder that may not
  // exist, and workspace settings would be written into the repository.
  async function settings(values: Record<string, unknown>): Promise<void> {
    const configuration = vscode.workspace.getConfiguration();
    for (const [key, value] of Object.entries(values)) {
      // Clearing a setting that was never set still costs a write and a
      // round trip through the extension host, which is what pushed the
      // teardown past mocha's timeout on slower runners.
      if (
        value === undefined &&
        configuration.inspect(key)?.globalValue === undefined
      ) {
        continue;
      }
      await configuration.update(
        key,
        value,
        vscode.ConfigurationTarget.Global,
      );
    }
  }

  /** Resolve under `values`, returning the specification and what was logged. */
  async function resolve(values: Record<string, unknown>, interpreter?: string) {
    await settings(values);
    const lines: string[] = [];
    const spec = await resolveExecutable(
      extension!.extensionUri,
      fakeEnvironment(interpreter),
      recordingChannel(lines),
    );
    return {spec, log: lines.join('\n')};
  }

  setup(function () {
    if (extension === undefined) {
      this.skip();
    }
  });

  teardown(async () => {
    await settings({
      'pyrefly.lspPath': undefined,
      'pyrefly.pyreflyExecutable': undefined,
      'pyrefly.lspArguments': undefined,
    });
  });

  test('lspPath wins over the executable setting', async () => {
    const chosen = join(tmpdir(), 'somewhere', 'pyrefly');
    const {spec, log} = await resolve({
      'pyrefly.lspPath': chosen,
      'pyrefly.pyreflyExecutable': 'from-environment',
    });
    assert.strictEqual(spec.command, chosen);
    assert.ok(log.includes('mode=pyrefly.lspPath'), log);
  });

  test('bundled asks for the shipped binary and is not a fallback', async () => {
    const {spec, log} = await resolve({'pyrefly.pyreflyExecutable': 'bundled'});
    assert.strictEqual(spec.command, bundled);
    assert.ok(!log.includes('fell back'), log);
  });

  test('an unrecognized executable setting falls back and says so', async () => {
    const {spec, log} = await resolve({
      'pyrefly.pyreflyExecutable': 'not-a-real-choice',
    });
    assert.strictEqual(spec.command, bundled);
    assert.ok(log.includes('not a recognized'), log);
  });

  test('empty arguments fall back to the lsp subcommand', async () => {
    const {spec} = await resolve({
      'pyrefly.pyreflyExecutable': 'bundled',
      'pyrefly.lspArguments': [],
    });
    assert.deepStrictEqual(spec.args, ['lsp']);
  });

  test('no active interpreter falls back to the bundled binary', async () => {
    const {spec, log} = await resolve(
      {'pyrefly.pyreflyExecutable': 'from-environment'},
      undefined,
    );
    assert.strictEqual(spec.command, bundled);
    assert.ok(log.includes('no active Python interpreter'), log);
  });

  test('a failing interpreter reports its stderr', async () => {
    // We pass in any program that will fail here to simulate a broken Python interpreter
    // or failing selection script. In this case, passing in node as a 'Python interpreter'
    // will fail from a syntax error, giving us the failed Pyrefly finder behavior we want
    // to test.
    const {spec, log} = await resolve(
      {'pyrefly.pyreflyExecutable': 'from-environment'},
      process.execPath,
    );
    assert.strictEqual(spec.command, bundled);
    assert.ok(log.includes('stderr:'), log);
  });
});
