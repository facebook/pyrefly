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
import {State} from 'vscode-languageclient/node';
import type {TestHooks} from '../extension';

suite('language server recovery', () => {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  const interpreterTimeout = 10000;
  let hooks: TestHooks | undefined;

  async function setLspPath(value: string): Promise<void> {
    await vscode.workspace
      .getConfiguration()
      .update('pyrefly.lspPath', value, vscode.ConfigurationTarget.Global);
  }

  suiteSetup(async function () {
    this.timeout(interpreterTimeout);
    if (extension === undefined) {
      return;
    }
    hooks = (await extension.activate()) as TestHooks;
  });

  setup(function () {
    if (hooks === undefined) {
      this.skip();
    }
  });

  teardown(async function () {
    this.timeout(interpreterTimeout);
    if (hooks === undefined) {
      return;
    }
    await setLspPath('');
    await hooks.restartIfLaunchSpecChanged(
      extension!.extensionUri,
      'test teardown',
    );
  });

  test('a binary that cannot start is reverted, and a server stays up', async function () {
    this.timeout(interpreterTimeout);
    const working = hooks!.currentCommand();
    const missing = join(tmpdir(), 'pyrefly-does-not-exist', 'pyrefly');

    await setLspPath(missing);
    assert.strictEqual(
      await hooks!.restartIfLaunchSpecChanged(extension!.extensionUri, 'test'),
      'failed',
    );
    // Reverting matters twice over: the next change still compares as a
    // change, and `restart()` cannot revive a client whose start failed, so a
    // running server here also shows the replacement client did its job.
    assert.strictEqual(hooks!.currentCommand(), working);
    assert.strictEqual(hooks!.clientState(), State.Running);
  });

  test('we should not restart if the binary hasn\'t changed', async function () {
    this.timeout(interpreterTimeout);
    // The guard that keeps a multi-root workspace quiet: an interpreter
    // change in a secondary folder still re-resolves, but the primary root
    // decides the binary, so nothing changes and the server is left alone.
    assert.strictEqual(
      await hooks!.restartIfLaunchSpecChanged(extension!.extensionUri, 'test'),
      'unchanged',
    );
    assert.strictEqual(hooks!.clientState(), State.Running);
  });

  test('a second bad setting is still noticed after a revert', async function () {
    this.timeout(interpreterTimeout);
    const working = hooks!.currentCommand();
    const first = join(tmpdir(), 'pyrefly-missing-one', 'pyrefly');
    const second = join(tmpdir(), 'pyrefly-missing-two', 'pyrefly');

    await setLspPath(first);
    await hooks!.restartIfLaunchSpecChanged(extension!.extensionUri, 'test');

    await setLspPath(second);
    assert.strictEqual(
      await hooks!.restartIfLaunchSpecChanged(extension!.extensionUri, 'test'),
      'failed',
    );
    // Both failures revert to the last binary that actually ran, not to
    // whatever was configured before them. Landing on `first` here would mean
    // the revert target is the previous setting rather than the previous
    // working server, and the next failure would strand us on a dead binary.
    assert.strictEqual(hooks!.currentCommand(), working);
    assert.notStrictEqual(hooks!.currentCommand(), first);
    assert.notStrictEqual(hooks!.currentCommand(), second);
    assert.strictEqual(hooks!.clientState(), State.Running);
  });
});
