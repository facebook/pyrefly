/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import {promises as fs} from 'fs';
import {join} from 'path';
import * as vscode from 'vscode';

/**
 * Its own `.vscode-test.mjs` configuration, because a failed activation sticks
 * for the lifetime of the window: VS Code caches the rejection, so running this
 * alongside the other suites would leave every later `activate()` rejecting.
 *
 * The fixture workspace sets `pyrefly.lspPath` to a path that does not exist,
 * which is read before activation and so decides the very first start.
 */
suite('activation with an unusable binary', () => {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  const binaryName = process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly';
  const plantedDir = join(
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '',
    'does-not-exist',
  );

  suiteSetup(async () => {
    // Windows will not let the recovery test below delete the binary it
    // planted, because the server it started is still running from that file.
    // Clearing it here — before anything activates — keeps the premise of the
    // whole suite true on a second run.
    await fs.rm(plantedDir, {recursive: true, force: true});
  });

  test('the fixture really points at a missing binary', async () => {
    assert.ok(extension, 'extension not found');
    assert.strictEqual(
      vscode.workspace.getConfiguration().get('pyrefly.lspPath'),
      './does-not-exist/pyrefly',
      'fixture settings were not applied',
    );
    // If a leftover binary survived a previous run, activation would succeed
    // and every test here would pass while proving nothing. Fail loudly.
    await assert.rejects(
      async () => await fs.access(join(plantedDir, binaryName)),
      'a binary is present where the fixture expects none',
    );
  });

  test('activation surfaces the failure rather than succeeding quietly', async function () {
    this.timeout(30000);
    assert.ok(extension);
    await assert.rejects(
      async () => await extension.activate(),
      'activation resolved despite the server being unable to start',
    );
  });

  test('the failure is remembered, not retried per call', async function () {
    this.timeout(30000);
    assert.ok(extension);
    // VS Code caches the activation result, so a second call reports the same
    // failure rather than spawning another doomed server.
    await assert.rejects(async () => await extension.activate());
  });

  test('its commands survive the failed activation', async () => {
    // This is what makes recovery possible at all: the registrations made
    // before the throw are still in place.
    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes('pyrefly.restartClient'), 'restart command gone');
    assert.ok(commands.includes('pyrefly.infer'), 'infer command gone');
  });

  test('a working binary at the configured path recovers the server', async function () {
    this.timeout(60000);
    assert.ok(extension);
    const folder = vscode.workspace.workspaceFolders?.[0];
    assert.ok(folder, 'expected the fixture folder to be open');

    // The setting keeps pointing at the same relative path; what changes is
    // that a usable binary now exists there. That is the "swap a broken
    // Pyrefly for a working one" case, without rewriting fixture settings.
    const working = vscode.Uri.joinPath(
      extension.extensionUri,
      'bin',
      process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly',
    ).fsPath;
    const placed = join(plantedDir, binaryName);

    await fs.mkdir(plantedDir, {recursive: true});
    await fs.copyFile(working, placed);
    await fs.chmod(placed, 0o755);
    try {
      await vscode.commands.executeCommand('pyrefly.restartClient');

      // A language feature is the only end-to-end evidence available here:
      // without a running server, no symbols come back.
      const document = await vscode.workspace.openTextDocument({
        language: 'python',
        content: 'def recovered() -> int:\n    return 1\n',
      });
      await vscode.window.showTextDocument(document);
      const symbols = await vscode.commands.executeCommand(
        'vscode.executeDocumentSymbolProvider',
        document.uri,
      );
      assert.ok(
        Array.isArray(symbols) && symbols.length > 0,
        'no document symbols, so no server came back',
      );
    } finally {
      await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
      // Best effort: on Windows the server started from this binary still holds
      // it open, so the unlink fails with EPERM. `suiteSetup` clears it on the
      // next run, when no process owns it.
      await fs.rm(plantedDir, {recursive: true, force: true}).catch(() => {});
    }
  });

  test('VS Code still counts the extension as active', () => {
    assert.ok(extension);
    // Surprising, but it is what makes the recovery path reachable: activation
    // ran, so the extension is active even though `activate()` threw, and
    // everything registered before the throw — the configuration listeners and
    // the commands — is still live. If this ever flips to false, a corrected
    // setting would no longer be able to bring a server up.
    assert.strictEqual(extension.isActive, true);
  });
});
