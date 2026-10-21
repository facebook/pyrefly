/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {expectWorkspaceSettingsAtStartup} from '../common';

/**
 * The shim forwards the client's configuration responses at once, as when the
 * Python extension has the interpreter ready. The server must still answer
 * startup requests with the workspace settings and never show diagnostics from
 * default settings.
 */
suite('startup with an instant initial configuration response', () => {
  test('the Python extension is installed', () => {
    assert.ok(vscode.extensions.getExtension('ms-python.python'));
  });

  test('startup requests and diagnostics use the interpreter from the Python extension', async function () {
    this.timeout(60000);
    await expectWorkspaceSettingsAtStartup();
  });
});
