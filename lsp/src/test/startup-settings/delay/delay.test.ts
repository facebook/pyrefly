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
 * The shim holds the client's configuration responses for 3 seconds, as a slow
 * Python extension does, so the editor's startup requests reach the server
 * before the interpreter does. The server must hold those requests until the
 * interpreter arrives.
 */
suite('startup with a late initial configuration response', () => {
  test('the Python extension is installed', () => {
    assert.ok(vscode.extensions.getExtension('ms-python.python'));
  });

  test('startup requests and diagnostics use the interpreter from the Python extension', async function () {
    this.timeout(60000);
    const log = await expectWorkspaceSettingsAtStartup();
    assert.ok(
      log.indexOf('textDocument/definition') <
        log.indexOf('workspace/configuration response'),
      `the definition request must reach the server before the settings: ${log}`,
    );
  });
});
