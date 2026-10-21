/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {definition, mainPy, openMainPy, shimLog, waitFor} from '../common';

/**
 * The shim never forwards the client's first configuration response. The
 * server must still answer the editor's startup requests, with default
 * settings, instead of holding them forever.
 */
suite('startup without an initial configuration response', () => {
  test('a startup request is answered with default settings', async function () {
    this.timeout(60000);
    await openMainPy();
    const targets = await definition(mainPy(), new vscode.Position(1, 17));

    assert.deepStrictEqual(
      targets.map(target => target.path.split('/').pop()),
      ['bar.py'],
    );
    await waitFor(
      'the shim to drop the first configuration response',
      async () => (await shimLog()).some(entry => entry.dropped !== undefined),
      30000,
    );
  });
});
