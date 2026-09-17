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

suite('find_pyrefly.py', () => {
  // The selection logic itself is covered by resources/test/test_find_pyrefly.py,
  // which runs on pull requests and needs no extension host. What only this suite
  // can show is that the script is actually packaged into the VSIX and reachable
  // where the extension looks for it.
  test('is packaged with the extension', async () => {
    const extension = vscode.extensions.getExtension('meta.pyrefly');
    assert.ok(extension);
    await fs.access(
      join(extension.extensionPath, 'resources', 'find_pyrefly.py'),
    );
  });
});
