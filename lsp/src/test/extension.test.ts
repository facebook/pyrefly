/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
	const extension: vscode.Extension<unknown> | undefined = vscode.extensions.getExtension('meta.pyrefly');

	test('Test activation', async function () {
		// On macos-13, we've noticed successful test activation take up to 3500ms.
		this.timeout(10000);
		await extension?.activate();
		assert.ok(true);
	});
});
