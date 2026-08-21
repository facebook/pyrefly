/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import {promises as fs} from 'fs';
import {homedir, tmpdir} from 'os';
import {dirname, join} from 'path';
import * as vscode from 'vscode';
import {resolveLspPath} from '../extension';

suite('Extension Test Suite', () => {
	const extension: vscode.Extension<unknown> | undefined = vscode.extensions.getExtension('meta.pyrefly');

	test('Test activation', async function () {
		// On macos-13, we've noticed successful test activation take up to 3500ms.
		this.timeout(10000);
		await extension?.activate();
		assert.ok(true);
	});

	test('Infer types in the current file', async function () {
		this.timeout(30000);
		assert.ok(extension);
		await extension.activate();

		const directory = await fs.mkdtemp(join(tmpdir(), 'pyrefly-infer-'));
		const uri = vscode.Uri.file(join(directory, 'test.py'));
		try {
			await vscode.workspace.fs.writeFile(
				uri,
				Buffer.from('def foo():\n    return 1\n'),
			);
			const document = await vscode.workspace.openTextDocument(uri);
			await vscode.window.showTextDocument(document);

			await vscode.commands.executeCommand('pyrefly.infer');

			const result = Buffer.from(
				await vscode.workspace.fs.readFile(uri),
			).toString();
			assert.strictEqual(result, 'def foo() -> int:\n    return 1\n');
		} finally {
			await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
			await fs.rm(directory, {recursive: true, force: true});
		}
	});
});

suite('resolveLspPath', () => {
	const workspace = vscode.Uri.file(join(tmpdir(), 'pyrefly-workspace'));

	test('don\'t touch default', () => {
		assert.strictEqual(resolveLspPath('', workspace), '');
	});

	test('keep $PATH lookup untouched', () => {
		assert.strictEqual(resolveLspPath('pyrefly', workspace), 'pyrefly');
	});

	test('leave an absolute path alone', () => {
		const absolute = join(workspace.fsPath, 'pyrefly');
		assert.strictEqual(resolveLspPath(absolute, workspace), absolute);
	});

	test('resolve relative paths against workspace root', () => {
		for (const relative of ['./bin/pyrefly', 'target/debug/pyrefly']) {
			assert.strictEqual(
				resolveLspPath(relative, workspace),
				join(workspace.fsPath, relative),
			);
		}
	});

	test('resolves parent-relative paths against workspace root', () => {
		assert.strictEqual(
			resolveLspPath('../target/debug/pyrefly', workspace),
			join(dirname(workspace.fsPath), 'target', 'debug', 'pyrefly'),
		);
	});

	test('handle windows path separators', () => {
		const result = resolveLspPath('.\\bin\\pyrefly.exe', workspace);
		if (process.platform === 'win32') {
			assert.strictEqual(result, join(workspace.fsPath, 'bin', 'pyrefly.exe'));
		} else {
			assert.strictEqual(result, '.\\bin\\pyrefly.exe');
		}
	});

	test('expand a ~-relative path against homedir', () => {
		assert.strictEqual(
			resolveLspPath('~/bin/pyrefly', workspace),
			join(homedir(), 'bin', 'pyrefly'),
		);
	});

	test('don\'t do anything with an unknown workspace', () => {
		assert.strictEqual(
			resolveLspPath('./bin/pyrefly', undefined),
			'./bin/pyrefly',
		);
	});
});
