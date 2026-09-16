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
import {State} from 'vscode-languageclient/node';
import {resolveLspPath, resolveExecutable} from '../lspPath';
import {PythonEnvironment} from '../python-environment';
import type {TestHooks} from '../extension';

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
	const workspace = join(tmpdir(), 'pyrefly-workspace');

	test('don\'t touch default', () => {
		assert.strictEqual(resolveLspPath('', workspace), '');
	});

	test('keep $PATH lookup untouched', () => {
		assert.strictEqual(resolveLspPath('pyrefly', workspace), 'pyrefly');
	});

	test('leave an absolute path alone', () => {
		const absolute = join(workspace, 'pyrefly');
		assert.strictEqual(resolveLspPath(absolute, workspace), absolute);
	});

	test('resolve relative paths against workspace root', () => {
		for (const relative of ['./bin/pyrefly', 'target/debug/pyrefly']) {
			assert.strictEqual(
				resolveLspPath(relative, workspace),
				join(workspace, relative),
			);
		}
	});

	test('resolves parent-relative paths against workspace root', () => {
		assert.strictEqual(
			resolveLspPath('../target/debug/pyrefly', workspace),
			join(dirname(workspace), 'target', 'debug', 'pyrefly'),
		);
	});

	test('handle windows path separators', () => {
		const result = resolveLspPath('.\\bin\\pyrefly.exe', workspace);
		if (process.platform === 'win32') {
			assert.strictEqual(result, join(workspace, 'bin', 'pyrefly.exe'));
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

/** Records what the binary selection logged, so tests can read the reason. */
function recordingChannel(lines: string[]): vscode.OutputChannel {
	return {
		name: 'recording',
		append: (value: string) => void lines.push(value),
		appendLine: (value: string) => void lines.push(value),
		replace: () => {},
		clear: () => {},
		show: () => {},
		hide: () => {},
		dispose: () => {},
	} as unknown as vscode.OutputChannel;
}

/** Only `getInterpreterPath` is ever called, but the class is nominally typed. */
function fakeEnvironment(interpreter: string | undefined): PythonEnvironment {
	return {
		getInterpreterPath: async () => interpreter,
	} as unknown as PythonEnvironment;
}

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

	test('resolves a relative lspPath against the primary root', async function () {
		const root = vscode.workspace.workspaceFolders?.[0]?.uri;
		if (root === undefined) {
			this.skip();
		}
		const {spec} = await resolve({'pyrefly.lspPath': './bin/pyrefly'});
		assert.strictEqual(spec.command, join(root!.fsPath, 'bin', 'pyrefly'));
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
		// Node can't run Python, so it should expectedly fail, which is a stand-in
		// for an interpreter that can't execute the finder script. Output looks something
		// like this:
		// ```
		// pyrefly/lsp/resources/find_pyrefly.py:1
		// # Copyright (c) Meta Platforms, Inc. and affiliates.
		// ^
		//
		// SyntaxError: Invalid or unexpected token
		// at Object.compileFunction (node:vm:360:18)
		// at wrapSafe (node:internal/modules/cjs/loader:1126:15)
		// at Module._compile (node:internal/modules/cjs/loader:1162:27)
		// at Object.Module._extensions..js (node:internal/modules/cjs/loader:1252:10)
		// at Module.load (node:internal/modules/cjs/loader:1076:32)
		// at Function.Module._load (node:internal/modules/cjs/loader:911:12)
		// at Function.executeUserEntryPoint [as runMain] (node:internal/modules/run_main:81:12)
		// at node:internal/main/run_main_module:22:47
		// ```
		const {spec, log} = await resolve(
			{'pyrefly.pyreflyExecutable': 'from-environment'},
			process.execPath,
		);
		assert.strictEqual(spec.command, bundled);
		assert.ok(log.includes('stderr:'), log);
	});
});

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
