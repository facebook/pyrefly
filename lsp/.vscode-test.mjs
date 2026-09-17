/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { defineConfig } from '@vscode/test-cli';

// One configuration per window shape. The multi-root tests need two folders
// open; the failed-start tests open a workspace whose settings break the first
// launch, and VS Code caches that failure for the lifetime of the window, so
// sharing one with the rest of the suite would leave every later activation
// rejecting. The globs are disjoint — note the single star, so the default set
// reaches into neither subdirectory — and `--label` picks one.
// Writing a setting makes the extension re-resolve its binary and restart, so a
// test that awaits `configuration.update` is also waiting on a server launch.
// Mocha's 2s default is not enough for that on the slower CI runners.
const mocha = {timeout: 20000};

export default defineConfig([
	{
		label: 'single-root',
		files: 'dist/test/*.test.js',
		workspaceFolder: "../pyrefly/lib/test/lsp/test_files",
		mocha
	},
	{
		label: 'multi-root',
		files: 'dist/test/multi-root/*.test.js',
		// `workspaceFolder` is handed to the VS Code CLI as the path to open, so
		// a `.code-workspace` file opens its folders as a multi-root workspace.
		workspaceFolder: "./src/test/fixtures/multi-root.code-workspace",
		mocha,
		// The manifest declares `untrustedWorkspaces.supported: false`, and a
		// trust prompt would leave the extension restricted.
		launchArgs: ['--disable-workspace-trust']
	},
	{
		label: 'failed-start',
		files: 'dist/test/failed-start/*.test.js',
		mocha,
		// Workspace settings are read before activation, so this is how the very
		// first `client.start()` is made to fail.
		workspaceFolder: "./src/test/fixtures/failed-start",
		launchArgs: ['--disable-workspace-trust']
	}
]);
