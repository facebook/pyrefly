/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Builds the workspaces for the startup-settings tests. They live outside the
// repository for two reasons: the settings need absolute paths, and Pyrefly
// finds a virtual environment near a project on its own, which would let the
// default settings resolve the same imports as the client's settings.

import {execFileSync} from 'child_process';
import {mkdirSync, mkdtempSync, writeFileSync} from 'fs';
import {tmpdir} from 'os';
import {dirname, join} from 'path';
import {fileURLToPath} from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const windows = process.platform === 'win32';

function write(path, text) {
  mkdirSync(dirname(path), {recursive: true});
  writeFileSync(path, text);
}

/**
 * Make one workspace per mode of `lsp-shim.js`, sharing one virtual
 * environment that only `python.defaultInterpreterPath` points at.
 * Returns the workspace folder for each mode.
 */
export function makeStartupSettingsWorkspaces(modes) {
  const root = mkdtempSync(join(tmpdir(), 'pyrefly-startup-settings-'));
  // Pyrefly also finds a virtual environment beside a project's ancestors, so
  // the environment gets a temporary directory of its own.
  const venv = join(mkdtempSync(join(tmpdir(), 'pyrefly-startup-venv-')), 'venv');
  execFileSync(windows ? 'python' : 'python3', ['-m', 'venv', '--without-pip', venv]);
  const python = join(venv, windows ? 'Scripts/python.exe' : 'bin/python');
  const purelib = execFileSync(python, [
    '-c',
    'import sysconfig; print(sysconfig.get_paths()["purelib"])',
  ])
    .toString()
    .trim();
  write(join(purelib, 'fakepkg', '__init__.py'), 'VALUE: int = 1\n');

  const server = join(here, '../../../bin', windows ? 'pyrefly.exe' : 'pyrefly');
  const folders = {};
  for (const mode of modes) {
    const folder = join(root, mode);
    write(
      join(folder, 'main.py'),
      'import fakepkg\nfrom bar import helper\n\ny: str = fakepkg.VALUE\nhelper()\n',
    );
    write(join(folder, 'bar.py'), 'def helper() -> None: ...\n');
    write(
      join(folder, '.vscode', 'settings.json'),
      JSON.stringify(
        {
          'python.defaultInterpreterPath': python,
          'python.languageServer': 'None',
          'python.pyrefly.typeCheckingMode': 'default',
          'pyrefly.lspPath': process.execPath,
          'pyrefly.lspArguments': [
            join(here, 'lsp-shim.js'),
            server,
            mode,
            join(folder, '.vscode', 'lsp-shim.jsonl'),
            'lsp',
          ],
        },
        null,
        4,
      ),
    );
    folders[mode] = folder;
  }
  return folders;
}
