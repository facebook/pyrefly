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
import {PythonEnvironments} from '@vscode/python-environments';
import {PythonExtension} from '@vscode/python-extension';

export const fixtureFolder = (): vscode.Uri =>
  vscode.workspace.workspaceFolders![0].uri;

export const mainPy = (): vscode.Uri =>
  vscode.Uri.joinPath(fixtureFolder(), 'main.py');

/** The client messages in the order that `lsp-shim.js` forwarded them. */
export async function shimLog(): Promise<Record<string, string>[]> {
  const text = await fs.readFile(
    join(fixtureFolder().fsPath, '.vscode', 'lsp-shim.jsonl'),
    'utf8',
  );
  return text
    .split('\n')
    .filter(line => line !== '')
    .map(line => JSON.parse(line));
}

/**
 * The interpreter that the extension would send for the fixture folder. This
 * asks the same APIs in the same order as `PythonEnvironment`: the Python
 * Environments extension first, then the Python extension.
 */
async function selectedInterpreter(): Promise<string | undefined> {
  try {
    const api = await PythonEnvironments.api();
    return (await api.getEnvironment(fixtureFolder()))?.execInfo?.run
      ?.executable;
  } catch {
    const api = await PythonExtension.api();
    return (await api.environments.getActiveEnvironmentPath(fixtureFolder()))
      .path;
  }
}

/**
 * Wait until the Python extensions select `python.defaultInterpreterPath`.
 * Right after they start, they can select the machine's interpreter instead,
 * and then the client's settings do not name the virtual environment.
 * The paths are compared as `fsPath`s, because on Windows the Python
 * extensions report the drive letter in lower case.
 */
export async function waitForConfiguredInterpreter(): Promise<void> {
  const expected = vscode.Uri.file(
    vscode.workspace
      .getConfiguration('python', fixtureFolder())
      .get<string>('defaultInterpreterPath')!,
  ).fsPath;
  let actual: string | undefined;
  await waitFor(
    'the Python extensions to select python.defaultInterpreterPath',
    async () => {
      actual = await selectedInterpreter();
      return actual !== undefined && vscode.Uri.file(actual).fsPath === expected;
    },
    30000,
  ).catch(error => {
    throw new Error(`${error.message}; they selected ${actual}`);
  });
}

/**
 * Activate the extension and open `main.py`. The extension starts the server
 * on activation, so the shim's mode decides whether the editor's requests
 * reach the server before the client's settings do.
 */
export async function openMainPy(): Promise<vscode.TextDocument> {
  await vscode.extensions.getExtension('meta.pyrefly')!.activate();
  const document = await vscode.workspace.openTextDocument(mainPy());
  await vscode.window.showTextDocument(document);
  return document;
}

export async function definition(
  uri: vscode.Uri,
  position: vscode.Position,
): Promise<vscode.Uri[]> {
  const locations = await vscode.commands.executeCommand<
    (vscode.Location | vscode.LocationLink)[]
  >('vscode.executeDefinitionProvider', uri, position);
  return locations.map(location =>
    'targetUri' in location ? location.targetUri : location.uri,
  );
}

/** Every set of Pyrefly diagnostics that VS Code reported for `uri`. */
export function recordDiagnostics(uri: vscode.Uri): {
  sets: vscode.Diagnostic[][];
  dispose: () => void;
} {
  const sets: vscode.Diagnostic[][] = [];
  const subscription = vscode.languages.onDidChangeDiagnostics(event => {
    if (event.uris.some(changed => changed.toString() === uri.toString())) {
      sets.push(
        vscode.languages
          .getDiagnostics(uri)
          .filter(diagnostic => diagnostic.source === 'Pyrefly'),
      );
    }
  });
  return {sets, dispose: () => subscription.dispose()};
}

export function diagnosticCode(diagnostic: vscode.Diagnostic): string {
  const code = diagnostic.code;
  return String(typeof code === 'object' ? code.value : code);
}

export async function waitFor(
  description: string,
  condition: () => boolean | Promise<boolean>,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!(await condition())) {
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${description}`);
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

/**
 * Open `main.py` and check that startup go-to-definition and every diagnostic
 * set that the editor shows use the workspace settings. Only the interpreter
 * from the Python extension can import `fakepkg`, and only the workspace
 * settings select the preset that reports `bad-assignment`. Returns the client
 * messages in the order that the shim forwarded them.
 */
export async function expectWorkspaceSettingsAtStartup(): Promise<
  (string | undefined)[]
> {
  const diagnostics = recordDiagnostics(mainPy());
  try {
    await waitForConfiguredInterpreter();
    await openMainPy();
    const targets = await definition(mainPy(), new vscode.Position(0, 8));

    let log: (string | undefined)[] = [];
    await waitFor(
      'the shim to forward the configuration response',
      async () => {
        log = (await shimLog()).map(entry => entry.method);
        return log.includes('workspace/configuration response');
      },
      30000,
    );
    assert.deepStrictEqual(
      targets.map(target => target.path.split('/').slice(-2).join('/')),
      ['fakepkg/__init__.py'],
    );

    await waitFor(
      'the bad-assignment error in main.py',
      () =>
        diagnostics.sets.some(set =>
          set.some(d => diagnosticCode(d) === 'bad-assignment'),
        ),
      30000,
    ).catch(error => {
      throw new Error(
        `${error.message}; the editor showed ${JSON.stringify(
          diagnostics.sets.map(set => set.map(diagnosticCode)),
        )}`,
      );
    });
    // Each set is what the editor showed at one moment. Default settings use
    // the `basic` preset, which hides `bad-assignment`, and do not name the
    // interpreter that can import `fakepkg`. A set from them is empty or
    // reports `missing-import`.
    assert.deepStrictEqual(
      diagnostics.sets.map(set => set.map(diagnosticCode)),
      diagnostics.sets.map(() => ['bad-assignment']),
    );
    return log;
  } finally {
    diagnostics.dispose();
  }
}
