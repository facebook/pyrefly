/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {LanguageClient, LanguageClientOptions} from 'vscode-languageclient/browser';
import {
  CONFIG_FILES, READ_FILE, SUPPORTED_SCHEMES, ProjectOptions, ReadParams, WorkspaceFile,
} from './web-session';

let client: LanguageClient | undefined;

/** Use the resource's editor setting only when TOML supplies no Python version. */
function configuredPythonVersion(uri?: vscode.Uri): string {
  return vscode.workspace.getConfiguration('python.pyrefly', uri)
    .get<string>('pythonVersion', '').trim() || '3.12';
}

/** Read only inside the requesting workspace. Never search another root. */
async function readWorkspaceFile(params: ReadParams): Promise<WorkspaceFile | null> {
  const folder = vscode.workspace.workspaceFolders?.find(item =>
    item.uri.toString() === params.rootUri,
  );
  if (!folder) {
    return null;
  }
  for (const filename of params.filenames) {
    if (filename.includes('\\') ||
        filename.split('/').some(part => part === '' || part === '..' || part === '.')) {
      throw new Error('Invalid workspace file path: ' + filename);
    }
    const uri = vscode.Uri.joinPath(folder.uri, ...filename.split('/'));
    try {
      const data = await vscode.workspace.fs.readFile(uri);
      if (data.byteLength > 1024 * 1024) {
        throw new Error('Pyrefly web supports files up to 1 MiB: ' + uri.toString());
      }
      return {uri: uri.toString(), filename, content: new TextDecoder().decode(data)};
    } catch (error) {
      if (!(error instanceof vscode.FileSystemError) || error.code !== 'FileNotFound') {
        throw error;
      }
    }
  }
  return null;
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const outputChannel = vscode.window.createOutputChannel('Pyrefly language server');
  const traceOutputChannel = vscode.window.createOutputChannel('Pyrefly language server trace');
  context.subscriptions.push(outputChannel, traceOutputChannel);
  const wasmUri = vscode.Uri.joinPath(context.extensionUri, 'dist', 'pyrefly_wasm_bg.wasm');
  let wasmBytes: Uint8Array;
  try {
    wasmBytes = await vscode.workspace.fs.readFile(wasmUri);
  } catch (error) {
    const message = 'Cannot read the Pyrefly WASM bundle. Run pyrefly_wasm/build.sh and rebuild the extension.';
    outputChannel.appendLine(message + ' ' + String(error));
    void vscode.window.showErrorMessage(message);
    return;
  }

  const projects: ProjectOptions[] = [];
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const files: Record<string, string> = {};
    for (const filename of CONFIG_FILES) {
      const file = await readWorkspaceFile({rootUri: folder.uri.toString(), filenames: [filename]});
      if (file) { files[filename] = file.content; }
    }
    projects.push({
      uri: folder.uri.toString(), pythonVersion: configuredPythonVersion(folder.uri), files,
    });
  }

  const watcher = vscode.workspace.createFileSystemWatcher('**/*.{py,pyi,toml}');
  const worker = new Worker(vscode.Uri.joinPath(
    context.extensionUri, 'dist', 'pyrefly-web-worker.js',
  ).toString(true));
  const clientOptions: LanguageClientOptions = {
    documentSelector: Array.from(new Set([
      ...SUPPORTED_SCHEMES, ...projects.map(project => vscode.Uri.parse(project.uri).scheme),
    ])).filter(scheme => scheme !== 'vscode-notebook-cell')
      .map(scheme => ({language: 'python', scheme})),
    outputChannel, traceOutputChannel,
    synchronize: {fileEvents: watcher},
    initializationOptions: {
      wasmBytes, projects, pythonVersion: configuredPythonVersion(),
    },
  };
  client = new LanguageClient('pyrefly', 'Pyrefly language server', clientOptions, worker);
  client.onRequest(READ_FILE, readWorkspaceFile);
  const activeClient = client;
  context.subscriptions.push(watcher, {
    dispose: () => {
      void activeClient.stop().finally(() => worker.terminate());
    },
  });
  await activeClient.start();
}

export async function deactivate(): Promise<void> {
  await client?.stop();
}
