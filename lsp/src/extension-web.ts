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

const INDEX_MAX_FILES = 2000;
const INDEX_MAX_BYTES = 5 * 1024 * 1024;
const INDEX_EXCLUDES =
  '**/{.git,node_modules,dist,build,.venv,venv,.mypy_cache,.pyrefly_cache}/**';

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel | undefined;

function createWorker(context: vscode.ExtensionContext): Worker {
  const workerUri = vscode.Uri.joinPath(
    context.extensionUri,
    'dist',
    'pyrefly-web-worker.js',
  );
  return new Worker(workerUri.toString(true));
}

async function indexWorkspaceFiles(client: LanguageClient): Promise<void> {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return;
  }

  outputChannel?.appendLine('Indexing workspace files for Pyrefly web...');
  const pyFiles = await vscode.workspace.findFiles(
    '**/*.{py,pyi}',
    INDEX_EXCLUDES,
    INDEX_MAX_FILES,
  );
  const configFiles = await vscode.workspace.findFiles(
    '**/pyrefly.toml',
    INDEX_EXCLUDES,
    INDEX_MAX_FILES,
  );

  const uris = new Map<string, vscode.Uri>();
  for (const uri of [...pyFiles, ...configFiles]) {
    uris.set(uri.toString(), uri);
  }

  const files: Record<string, string> = {};
  let totalBytes = 0;
  const decoder = new TextDecoder('utf-8');

  for (const uri of uris.values()) {
    const data = await vscode.workspace.fs.readFile(uri);
    totalBytes += data.byteLength;
    if (totalBytes > INDEX_MAX_BYTES) {
      vscode.window.showWarningMessage(
        'Pyrefly web indexing stopped early due to workspace size. Open files will still work.',
      );
      break;
    }
    const relativePath = vscode.workspace.asRelativePath(uri, false);
    files[relativePath] = decoder.decode(data);
  }

  await client.sendRequest('pyrefly/setWorkspaceFiles', {files});
  outputChannel?.appendLine(
    `Indexed ${Object.keys(files).length} files for Pyrefly web.`,
  );
}

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server',
    );
  }

  const worker = createWorker(context);
  const wasmUri = await vscode.env.asExternalUri(
    vscode.Uri.joinPath(context.extensionUri, 'dist', 'pyrefly_wasm_bg.wasm'),
  );
  outputChannel.appendLine(
    `Pyrefly web: using wasm URI ${wasmUri.toString(true)}`,
  );
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{language: 'python'}],
    outputChannel,
    initializationOptions: {
      wasmUri: wasmUri.toString(true),
    },
  };

  client = new LanguageClient(
    'pyrefly',
    'Pyrefly language server',
    clientOptions,
    worker,
  );

  const startPromise = client.start();
  context.subscriptions.push({
    dispose: () => {
      void client?.stop();
    },
  });
  await startPromise;

  try {
    await indexWorkspaceFiles(client);
  } catch (error) {
    outputChannel?.appendLine(
      `Failed to index workspace files for Pyrefly web: ${String(error)}`,
    );
  }
}

export async function deactivate(): Promise<void> {
  await client?.stop();
  outputChannel?.dispose();
}
