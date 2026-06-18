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

const READ_WORKSPACE_FILE_REQUEST = 'pyrefly/readWorkspaceFile';
const INITIAL_CONFIG_FILES = ['pyrefly.toml', 'pyproject.toml'];

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel | undefined;
let traceOutputChannel: vscode.OutputChannel | undefined;

function createWorker(context: vscode.ExtensionContext): Worker {
  const workerUri = vscode.Uri.joinPath(
    context.extensionUri,
    'dist',
    'pyrefly-web-worker.js',
  );
  return new Worker(workerUri.toString(true));
}

function configuredPythonVersion(): string | undefined {
  const configured = vscode.workspace
    .getConfiguration('python.pyrefly')
    .get<string>('pythonVersion', '')
    .trim();
  return configured === '' ? undefined : configured;
}

async function readInitialConfigFiles(): Promise<Record<string, string>> {
  const files: Record<string, string> = {};
  const decoder = new TextDecoder('utf-8');
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    for (const filename of INITIAL_CONFIG_FILES) {
      if (files[filename] != null) {
        continue;
      }
      try {
        files[filename] = decoder.decode(
          await vscode.workspace.fs.readFile(
            vscode.Uri.joinPath(folder.uri, filename),
          ),
        );
      } catch {
        // Missing config files are normal. Rust/WASM owns parsing once found.
      }
    }
  }
  return files;
}

async function readWorkspaceFile(
  params: {filenames: string[]},
): Promise<{filename: string; content: string} | null> {
  const decoder = new TextDecoder('utf-8');
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    for (const filename of params.filenames) {
      if (
        filename.startsWith('/') ||
        filename.split('/').some(part => part === '' || part === '..')
      ) {
        continue;
      }
      const uri = vscode.Uri.joinPath(folder.uri, ...filename.split('/'));
      try {
        return {
          filename,
          content: decoder.decode(await vscode.workspace.fs.readFile(uri)),
        };
      } catch {}
    }
  }
  return null;
}

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server',
    );
  }
  if (!traceOutputChannel) {
    traceOutputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server trace',
    );
  }

  const wasmFileUri = vscode.Uri.joinPath(
    context.extensionUri,
    'dist',
    'pyrefly_wasm_bg.wasm',
  );
  const wasmUri = await vscode.env.asExternalUri(wasmFileUri);
  let wasmBytes: Uint8Array;
  try {
    wasmBytes = await vscode.workspace.fs.readFile(wasmFileUri);
  } catch (error) {
    const message =
      'Pyrefly web failed to load the WASM bundle. Run pyrefly_wasm/build.sh and rebuild the extension.';
    outputChannel.appendLine(`${message} ${String(error)}`);
    void vscode.window.showErrorMessage(message);
    return;
  }

  const worker = createWorker(context);

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{language: 'python'}],
    outputChannel,
    traceOutputChannel,
    initializationOptions: {
      // Prefer bytes so the worker doesn't need to `fetch(...)` with a custom URL scheme.
      wasmBytes,
      // Keep URI as a fallback / for debugging.
      wasmUri: wasmUri.toString(true),
      pythonVersion: configuredPythonVersion(),
      initialFiles: await readInitialConfigFiles(),
    },
  };

  client = new LanguageClient(
    'pyrefly',
    'Pyrefly language server',
    clientOptions,
    worker,
  );
  client.onRequest(READ_WORKSPACE_FILE_REQUEST, readWorkspaceFile);

  const startPromise = client.start();
  context.subscriptions.push({
    dispose: () => {
      void client?.stop();
    },
  });
  await startPromise;
}

export async function deactivate(): Promise<void> {
  await client?.stop();
  outputChannel?.dispose();
  traceOutputChannel?.dispose();
}
