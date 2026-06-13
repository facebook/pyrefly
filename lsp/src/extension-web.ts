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

  const worker = createWorker(context);
  const wasmFileUri = vscode.Uri.joinPath(
    context.extensionUri,
    'dist',
    'pyrefly_wasm_bg.wasm',
  );
  const wasmUri = await vscode.env.asExternalUri(wasmFileUri);
  const wasmBytes = await vscode.workspace.fs.readFile(wasmFileUri);

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{language: 'python'}],
    outputChannel,
    traceOutputChannel,
    initializationOptions: {
      // Prefer bytes so the worker doesn't need to `fetch(...)` with a custom URL scheme.
      wasmBytes,
      // Keep URI as a fallback / for debugging.
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
}

export async function deactivate(): Promise<void> {
  await client?.stop();
  outputChannel?.dispose();
  traceOutputChannel?.dispose();
}
