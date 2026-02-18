/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {ExtensionContext, workspace} from 'vscode';
import * as vscode from 'vscode';
import {
  CancellationToken,
  ConfigurationItem,
  ConfigurationParams,
  ConfigurationRequest,
  DidChangeConfigurationNotification,
  LanguageClient,
  LanguageClientOptions,
  LSPAny,
  ResponseError,
  ServerOptions,
} from 'vscode-languageclient/node';
import {PythonExtension} from '@vscode/python-extension';
import {updateStatusBar, getStatusBarItem} from './status-bar';
import {runDocstringFoldingCommand} from './docstring';
import {
  triggerMsPythonRefreshLanguageServers,
  disableWindsurfPyrightIfInstalled,
  disableBasedPyrightIfInstalled,
  disableCursorPyrightIfInstalled,
} from './extension-interop';

let client: LanguageClient;
let outputChannel: vscode.OutputChannel;
let traceOutputChannel: vscode.OutputChannel;

/// Get a setting at the path, or throw an error if it's not set.
function requireSetting<T>(path: string): T {
  const ret: T | undefined = vscode.workspace.getConfiguration().get(path);
  if (ret == undefined) {
    throw new Error(`Setting "${path}" was not configured`);
  }
  return ret;
}


/**
 * This function adds the pythonPath to any section with configuration of 'python'.
 * Our language server expects the pythonPath from VSCode configurations but this setting is not stored in VSCode
 * configurations. The Python extension used to store pythonPath in this section but no longer does. Details:
 * https://github.com/microsoft/pyright/commit/863721687bc85a54880423791c79969778b19a3f
 *
 * Example:
 * - Pyrefly asks for a configurationItem for {scopeUri: '/home/project', section: 'python'}
 * - VSCode returns a configuration of {setting: 'value'} from settings.json
 * - This function will add pythonPath: '/usr/bin/python3' from the Python extension to the configuration
 * - {setting: 'value', pythonPath: '/usr/bin/python3'} is returned
 *
 * @param pythonExtension the python extension API
 * @param configurationItems the sections within the workspace
 * @param configuration the configuration returned by vscode in response to a workspace/configuration request (usually what's in settings.json)
 * corresponding to the sections described in configurationItems
 */
async function overridePythonPath(
  pythonExtension: PythonExtension | undefined,
  configurationItems: ConfigurationItem[],
  configuration: (object | null)[],
): Promise<(object | null)[]> {
  const getPythonPathForConfigurationItem = async (index: number) => {
    if (!pythonExtension) {
      return undefined;
    }
    if (
      configurationItems.length <= index ||
      configurationItems[index].section !== 'python'
    ) {
      return undefined;
    }
    let scopeUri = configurationItems[index].scopeUri;
    return await pythonExtension.environments.getActiveEnvironmentPath(
      scopeUri === undefined ? undefined : vscode.Uri.parse(scopeUri),
    ).path;
  };
  const newResult = await Promise.all(
    configuration.map(async (item, index) => {
      const pythonPath = await getPythonPathForConfigurationItem(index);
      if (pythonPath === undefined) {
        return item;
      } else {
        return {...item, pythonPath};
      }
    }),
  );
  return newResult;
}

export async function activate(context: ExtensionContext) {
  // Initialize the output channel if it doesn't exist
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server',
    );
  }

  // Initialize the trace output channel for separate trace logs
  if (!traceOutputChannel) {
    traceOutputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server trace',
    );
  }

  const path: string = requireSetting('pyrefly.lspPath');
  const args: [string] = requireSetting('pyrefly.lspArguments');

  const bundledPyreflyPath = vscode.Uri.joinPath(
    context.extensionUri,
    'bin',
    // process.platform returns win32 on any windows CPU architecture
    process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly',
  );

  let pythonExtension: PythonExtension | undefined;
  try {
    pythonExtension = await PythonExtension.api();
  } catch (error) {
    vscode.window.showWarningMessage(
      'Pyrefly: Python extension not available. Python interpreter detection will be limited.',
    );
  }

  // Otherwise to spawn the server
  let serverOptions: ServerOptions = {
    command: path === '' ? bundledPyreflyPath.fsPath : path,
    args: args,
  };
  let rawInitialisationOptions = vscode.workspace.getConfiguration('pyrefly');

  // Options to control the language client
  let clientOptions: LanguageClientOptions = {
    initializationOptions: rawInitialisationOptions,
    // Register the server for Python documents
    documentSelector: [
      {scheme: 'file', language: 'python'},
      // Support for unsaved/untitled files
      {scheme: 'untitled', language: 'python'},
      // Support for notebook cells
      {scheme: 'vscode-notebook-cell', language: 'python'},
      // Support for in-memory documents like the Positron Console
      {scheme: 'inmemory', language: 'python'},
    ],
    // Support for notebooks
    // @ts-ignore
    notebookDocumentSync: {
      notebookSelector: [
        {
          notebook: {notebookType: 'jupyter-notebook'},
          cells: [{language: 'python'}],
        },
      ],
    },
    outputChannel: outputChannel,
    traceOutputChannel: traceOutputChannel,
    middleware: {
      workspace: {
        configuration: async (
          params: ConfigurationParams,
          token: CancellationToken,
          next: ConfigurationRequest.HandlerSignature,
        ): Promise<LSPAny[] | ResponseError<void>> => {
          const result = await next(params, token);
          if (result instanceof ResponseError) {
            return result;
          }
          const newResult = await overridePythonPath(
            pythonExtension,
            params.items,
            result as (object | null)[],
          );
          return newResult;
        },
      },
    },
  };

  // Create the language client and start the client.
  client = new LanguageClient(
    'pyrefly',
    'Pyrefly language server',
    serverOptions,
    clientOptions,
  );

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(async () => {
      await updateStatusBar(client);
    }),
  );

  if (pythonExtension) {
    context.subscriptions.push(
      pythonExtension.environments.onDidChangeActiveEnvironmentPath(() => {
        client.sendNotification(DidChangeConfigurationNotification.type, {
          settings: {},
        });
      }),
    );
  }

  context.subscriptions.push(
    workspace.onDidChangeConfiguration(async event => {
      if (event.affectsConfiguration('python.pyrefly')) {
        client.sendNotification(DidChangeConfigurationNotification.type, {
          settings: {},
        });
      }
      await updateStatusBar(client);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.restartClient', async () => {
      await client.stop();
      // Clear the output channel but don't dispose it
      outputChannel.clear();
      traceOutputChannel.clear();
      client = new LanguageClient(
        'pyrefly',
        'Pyrefly language server',
        serverOptions,
        clientOptions,
      );
      await client.start();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.foldAllDocstrings', async () => {
      await runDocstringFoldingCommand(client, outputChannel, 'editor.fold');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.unfoldAllDocstrings', async () => {
      await runDocstringFoldingCommand(client, outputChannel, 'editor.unfold');
    }),
  );

  // When our extension is activated, make sure ms-python knows
  // TODO(kylei): remove this hack once ms-python has this behavior
  if (pythonExtension) {
    await triggerMsPythonRefreshLanguageServers();
  }

  vscode.workspace.onDidChangeConfiguration(async e => {
    if (
      pythonExtension &&
      e.affectsConfiguration(`python.pyrefly.disableLanguageServices`)
    ) {
      // TODO(kylei): remove this hack once ms-python has this behavior
      await triggerMsPythonRefreshLanguageServers();
    }
  });

  // Disable Windsurf Pyright language services if the extension is installed
  await disableWindsurfPyrightIfInstalled();

  // Disable Cursor Pyright language services if the extension is installed
  await disableCursorPyrightIfInstalled();

  // Disable Based Pyright language services if the extension is installed and Pyrefly is enabled
  const pyreflyDisabled = vscode.workspace
    .getConfiguration('python.pyrefly')
    .get<boolean>('disableLanguageServices', false);
  if (!pyreflyDisabled) {
    await disableBasedPyrightIfInstalled();
  }

  // Start the client. This will also launch the server
  await client.start();

  await updateStatusBar(client);
  const statusBarItem = getStatusBarItem();
  if (statusBarItem) {
    context.subscriptions.push(statusBarItem);
  }
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  // Dispose the output channels when the extension is deactivated
  if (outputChannel) {
    outputChannel.dispose();
  }
  if (traceOutputChannel) {
    traceOutputChannel.dispose();
  }
  return client.stop();
}
