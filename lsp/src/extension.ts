/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {ExtensionContext, workspace} from 'vscode';
import * as path from 'path';
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
let runTerminal: vscode.Terminal | undefined;

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
  pythonExtension: PythonExtension,
  configurationItems: ConfigurationItem[],
  configuration: (object | null)[],
): Promise<(object | null)[]> {
  const getPythonPathForConfigurationItem = async (index: number) => {
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

function shellQuote(value: string): string {
  if (value.length === 0) {
    return "''";
  }
  if (process.platform === 'win32') {
    return `"${value.replace(/"/g, '\\"')}"`;
  }
  return "'" + value.replace(/'/g, "'\"'\"'") + "'";
}

function getRunTerminal(cwd?: string): vscode.Terminal {
  if (!runTerminal) {
    runTerminal = vscode.window.createTerminal({
      name: 'Pyrefly Run',
      cwd,
    });
  }
  return runTerminal;
}

function moduleNameFromPath(uri: vscode.Uri): string | undefined {
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
  if (!workspaceFolder) {
    return undefined;
  }
  let relativePath = path.relative(workspaceFolder.uri.fsPath, uri.fsPath);
  if (relativePath.startsWith('..')) {
    return undefined;
  }
  if (relativePath.endsWith('.py')) {
    relativePath = relativePath.slice(0, -3);
  }
  return relativePath
    .split(path.sep)
    .filter(part => part.length > 0)
    .join('.');
}

async function runTestAtLocation(
  args: {
    uri?: string;
    position?: {line: number; character: number};
    testName?: string;
    className?: string;
    isUnittest?: boolean;
  },
  pythonExtension: PythonExtension,
) {
  if (!args.uri) {
    return;
  }
  const uri = vscode.Uri.parse(args.uri);
  const envPath = await pythonExtension.environments.getActiveEnvironmentPath(
    uri,
  );
  const pythonPath = envPath.path;
  const interpreter = pythonPath.length > 0 ? pythonPath : 'python';
  const cwd = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
  const terminal = getRunTerminal(cwd);

  const className = args.className;
  const testName = args.testName;
  const isUnittest = args.isUnittest === true;

  if (args.position && !testName && !className) {
    const document = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(document, {
      preview: false,
    });
    const position = new vscode.Position(
      args.position.line,
      args.position.character,
    );
    editor.selection = new vscode.Selection(position, position);
    editor.revealRange(new vscode.Range(position, position));
    await vscode.commands.executeCommand('testing.runAtCursor');
    return;
  }

  let command: string | undefined;
  if (isUnittest) {
    const moduleName = moduleNameFromPath(uri);
    if (moduleName) {
      let target = moduleName;
      if (className) {
        target = `${target}.${className}`;
      }
      if (testName) {
        target = `${target}.${testName}`;
      }
      command = `${shellQuote(interpreter)} -m unittest ${shellQuote(target)}`;
    } else {
      command = `${shellQuote(interpreter)} -m unittest ${shellQuote(uri.fsPath)}`;
    }
  } else {
    let nodeId = uri.fsPath;
    if (className) {
      nodeId = `${nodeId}::${className}`;
    }
    if (testName) {
      nodeId = `${nodeId}::${testName}`;
    }
    command = `${shellQuote(interpreter)} -m pytest ${shellQuote(nodeId)}`;
  }

  terminal.show(true);
  terminal.sendText(command ?? `${shellQuote(interpreter)} -m pytest`);
}

async function runMainFile(
  args: {uri?: string},
  pythonExtension: PythonExtension,
) {
  if (!args.uri) {
    return;
  }
  const uri = vscode.Uri.parse(args.uri);
  const envPath = await pythonExtension.environments.getActiveEnvironmentPath(
    uri,
  );
  const pythonPath = envPath.path;
  const interpreter = pythonPath.length > 0 ? pythonPath : 'python';
  const command = `${shellQuote(interpreter)} ${shellQuote(uri.fsPath)}`;
  const cwd = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
  const terminal = getRunTerminal(cwd);
  terminal.show(true);
  terminal.sendText(command);
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

  let pythonExtension = await PythonExtension.api();

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

  context.subscriptions.push(
    pythonExtension.environments.onDidChangeActiveEnvironmentPath(() => {
      client.sendNotification(DidChangeConfigurationNotification.type, {
        settings: {},
      });
    }),
  );

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

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.runTest', async args => {
      await runTestAtLocation(
        args as {
          uri?: string;
          position?: {line: number; character: number};
          testName?: string;
          className?: string;
          isUnittest?: boolean;
        },
        pythonExtension,
      );
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.runMain', async args => {
      await runMainFile(args as {uri?: string}, pythonExtension);
    }),
  );

  // When our extension is activated, make sure ms-python knows
  // TODO(kylei): remove this hack once ms-python has this behavior
  await triggerMsPythonRefreshLanguageServers();

  vscode.workspace.onDidChangeConfiguration(async e => {
    if (e.affectsConfiguration(`python.pyrefly.disableLanguageServices`)) {
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
  if (runTerminal) {
    runTerminal.dispose();
    runTerminal = undefined;
  }
  return client.stop();
}
