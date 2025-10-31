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

let client: LanguageClient;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

/// Get a setting at the path, or throw an error if it's not set.
function requireSetting<T>(path: string): T {
  const ret: T | undefined = vscode.workspace.getConfiguration().get(path);
  if (ret == undefined) {
    throw new Error(`Setting "${path}" was not configured`);
  }
  return ret;
}

/// Update the status bar based on current configuration
async function updateStatusBar() {
  const document = vscode.window.activeTextEditor?.document;
  if (
    document == null ||
    document.uri.scheme !== 'file' ||
    document.languageId !== 'python'
  ) {
    statusBarItem?.hide();
    return;
  }
  let status;
  try {
    status = await client.sendRequest(
      'pyrefly/textDocument/typeErrorDisplayStatus',
      client.code2ProtocolConverter.asTextDocumentItem(document),
    );
  } catch {
    statusBarItem?.hide();
    return;
  }

  if (!statusBarItem) {
    statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
    );
    statusBarItem.name = 'Pyrefly';
  }

  switch (status) {
    case 'disabled-due-to-missing-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is disabled by default.
Create a [\`pyrefly.toml\`](https://pyrefly.org/en/docs/configuration/) file or set disableTypeErrors to false in settings to show type errors.`);
      break;
    case 'disabled-in-ide-config':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is explicitly disabled.
No errors will be shown even if there is a [\`pyrefly.toml\`](https://pyrefly.org/en/docs/configuration/) file.`);
      break;
    case 'disabled-in-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip = new vscode.MarkdownString(
        `Pyrefly type checking is disabled through a config file.`,
      );
      break;
    case 'enabled-in-ide-config':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is explicitly enabled.\nType errors will always be shown.',
      );
      break;
    case 'enabled-in-config-file':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is enabled through a config file.',
      );
      break;
    default:
      statusBarItem?.hide();
      return;
  }
  statusBarItem.show();
}

async function getDocstringRanges(
  document: vscode.TextDocument,
): Promise<vscode.Range[]> {
  const identifier = client.code2ProtocolConverter.asTextDocumentIdentifier(
    document,
  );
  const response = (await client.sendRequest(
    'pyrefly/textDocument/docstringRanges',
    identifier,
  )) as Array<{
    start: {line: number; character: number};
    end: {line: number; character: number};
  }> | null;

  if (!response) {
    return [];
  }

  return response.map(range => client.protocol2CodeConverter.asRange(range));
}

async function runDocstringFoldingCommand(
  commandId: 'editor.fold' | 'editor.unfold',
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (
    editor == null ||
    editor.document.uri.scheme !== 'file' ||
    editor.document.languageId !== 'python'
  ) {
    return;
  }

  try {
    const ranges = await getDocstringRanges(editor.document);
    if (ranges.length === 0) {
      return;
    }

    const seen = new Set<string>();
    const uniqueRanges = ranges.filter(range => {
      const key = `${range.start.line}:${range.start.character}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });

    const selectionLines = uniqueRanges
      .map(range => {
        if (range.start.line === range.end.line) {
          return null;
        }
        return range.start.line;
      })
      .filter((line): line is number => line != null)
      .filter((line, index, arr) => arr.indexOf(line) === index)
      .sort((a, b) => a - b);

    if (selectionLines.length === 0) {
      return;
    }

    await vscode.commands.executeCommand(commandId, {
      selectionLines,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : `Unknown error: ${String(error)}`;
    outputChannel?.appendLine(
      `Failed to ${commandId === 'editor.fold' ? 'fold' : 'unfold'} docstrings: ${message}`,
    );
  }
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

export async function activate(context: ExtensionContext) {
  // Initialize the output channel if it doesn't exist
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel(
      'Pyrefly language server',
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
    // Register the server for Starlark documents
    documentSelector: [{scheme: 'file', language: 'python'}],
    outputChannel: outputChannel,
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
      await updateStatusBar();
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
      if (event.affectsConfiguration('python.pyrefly') ||
          event.affectsConfiguration('python.analysis')) {
        client.sendNotification(DidChangeConfigurationNotification.type, {
          settings: {},
        });
      }
      await updateStatusBar();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.restartClient', async () => {
      await client.stop();
      // Clear the output channel but don't dispose it
      outputChannel.clear();
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
      await runDocstringFoldingCommand('editor.fold');
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.unfoldAllDocstrings', async () => {
      await runDocstringFoldingCommand('editor.unfold');
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

  // Start the client. This will also launch the server
  await client.start();

  await updateStatusBar();
  context.subscriptions.push(statusBarItem);
}

/**
 * This function will trigger the ms-python extension to reasses which language server to spin up.
 * It does this by changing languageServer setting: this triggers a refresh of active language
 * servers:
 * https://github.com/microsoft/vscode-python/blob/main/src/client/languageServer/watcher.ts#L296
 *
 * We then change the setting back so we don't end up messing up the users settings.
 */
async function triggerMsPythonRefreshLanguageServers() {
  const config = vscode.workspace.getConfiguration('python');
  const setting = 'languageServer';
  let previousSetting = config.get(setting);
  // without the target, we will crash here with "Unable to write to Workspace Settings
  // because no workspace is opened. Please open a workspace first and try again."
  await config.update(
    setting,
    previousSetting === 'None' ? 'Default' : 'None',
    vscode.ConfigurationTarget.Global,
  );
  await config.update(
    setting,
    previousSetting,
    vscode.ConfigurationTarget.Global,
  );
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  // Dispose the output channel when the extension is deactivated
  if (outputChannel) {
    outputChannel.dispose();
  }
  return client.stop();
}
