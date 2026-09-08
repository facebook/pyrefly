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
import {execFile} from 'child_process';
import {basename, dirname} from 'path';
import {
  CancellationToken,
  ConfigurationItem,
  ConfigurationParams,
  ConfigurationRequest,
  DidChangeConfigurationNotification,
  Executable,
  LanguageClient,
  LanguageClientOptions,
  LSPAny,
  ResponseError,
  ServerOptions,
  State,
} from 'vscode-languageclient/node';
import {
  TYPE_ERROR_DISPLAY_STATUS_VERSION,
  getStatusBarItem,
  updateStatusBar,
} from './status-bar';
import {runDocstringFoldingCommand} from './docstring';
import {registerCodeLensCommands} from './codeLens';
import {registerHoverProvider} from './hover';
import {PythonEnvironment} from './python-environment';
import {
  triggerMsPythonRefreshLanguageServersIfInstalled,
} from './extension-interop';
import {describeError, resolveExecutable} from './lspPath';

let client: LanguageClient;
let outputChannel: vscode.OutputChannel;
let traceOutputChannel: vscode.OutputChannel;
let inferOutputChannel: vscode.OutputChannel;
/**
 * The launch specification `client` was constructed with. `LanguageClient`
 * re-reads it on every start, so mutating it in place is what lets a restart
 * pick up a binary resolved from a different Python environment.
 */
let launchSpec: Executable;
let clientOptions: LanguageClientOptions;
let pythonEnv: PythonEnvironment;

/**
 * Restarts are serialized: `client.restart()` is not re-entrant, and the
 * restart command, an interpreter change and a settings change can each ask
 * for one at any time.
 */
let restartQueue: Promise<unknown> = Promise.resolve();

function queueRestart<T>(work: () => Promise<T>): Promise<T> {
  const next = restartQueue.catch(() => {}).then(work);
  restartQueue = next;
  return next;
}

function logServerVersion(): void {
  outputChannel.appendLine(
    `Pyrefly language server version: ${client.initializeResult?.serverInfo?.version ?? '<unknown>'}`,
  );
}

/**
 * Re-resolve the binary and arguments into the launch specification. Returns
 * whether either changed, which is exactly when a restart is needed for the
 * new selection to take effect.
 */
async function refreshLaunchSpec(extensionUri: vscode.Uri): Promise<boolean> {
  const next = await resolveExecutable(extensionUri, pythonEnv, outputChannel);
  const changed =
    next.command !== launchSpec.command ||
    JSON.stringify(next.args) !== JSON.stringify(launchSpec.args);
  launchSpec.command = next.command;
  launchSpec.args = next.args;
  return changed;
}

/**
 * Restart the client, reverting to `fallback` if the new specification does
 * not come up. Returns whether the restart succeeded as asked.
 *
 * A `LanguageClient` whose start fails moves to `StartFailed` and caches the
 * rejection, so it never reaches `start()` again — neither a corrected setting
 * nor the restart command could recover it. Reverting therefore means building
 * a fresh client rather than restarting this one.
 */
async function restartOrRevert(fallback: Executable): Promise<boolean> {
  try {
    await client.restart();
    logServerVersion();
    return true;
  } catch (error) {
    const attempted = launchSpec.command;
    const message = describeError(error);
    outputChannel.appendLine(
      `Could not start the Pyrefly language server at ${attempted}: ${message}`,
    );
    // Put the specification back too, so that the next interpreter or settings
    // change is still seen as a change and gets to retry. Leaving the failed
    // selection in place would make every later comparison report "no change".
    launchSpec.command = fallback.command;
    launchSpec.args = fallback.args;
    try {
      // Be safe in case client.dispose() runs synchronously.
      await client.dispose().catch(() => {});
    } catch {}
    client = new LanguageClient(
      'pyrefly',
      'Pyrefly language server',
      launchSpec,
      clientOptions,
    );
    try {
      await client.start();
      logServerVersion();
    } catch (revertError) {
      outputChannel.appendLine(
        `Pyrefly language server is not running: ${describeError(revertError)}`,
      );
    }
    // The status bar is still describing the client we just disposed. Refreshing
    // it against the replacement covers both outcomes above: `updateStatusBar`
    // hides the item when the server does not answer.
    await updateStatusBar(client);
    void vscode.window.showErrorMessage(
      `Pyrefly could not start ${attempted}, and went back to ${fallback.command}. See the "Pyrefly language server" output for details.`,
    );
    return false;
  }
}

/** Restart the client if the launch specification changed. */
async function restartIfLaunchSpecChanged(
  extensionUri: vscode.Uri,
  reason: string,
): Promise<'restarted' | 'unchanged' | 'failed' | 'not-running'> {
  return queueRestart(async () => {
    const previous: Executable = {
      command: launchSpec.command,
      args: launchSpec.args,
    };
    let changed: boolean;
    try {
      changed = await refreshLaunchSpec(extensionUri);
    } catch (error) {
      outputChannel.appendLine(
        `Could not resolve the Pyrefly binary, keeping ${launchSpec.command}: ${describeError(error)}`,
      );
      return 'failed';
    }
    // Both callers are registered before activation's own `client.start()`, so
    // a change can arrive with no server to restart. Recording it above is what
    // makes it take effect anyway: the client reads the launch specification
    // when it spawns the process, which has not happened yet.
    if (client.state !== State.Running) {
      return 'not-running';
    }
    if (!changed) {
      return 'unchanged';
    }
    outputChannel.appendLine(
      `Restarting the Pyrefly language server because ${reason}.`,
    );
    return (await restartOrRevert(previous)) ? 'restarted' : 'failed';
  });
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
 */
async function overridePythonPath(
  configurationItems: ConfigurationItem[],
  configuration: (object | null)[],
): Promise<(object | null)[]> {
  const newResult = await Promise.all(
    configuration.map(async (item, index) => {
      if (
        configurationItems.length <= index ||
        configurationItems[index].section !== 'python'
      ) {
        return item;
      }
      const scopeUri = configurationItems[index].scopeUri;
      const pythonPath = await pythonEnv.getInterpreterPath(
        scopeUri === undefined ? undefined : vscode.Uri.parse(scopeUri),
      );
      if (pythonPath === undefined) {
        return item;
      }
      return {...item, pythonPath};
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
  if (!inferOutputChannel) {
    inferOutputChannel = vscode.window.createOutputChannel('Pyrefly infer');
  }

  pythonEnv = new PythonEnvironment(context);

  launchSpec = await resolveExecutable(
    context.extensionUri,
    pythonEnv,
    outputChannel,
  );

  // `getConfiguration` returns a `WorkspaceConfiguration` proxy, not a
  // plain object: spread (`{...cfg}`) and `Object.assign({}, cfg)` rely
  // on own enumerable properties and may silently drop the configured
  // values. JSON-roundtrip via the proxy's `toJSON` (the same path
  // `vscode-languageclient` itself takes when serializing
  // `initializationOptions`) gives us a faithful plain object to merge
  // with.
  const rawInitialisationOptions = JSON.parse(
    JSON.stringify(vscode.workspace.getConfiguration('pyrefly') ?? {}),
  );
  // Proposed APIs are omitted at runtime when the editor has not granted access.
  // In that case, let vscode-languageclient register the ordinary LSP hover provider.
  const supportsHoverVerbosity = vscode.VerboseHover !== undefined;

  // Opt into the V2 wire shape for the typeErrorDisplayStatus request.
  // An older binary that doesn't know V2 still returns its V1 bare
  // string when the field is absent / unrecognized, so declaring V2 is
  // safe even against pre-V2 binaries — V1's bare-string response is
  // distinguishable by shape (`typeof resp === 'string'`) and the V1
  // renderer below handles it.
  const initializationOptions = {
    ...rawInitialisationOptions,
    pyrefly: {
      ...((rawInitialisationOptions as any).pyrefly ?? {}),
      typeErrorDisplayStatusVersion: TYPE_ERROR_DISPLAY_STATUS_VERSION,
      customHoverProvider: supportsHoverVerbosity,
    },
  };

  // Options to control the language client
  clientOptions = {
    initializationOptions,
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
    // Support for any notebook type
    // @ts-ignore
    notebookDocumentSync: {
      notebookSelector: [
        {
          notebook: '*',
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
          return await overridePythonPath(
            params.items,
            result as (object | null)[],
          );
        },
      },
    },
  };

  const serverOptions: ServerOptions = launchSpec;

  // Create the language client and start the client.
  client = new LanguageClient(
    'pyrefly',
    'Pyrefly language server',
    serverOptions,
    clientOptions,
  );
  if (supportsHoverVerbosity) {
    registerHoverProvider(context, () => client);
  }

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(async () => {
      await updateStatusBar(client);
    }),
  );

  pythonEnv
    .onDidChangeInterpreter(async () => {
      // A new environment may ship its own Pyrefly. If it does not, the server
      // stays up and only needs to hear about the new interpreter. A failed
      // restart leaves no running server to notify.
      const outcome = await restartIfLaunchSpecChanged(
        context.extensionUri,
        'the active Python interpreter changed',
      );
      if (outcome === 'unchanged') {
        client.sendNotification(DidChangeConfigurationNotification.type, {
          settings: {},
        });
      }
    })
    .then(disposable => {
      if (disposable) {
        context.subscriptions.push(disposable);
      }
    });

  context.subscriptions.push(
    workspace.onDidChangeConfiguration(async event => {
      if (event.affectsConfiguration('python.pyrefly')) {
        client.sendNotification(DidChangeConfigurationNotification.type, {
          settings: {},
        });
      }
      if (
        event.affectsConfiguration('pyrefly.lspPath') ||
        event.affectsConfiguration('pyrefly.pyreflyExecutable') ||
        event.affectsConfiguration('pyrefly.lspArguments')
      ) {
        await restartIfLaunchSpecChanged(
          context.extensionUri,
          'the configured Pyrefly executable changed',
        );
      }
      await updateStatusBar(client);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.restartClient', async () => {
      // Clear the output channel but don't dispose it
      outputChannel.clear();
      traceOutputChannel.clear();
      await queueRestart(async () => {
        const previous: Executable = {
          command: launchSpec.command,
          args: launchSpec.args,
        };
        try {
          await refreshLaunchSpec(context.extensionUri);
        } catch (error) {
          // This command is the escape hatch from a bad selection, so restart
          // with the previous one rather than not restarting at all.
          outputChannel.appendLine(
            `Could not re-resolve the Pyrefly binary, restarting with the previous one: ${describeError(error)}`,
          );
        }
        await restartOrRevert(previous);
      });
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
    vscode.commands.registerCommand('pyrefly.infer', async () => {
      const document = vscode.window.activeTextEditor?.document;
      if (
        document === undefined ||
        document.languageId !== 'python' ||
        document.uri.scheme !== 'file'
      ) {
        await vscode.window.showErrorMessage(
          'Open a saved Python file before running Pyrefly infer.',
        );
        return;
      }
      if (!(await document.save())) {
        await vscode.window.showErrorMessage(
          `Pyrefly could not save ${basename(document.uri.fsPath)} before inferring types.`,
        );
        return;
      }

      const cwd =
        vscode.workspace.getWorkspaceFolder(document.uri)?.uri.fsPath ??
        dirname(document.uri.fsPath);
      inferOutputChannel.clear();
      try {
        await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Pyrefly: Inferring types in ${basename(document.uri.fsPath)}`,
          },
          async () => {
            await new Promise<void>((resolve, reject) => {
              execFile(
                launchSpec.command,
                ['infer', document.uri.fsPath],
                {cwd},
                (error, stdout, stderr) => {
                  inferOutputChannel.append(stdout);
                  inferOutputChannel.append(stderr);
                  if (error) {
                    reject(error);
                  } else {
                    resolve();
                  }
                },
              );
            });
          },
        );
      } catch (error) {
        inferOutputChannel.show(true);
        const message = error instanceof Error ? error.message : String(error);
        await vscode.window.showErrorMessage(
          `Pyrefly could not infer types in ${basename(document.uri.fsPath)}: ${message}`,
        );
      }
    }),
  );
  registerCodeLensCommands(context, pythonEnv);

  // When our extension is activated, make sure ms-python knows
  // TODO(kylei): remove this hack once ms-python has this behavior
  await triggerMsPythonRefreshLanguageServersIfInstalled();

  vscode.workspace.onDidChangeConfiguration(async e => {
    if (e.affectsConfiguration(`python.pyrefly.disableLanguageServices`)) {
      // TODO(kylei): remove this hack once ms-python has this behavior
      await triggerMsPythonRefreshLanguageServersIfInstalled();
    }
  });

  // Start the client. This will also launch the server
  await client.start();
  logServerVersion();

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
  if (inferOutputChannel) {
    inferOutputChannel.dispose();
  }
  return client.stop();
}
