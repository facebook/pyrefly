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
  TYPE_ERROR_DISPLAY_STATUS_CHANGED_METHOD,
  TYPE_ERROR_DISPLAY_STATUS_VERSION,
  getStatusBarItem,
  scheduleStatusBarUpdate,
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
 * re-reads it on every start, so mutating it in place is how a restart picks up
 * a different binary.
 */
let launchSpec: Executable;
let clientOptions: LanguageClientOptions;
let pythonEnv: PythonEnvironment;
/** Set once activation's own `client.start()` has settled, successfully or not. */
let activated = false;

/** `client.restart()` is not re-entrant, and three callers can ask at once. */
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

/** Re-resolve into the launch specification, reporting whether it changed. */
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
 * Start a server on the current launch specification with a fresh client.
 *
 * A client whose start failed caches the rejection and never reaches `start()`
 * again, so a stopped client cannot be revived, only replaced.
 */
async function replaceClient(): Promise<void> {
  await client.dispose().catch(() => {});
  client = new LanguageClient(
    'pyrefly',
    'Pyrefly language server',
    launchSpec,
    clientOptions,
  );
  try {
    await client.start();
    logServerVersion();
  } finally {
    // Hides the item when the server did not come up.
    await updateStatusBar(client);
  }
}

/** Restart, reverting to `fallback` if the new specification does not come up. */
async function restartOrRevert(fallback: Executable): Promise<boolean> {
  try {
    await client.restart();
    logServerVersion();
    return true;
  } catch (error) {
    const attempted = launchSpec.command;
    outputChannel.appendLine(
      `Could not start the Pyrefly language server at ${attempted}: ${describeError(error)}`,
    );
    // Revert the specification too, so the next change still compares as one.
    launchSpec.command = fallback.command;
    launchSpec.args = fallback.args;
    try {
      await replaceClient();
    } catch (revertError) {
      outputChannel.appendLine(
        `Pyrefly language server is not running: ${describeError(revertError)}`,
      );
    }
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
    // Activation reads the specification when it spawns the server, so recording
    // it above is all a change arriving before then needs.
    if (!activated) {
      return 'not-running';
    }
    if (!changed) {
      return 'unchanged';
    }
    outputChannel.appendLine(
      `Restarting the Pyrefly language server because ${reason}.`,
    );
    if (client.state === State.Running) {
      return (await restartOrRevert(previous)) ? 'restarted' : 'failed';
    }
    // An earlier start left nothing running, so a corrected setting is the cue
    // to try again rather than wait for a window reload.
    try {
      await replaceClient();
      return 'restarted';
    } catch (error) {
      outputChannel.appendLine(
        `Could not start the Pyrefly language server at ${launchSpec.command}: ${describeError(error)}`,
      );
      return 'failed';
    }
  });
}

/**
 * Reaching the restart machinery directly, so that tests drive it instead of
 * racing the configuration listener and polling for the result.
 *
 * `activate` returns this rather than the module exporting it: esbuild bundles
 * each entry point separately, so a test that imported this module would get
 * its own copy of these globals rather than the running extension's.
 */
export interface TestHooks {
  restartIfLaunchSpecChanged: typeof restartIfLaunchSpecChanged;
  clientState: () => State;
  currentCommand: () => string;
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

export async function activate(
  context: ExtensionContext,
): Promise<TestHooks> {
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
      pushTypeErrorDisplayStatus: true,
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

  context.subscriptions.push(
    client.onNotification(TYPE_ERROR_DISPLAY_STATUS_CHANGED_METHOD, () => {
      scheduleStatusBarUpdate(client);
    }),
  );

  pythonEnv
    .onDidChangeInterpreter(async () => {
      const outcome = await restartIfLaunchSpecChanged(
        context.extensionUri,
        'the active Python interpreter changed',
      );
      // A restart picks the interpreter up during initialize. Any other server
      // still running has to be told, including one kept because re-resolution
      // failed.
      if (outcome !== 'restarted' && client.state === State.Running) {
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
  try {
    await client.start();
  } finally {
    // Even a failed start hands responsibility to the listeners above, which
    // can bring a server up once a setting is corrected.
    activated = true;
  }
  logServerVersion();

  await updateStatusBar(client);
  const statusBarItem = getStatusBarItem();
  if (statusBarItem) {
    context.subscriptions.push(statusBarItem);
  }

  return {
    restartIfLaunchSpecChanged,
    clientState: () => client.state,
    currentCommand: () => launchSpec.command,
  };
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
