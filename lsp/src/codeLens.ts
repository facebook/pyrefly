/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as path from 'path';
import * as vscode from 'vscode';
import {ExtensionContext} from 'vscode';
import {PythonExtension} from '@vscode/python-extension';

type CodeLensPosition = {
  line: number;
  character: number;
};

type RunMainArgs = {
  uri?: string;
};

type RunTestArgs = {
  uri?: string;
  position?: CodeLensPosition;
  testName?: string;
  className?: string;
  isUnittest?: boolean;
};

const TASK_SOURCE = 'pyrefly';

async function interpreterForUri(
  uri: vscode.Uri,
  pythonExtension: PythonExtension,
): Promise<string | undefined> {
  const envPath = await pythonExtension.environments.getActiveEnvironmentPath(
    uri,
  );
  return envPath.path.length > 0 ? envPath.path : undefined;
}

function scopeForUri(uri: vscode.Uri): vscode.WorkspaceFolder | vscode.TaskScope {
  return vscode.workspace.getWorkspaceFolder(uri) ?? vscode.TaskScope.Workspace;
}

function cwdForUri(uri: vscode.Uri): string | undefined {
  return vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
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
  } else {
    return undefined;
  }
  return relativePath
    .split(path.sep)
    .filter(part => part.length > 0)
    .join('.');
}

async function runAtCursor(
  uri: vscode.Uri,
  position: CodeLensPosition,
): Promise<void> {
  const document = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(document, {
    preview: false,
  });
  const cursor = new vscode.Position(position.line, position.character);
  editor.selection = new vscode.Selection(cursor, cursor);
  editor.revealRange(new vscode.Range(cursor, cursor));
  await vscode.commands.executeCommand('testing.runAtCursor');
}

async function executeProcessTask(
  uri: vscode.Uri,
  definition: vscode.TaskDefinition,
  label: string,
  command: string,
  args: string[],
): Promise<void> {
  // Use ProcessExecution so VS Code passes argv directly instead of relying on
  // shell-specific quoting rules.
  const task = new vscode.Task(
    definition,
    scopeForUri(uri),
    label,
    TASK_SOURCE,
    new vscode.ProcessExecution(command, args, {
      cwd: cwdForUri(uri),
    }),
  );
  task.presentationOptions = {
    reveal: vscode.TaskRevealKind.Always,
    panel: vscode.TaskPanelKind.Dedicated,
    focus: true,
    clear: false,
    showReuseMessage: false,
  };
  await vscode.tasks.executeTask(task);
}

async function runMainFile(
  args: RunMainArgs,
  pythonExtension: PythonExtension,
): Promise<void> {
  if (!args.uri) {
    return;
  }
  const uri = vscode.Uri.parse(args.uri);
  const interpreter = await interpreterForUri(uri, pythonExtension);
  if (!interpreter) {
    void vscode.window.showErrorMessage(
      'Pyrefly could not determine a Python interpreter for this file.',
    );
    return;
  }
  await executeProcessTask(
    uri,
    {type: TASK_SOURCE, action: 'runMain'},
    'Pyrefly: Run File',
    interpreter,
    [uri.fsPath],
  );
}

async function runTestAtLocation(
  args: RunTestArgs,
  pythonExtension: PythonExtension,
): Promise<void> {
  if (!args.uri) {
    return;
  }
  const uri = vscode.Uri.parse(args.uri);
  const className = args.className;
  const testName = args.testName;

  if (args.position && !testName && !className) {
    await runAtCursor(uri, args.position);
    return;
  }

  const interpreter = await interpreterForUri(uri, pythonExtension);
  if (!interpreter) {
    void vscode.window.showErrorMessage(
      'Pyrefly could not determine a Python interpreter for this file.',
    );
    return;
  }
  if (args.isUnittest === true) {
    const moduleName = moduleNameFromPath(uri);
    if (!moduleName) {
      if (args.position) {
        await runAtCursor(uri, args.position);
      }
      return;
    }
    let target = moduleName;
    if (className) {
      target = `${target}.${className}`;
    }
    if (testName) {
      target = `${target}.${testName}`;
    }
    await executeProcessTask(
      uri,
      {type: TASK_SOURCE, action: 'runUnittest'},
      'Pyrefly: Run Test',
      interpreter,
      ['-m', 'unittest', target],
    );
    return;
  }

  let nodeId = uri.fsPath;
  if (className) {
    nodeId = `${nodeId}::${className}`;
  }
  if (testName) {
    nodeId = `${nodeId}::${testName}`;
  }
  await executeProcessTask(
    uri,
    {type: TASK_SOURCE, action: 'runPytest'},
    'Pyrefly: Run Test',
    interpreter,
    ['-m', 'pytest', nodeId],
  );
}

export function registerCodeLensCommands(
  context: ExtensionContext,
  pythonExtension: PythonExtension,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.runTest', async args => {
      await runTestAtLocation(args as RunTestArgs, pythonExtension);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.runMain', async args => {
      await runMainFile(args as RunMainArgs, pythonExtension);
    }),
  );
}
