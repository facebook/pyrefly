/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {execFile} from 'child_process';
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
  cwd?: string;
};

type RunTestArgs = {
  uri?: string;
  cwd?: string;
  position?: CodeLensPosition;
  testName?: string;
  className?: string;
  isUnittest?: boolean;
};

const TASK_SOURCE = 'pyrefly';
const OPEN_RUNNABLE_CODE_LENS_SETTING = 'Open Runnable CodeLens Setting';
const DISABLE_RUNNABLE_CODE_LENS = 'Disable Runnable CodeLens';
const shownRunnableCodeLensErrors = new Set<string>();

function asObject(value: unknown): Record<string, unknown> | undefined {
  return value != null && typeof value === 'object' ? (value as Record<string, unknown>) : undefined;
}

function parsePosition(value: unknown): CodeLensPosition | undefined {
  const position = asObject(value);
  if (!position) {
    return undefined;
  }
  const line = position.line;
  const character = position.character;
  return typeof line === 'number' && typeof character === 'number'
    ? {line, character}
    : undefined;
}

function parseRunMainArgs(args: unknown): RunMainArgs | undefined {
  const parsed = asObject(args);
  if (!parsed) {
    return undefined;
  }
  return {
    uri: typeof parsed.uri === 'string' ? parsed.uri : undefined,
    cwd: typeof parsed.cwd === 'string' ? parsed.cwd : undefined,
  };
}

function parseRunTestArgs(args: unknown): RunTestArgs | undefined {
  const parsed = asObject(args);
  if (!parsed) {
    return undefined;
  }
  return {
    uri: typeof parsed.uri === 'string' ? parsed.uri : undefined,
    cwd: typeof parsed.cwd === 'string' ? parsed.cwd : undefined,
    position: parsePosition(parsed.position),
    testName: typeof parsed.testName === 'string' ? parsed.testName : undefined,
    className: typeof parsed.className === 'string' ? parsed.className : undefined,
    isUnittest:
      typeof parsed.isUnittest === 'boolean' ? parsed.isUnittest : undefined,
  };
}

function parseUri(rawUri: string | undefined): vscode.Uri | undefined {
  if (!rawUri) {
    return undefined;
  }
  try {
    return vscode.Uri.parse(rawUri);
  } catch {
    return undefined;
  }
}

async function interpreterForUri(
  uri: vscode.Uri,
  pythonExtension: PythonExtension,
): Promise<string | undefined> {
  const envPath = await pythonExtension.environments.getActiveEnvironmentPath(
    uri,
  );
  return envPath.path.length > 0 ? envPath.path : undefined;
}

function configurationTargetForUri(uri: vscode.Uri): vscode.ConfigurationTarget {
  return vscode.workspace.getWorkspaceFolder(uri) != null
    ? vscode.ConfigurationTarget.WorkspaceFolder
    : vscode.ConfigurationTarget.Workspace;
}

async function disableRunnableCodeLens(uri: vscode.Uri): Promise<void> {
  await vscode.workspace
    .getConfiguration('python.pyrefly', uri)
    .update(
      'runnableCodeLens',
      false,
      configurationTargetForUri(uri),
    );
}

async function showRunnableCodeLensError(
  uri: vscode.Uri,
  kind: string,
  message: string,
): Promise<void> {
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
  const errorKey = `${kind}:${workspaceFolder?.uri.toString() ?? 'workspace'}`;
  if (shownRunnableCodeLensErrors.has(errorKey)) {
    return;
  }
  shownRunnableCodeLensErrors.add(errorKey);
  const action = await vscode.window.showErrorMessage(
    message,
    OPEN_RUNNABLE_CODE_LENS_SETTING,
    DISABLE_RUNNABLE_CODE_LENS,
  );
  if (action === OPEN_RUNNABLE_CODE_LENS_SETTING) {
    await vscode.commands.executeCommand(
      'workbench.action.openSettings',
      'python.pyrefly.runnableCodeLens',
    );
  } else if (action === DISABLE_RUNNABLE_CODE_LENS) {
    await disableRunnableCodeLens(uri);
  }
}

async function canImportModule(
  interpreter: string,
  moduleName: string,
  cwd: string | undefined,
): Promise<boolean> {
  return await new Promise(resolve => {
    execFile(
      interpreter,
      ['-c', `import ${moduleName}`],
      {cwd},
      error => resolve(error == null),
    );
  });
}

function scopeForUri(uri: vscode.Uri): vscode.WorkspaceFolder | vscode.TaskScope {
  return vscode.workspace.getWorkspaceFolder(uri) ?? vscode.TaskScope.Workspace;
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
  cwd: string | undefined,
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
      cwd,
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
  const uri = parseUri(args.uri);
  if (!uri) {
    return;
  }
  const interpreter = await interpreterForUri(uri, pythonExtension);
  if (!interpreter) {
    void showRunnableCodeLensError(
      uri,
      'missing-interpreter',
      'Pyrefly could not determine a Python interpreter for this file. Ensure the correct interpreter is selected in your IDE before using runnable CodeLens.',
    );
    return;
  }
  await executeProcessTask(
    uri,
    args.cwd,
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
  const uri = parseUri(args.uri);
  if (!uri) {
    return;
  }
  const cwd = args.cwd;
  const className = args.className;
  const testName = args.testName;

  if (args.position && !testName && !className) {
    await runAtCursor(uri, args.position);
    return;
  }

  const interpreter = await interpreterForUri(uri, pythonExtension);
  if (!interpreter) {
    void showRunnableCodeLensError(
      uri,
      'missing-interpreter',
      'Pyrefly could not determine a Python interpreter for this file. Ensure the correct interpreter is selected in your IDE before using runnable CodeLens.',
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
      cwd,
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
  if (!(await canImportModule(interpreter, 'pytest', cwd))) {
    void showRunnableCodeLensError(
      uri,
      'missing-pytest',
      'Pyrefly could not import pytest from the selected interpreter. Select the correct interpreter or install pytest in that environment before using runnable CodeLens.',
    );
    return;
  }
  await executeProcessTask(
    uri,
    cwd,
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
      const parsedArgs = parseRunTestArgs(args);
      if (!parsedArgs) {
        return;
      }
      await runTestAtLocation(parsedArgs, pythonExtension);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('pyrefly.runMain', async args => {
      const parsedArgs = parseRunMainArgs(args);
      if (!parsedArgs) {
        return;
      }
      await runMainFile(parsedArgs, pythonExtension);
    }),
  );
}
