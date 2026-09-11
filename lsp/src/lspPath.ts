/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {execFile} from 'child_process';
import {homedir} from 'os';
import {join, parse, resolve} from 'path';
import {promisify} from 'util';
import * as vscode from 'vscode';
import {Executable} from 'vscode-languageclient/node';
import {PythonEnvironment} from './python-environment';

const execFileAsync = promisify(execFile);

function requireSettingOrUndefined<T>(path: string): T | undefined {
  return vscode.workspace.getConfiguration().get(path);
}

function requireSettingOrDefault<T>(path: string, default: T): T {
  const ret: T | undefined = requireSettingOrUndefined(path);
  if (ret == undefined) {
    return default;
  }
  return ret;
}

/// Get a setting at the path, or throw an error if it's not set.
export function requireSetting<T>(path: string): T {
  const ret: T | undefined = requireSettingOrUndefined(path);
  if (ret == undefined) {
    throw new Error(`Setting "${path}" was not configured`);
  }
  return ret;
}

/**
 * Resolve a `pyrefly.lspPath`, expanding ~ and handling relative paths.
 *
 * We tell a `$PATH` lookup apart from a relative or absolute by checking
 * for a path separator. `~\` is only a home-relative prefix on Windows;
 * on POSIX `\` is an ordinary filename character, so we leave it alone.
 */
export function resolveLspPath(
  lspPath: string,
  cwd: string | undefined,
): string {
  const isHomeRelative =
    lspPath.startsWith('~/') ||
    (process.platform === 'win32' && lspPath.startsWith('~\\'));
  if (isHomeRelative) {
    return join(homedir(), lspPath.slice(1));
  }
  if (cwd == null || parse(lspPath).dir === '') {
    return lspPath;
  }
  return resolve(cwd, lspPath);
}

// Activation blocks until the binary is resolved, so neither the Python
// extension nor the interpreter it points at may stall it indefinitely. This
// is one budget for the whole selection rather than per step: the interpreter
// lookup and the discovery run happen in sequence, so timing them separately
// would let activation block for twice as long as the number here suggests.
const SELECTION_TIMEOUT_MS = 5_000;

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  description: string,
): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new Error(`${description} did not answer within ${timeoutMs}ms`),
            ),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Append whatever the binary finder wrote to stderr. A traceback, or a warning
 * from an interpreter that still managed to answer, is the only clue to why a
 * fallback happened, and it is lost otherwise.
 */
function withStderr(reason: string, stderr: unknown): string {
  const text = typeof stderr === 'string' ? stderr.trim() : '';
  return text === '' ? reason : `${reason}; stderr: ${text}`;
}

/** How a binary was chosen, recorded so the output channel can explain it. */
interface BinarySelection {
  /** The setting value that decided which discovery path ran. */
  mode: string;
  /** The interpreter that was searched, when the environment was consulted. */
  interpreter?: string;
  /** Why the search produced nothing and the bundled binary was used. */
  fallbackReason?: string;
  path: string;
}

async function selectBinary(
  extensionUri: vscode.Uri,
  pythonEnv: PythonEnvironment,
  globalCwd: vscode.Uri | undefined,
): Promise<BinarySelection> {
  const lspPath: string = requireSettingOrDefault('pyrefly.lspPath', '');
  if (lspPath !== '') {
    return {
      mode: 'pyrefly.lspPath',
      path: resolveLspPath(lspPath, globalCwd?.fsPath),
    };
  }

  // Every case that does not produce a path falls through to the bundled binary.
  const bundledPath = vscode.Uri.joinPath(
    extensionUri,
    'bin',
    // process.platform returns win32 on any windows CPU architecture
    process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly',
  ).fsPath;

  const mode: string = requireSettingOrDefault('pyrefly.pyreflyExecutable', 'from-environment');
  if (mode !== '' && mode !== 'from-environment') {
    return {
      mode,
      path: bundledPath,
      // `bundled` is the deliberate way to ask for this binary. Any other value
      // is a typo, or a setting written by a newer version of the extension.
      fallbackReason:
        mode === 'bundled'
          ? undefined
          : 'it is not a recognized pyrefly.pyreflyExecutable value',
    };
  }

  const deadline = Date.now() + SELECTION_TIMEOUT_MS;

  let interpreter: string | undefined;
  try {
    interpreter = await withTimeout(
      pythonEnv.getInterpreterPath(globalCwd),
      deadline - Date.now(),
      'The Python extension',
    );
  } catch (error) {
    return {mode, path: bundledPath, fallbackReason: describeError(error)};
  }
  if (interpreter === undefined) {
    return {
      mode,
      path: bundledPath,
      fallbackReason: 'there is no active Python interpreter',
    };
  }

  const remainingMs = deadline - Date.now();
  if (remainingMs <= 0) {
    return {
      mode,
      interpreter,
      path: bundledPath,
      fallbackReason: `the interpreter lookup used the whole ${SELECTION_TIMEOUT_MS}ms selection budget`,
    };
  }

  const binaryFinder = vscode.Uri.joinPath(
    extensionUri,
    'resources',
    'find_pyrefly.py',
  );
  try {
    const {stdout, stderr} = await execFileAsync(
      interpreter,
      [binaryFinder.fsPath],
      {encoding: 'utf8', timeout: remainingMs},
    );
    // The script prints nothing rather than exiting non-zero when it finds no
    // Pyrefly, which is what keeps that ordinary answer distinguishable from an
    // uncaught exception in the search itself.
    const foundBinary = stdout.trim();
    if (foundBinary !== '') {
      return {mode, interpreter, path: foundBinary};
    }
    return {
      mode,
      interpreter,
      path: bundledPath,
      fallbackReason: withStderr(
        'the active Python environment has no Pyrefly installed',
        stderr,
      ),
    };
  } catch (error) {
    // Reaching here means the search broke rather than came up empty: a missing
    // or unusable interpreter, the timeout above, or a traceback out of the
    // script.
    return {
      mode,
      interpreter,
      path: bundledPath,
      fallbackReason: withStderr(
        describeError(error),
        (error as {stderr?: unknown}).stderr,
      ),
    };
  }
}

/**
 * Resolve the command and arguments the language server should be launched
 * with, from the current settings and the active Python environment.
 */
export async function resolveExecutable(
  extensionUri: vscode.Uri,
  pythonEnv: PythonEnvironment,
  logChannel: vscode.OutputChannel,
): Promise<Executable> {
  // There may be more than one URI due to multi-root workspaces, so just take the primary root.
  const globalCwd: vscode.Uri | undefined =
    vscode.workspace.workspaceFolders?.[0]?.uri;

  // `pyrefly.lspArguments` resolves to an empty array in some environments
  // (notably dev containers / remote, where the `machine-overridable` default
  // of `["lsp"]` is not applied). Spawning the binary with no subcommand makes
  // pyrefly print its help text and exit, which the client only sees as a
  // `write EPIPE` when it writes the `initialize` request. Fall back to the
  // `lsp` subcommand so the server always starts.
  const configuredArgs: string[] = requireSettingOrDefault('pyrefly.lspArguments', ['lsp']);
  const args: string[] = configuredArgs.length > 0 ? configuredArgs : ['lsp'];

  const selection = await selectBinary(extensionUri, pythonEnv, globalCwd);
  // The binary is chosen dynamically, so which one we picked and why is the
  // first thing anyone debugging a bad server needs to know.
  logChannel.appendLine(
    `Pyrefly binary selection: mode=${selection.mode}` +
      `, interpreter=${selection.interpreter ?? '<none>'}` +
      `, binary=${selection.path}` +
      `, arguments=${JSON.stringify(args)}` +
      (selection.fallbackReason === undefined
        ? ''
        : `, fell back to the bundled binary because ${selection.fallbackReason}`),
  );

  return {command: selection.path, args};
}
