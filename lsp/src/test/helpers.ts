/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {PythonEnvironment} from '../python-environment';

/** Records what the binary selection logged, so tests can read the reason. */
export function recordingChannel(lines: string[]): vscode.OutputChannel {
  return {
    name: 'recording',
    append: (value: string) => void lines.push(value),
    appendLine: (value: string) => void lines.push(value),
    replace: () => {},
    clear: () => {},
    show: () => {},
    hide: () => {},
    dispose: () => {},
  } as unknown as vscode.OutputChannel;
}

/**
 * Only `getInterpreterPath` is ever called, but the class is nominally typed.
 * `scopes` collects the workspace folders it was asked about, which is how the
 * tests check that selection consults the primary root and nothing else.
 */
export function fakeEnvironment(
  interpreter: string | undefined,
  scopes?: (vscode.Uri | undefined)[],
): PythonEnvironment {
  return {
    getInterpreterPath: async (uri?: vscode.Uri) => {
      scopes?.push(uri);
      return interpreter;
    },
  } as unknown as PythonEnvironment;
}
