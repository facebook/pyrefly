/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {PythonExtension} from '@vscode/python-extension';

export class PythonEnvironment {
  private api: Promise<PythonExtension | undefined>;

  constructor() {
    this.api = PythonExtension.api().catch(() => undefined);
  }

  async getInterpreterPath(
    uri?: vscode.Uri,
  ): Promise<string | undefined> {
    const ext = await this.api;
    if (!ext) {
      return undefined;
    }
    const envPath = await ext.environments.getActiveEnvironmentPath(uri);
    return envPath.path.length > 0 ? envPath.path : undefined;
  }

  async onDidChangeInterpreter(
    callback: () => void,
  ): Promise<vscode.Disposable | undefined> {
    const ext = await this.api;
    if (!ext) {
      return undefined;
    }
    return ext.environments.onDidChangeActiveEnvironmentPath(callback);
  }
}
