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

const DISMISSED_KEY = 'pyrefly.dismissedPythonExtensionWarning';

export class PythonEnvironment {
  private api: Promise<PythonExtension | undefined>;
  private pendingListeners: (() => void)[] = [];
  private context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.api = PythonExtension.api().catch(() => {
      if (!context.globalState.get(DISMISSED_KEY)) {
        const install = 'Install';
        const dismiss = "Don't Show Again";
        vscode.window
          .showInformationMessage(
            'Install the Python extension (ms-python.python) for improved experience with Pyrefly, including automatic Python environment detection.',
            install,
            dismiss,
          )
          .then(selection => {
            if (selection === install) {
              vscode.commands.executeCommand(
                'workbench.extensions.installExtension',
                'ms-python.python',
              );
            } else if (selection === dismiss) {
              context.globalState.update(DISMISSED_KEY, true);
            }
          });
      }
      this.retryPythonAPIOnExtensionInstall();
      return undefined;
    });
  }

  /**
   * Retry accessing python API on every extension install until successful. Once successful, we call all callbacks and no longer try.
   */
  private retryPythonAPIOnExtensionInstall() {
    const disposable = vscode.extensions.onDidChange(() => {
      PythonExtension.api()
        .then(ext => {
          this.api = Promise.resolve(ext);
          disposable.dispose();
          for (const listener of this.pendingListeners) {
            listener();
            ext.environments.onDidChangeActiveEnvironmentPath(listener);
          }
        })
        .catch(() => {});
    });
    this.context.subscriptions.push(disposable);
  }

  async getInterpreterPath(uri?: vscode.Uri): Promise<string | undefined> {
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
    if (ext) {
      return ext.environments.onDidChangeActiveEnvironmentPath(callback);
    }
    this.pendingListeners.push(callback);
    return undefined;
  }
}
