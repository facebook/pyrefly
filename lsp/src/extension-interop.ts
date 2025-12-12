/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';

/**
 * This function will trigger the ms-python extension to reasses which language server to spin up.
 * It does this by changing languageServer setting: this triggers a refresh of active language
 * servers:
 * https://github.com/microsoft/vscode-python/blob/main/src/client/languageServer/watcher.ts#L296
 *
 * We then change the setting back so we don't end up messing up the users settings.
 */
export async function triggerMsPythonRefreshLanguageServers() {
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

/**
 * This function checks if the Windsurf Pyright extension is installed, and if so,
 * disables its language services to avoid conflicts with Pyrefly.
 */
export async function disableWindsurfPyrightIfInstalled() {
  const windsurfPyrightExtension = vscode.extensions.getExtension(
    'codeium.windsurfpyright',
  );
  if (windsurfPyrightExtension) {
    const config = vscode.workspace.getConfiguration('windsurfPyright');
    const setting = 'disableLanguageServices';
    await config.update(setting, true, vscode.ConfigurationTarget.Global);
  }
}

/**
 * This function checks if the Based Pyright extension is installed, and if so,
 * disables its language services to avoid conflicts with Pyrefly.
 */
export async function disableBasedPyrightIfInstalled() {
  const basedPyrightExtension = vscode.extensions.getExtension(
    'detachhead.basedpyright',
  );
  if (basedPyrightExtension) {
    const config = vscode.workspace.getConfiguration('basedpyright');
    const setting = 'disableLanguageServices';
    await config.update(setting, true, vscode.ConfigurationTarget.Global);
  }
}

/**
 * This function checks if the Cursor Pyright extension is installed, and if so,
 * disables its language services to avoid conflicts with Pyrefly.
 */
export async function disableCursorPyrightIfInstalled() {
  const cursorPyrightExtension = vscode.extensions.getExtension(
    'anysphere.cursorpyright',
  );
  if (cursorPyrightExtension) {
    const config = vscode.workspace.getConfiguration('cursorpyright');
    const setting = 'disableLanguageServices';
    await config.update(setting, true, vscode.ConfigurationTarget.Global);
  }
}
