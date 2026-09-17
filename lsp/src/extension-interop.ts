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
export async function triggerMsPythonRefreshLanguageServersIfInstalled() {
  if (!vscode.extensions.getExtension('ms-python.python')) {
    return;
  }
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
