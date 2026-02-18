/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';

const WEB_WARNING_SHOWN_KEY = 'pyrefly.webWarningShown';
const ISSUE_URL = 'https://github.com/facebook/pyrefly/issues/2240';

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  const alreadyShown = context.globalState.get<boolean>(
    WEB_WARNING_SHOWN_KEY,
    false,
  );
  if (alreadyShown) {
    return;
  }

  const openIssueLabel = 'Open issue';
  const result = await vscode.window.showWarningMessage(
    'Pyrefly does not yet support VS Code Web (vscode.dev). Use the desktop app or a remote workspace for full functionality.',
    openIssueLabel,
  );
  if (result === openIssueLabel) {
    await vscode.env.openExternal(vscode.Uri.parse(ISSUE_URL));
  }

  await context.globalState.update(WEB_WARNING_SHOWN_KEY, true);
}

export function deactivate(): void {}
