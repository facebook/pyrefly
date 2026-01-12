/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {LanguageClient} from 'vscode-languageclient/node';

let statusBarItem: vscode.StatusBarItem;

/// Update the status bar based on current configuration
export async function updateStatusBar(client: LanguageClient) {
  const document = vscode.window.activeTextEditor?.document;
  if (
    document == null ||
    (document.uri.scheme !== 'file' &&
      document.uri.scheme !== 'vscode-notebook-cell' &&
      document.uri.scheme !== 'untitled') ||
    document.languageId !== 'python'
  ) {
    statusBarItem?.hide();
    return;
  }
  let status;
  try {
    status = await client.sendRequest(
      'pyrefly/textDocument/typeErrorDisplayStatus',
      client.code2ProtocolConverter.asTextDocumentItem(document),
    );
  } catch {
    statusBarItem?.hide();
    return;
  }

  if (!statusBarItem) {
    statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
    );
    statusBarItem.name = 'Pyrefly';
  }

  switch (status) {
    case 'disabled-due-to-missing-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is disabled by default.
Create a [\`pyrefly.toml\`](https://pyrefly.org/en/docs/configuration/) file or set disableTypeErrors to false in settings to show type errors.`);
      break;
    case 'disabled-in-ide-config':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is explicitly disabled.
No errors will be shown even if there is a [\`pyrefly.toml\`](https://pyrefly.org/en/docs/configuration/) file.`);
      break;
    case 'disabled-in-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip = new vscode.MarkdownString(
        `Pyrefly type checking is disabled through a config file.`,
      );
      break;
    case 'enabled-in-ide-config':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is explicitly enabled.\nType errors will always be shown.',
      );
      break;
    case 'enabled-in-config-file':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is enabled through a config file.',
      );
      break;
    default:
      statusBarItem?.hide();
      return;
  }
  statusBarItem.show();
}

export function getStatusBarItem(): vscode.StatusBarItem {
  return statusBarItem;
}
