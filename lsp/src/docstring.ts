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

async function getDocstringRanges(
  client: LanguageClient,
  document: vscode.TextDocument,
): Promise<vscode.Range[]> {
  const identifier =
    client.code2ProtocolConverter.asTextDocumentIdentifier(document);
  const response = (await client.sendRequest(
    'pyrefly/textDocument/docstringRanges',
    identifier,
  )) as Array<{
    start: {line: number; character: number};
    end: {line: number; character: number};
  }> | null;

  if (!response) {
    return [];
  }

  return response.map(range => client.protocol2CodeConverter.asRange(range));
}

export async function runDocstringFoldingCommand(
  client: LanguageClient,
  outputChannel: vscode.OutputChannel | undefined,
  commandId: 'editor.fold' | 'editor.unfold',
): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (
    editor == null ||
    editor.document.uri.scheme !== 'file' ||
    editor.document.languageId !== 'python'
  ) {
    return;
  }

  try {
    const ranges = await getDocstringRanges(client, editor.document);
    if (ranges.length === 0) {
      return;
    }

    const seen = new Set<string>();
    const uniqueRanges = ranges.filter(range => {
      const key = `${range.start.line}:${range.start.character}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });

    const selectionLines = uniqueRanges
      .map(range => {
        if (range.start.line === range.end.line) {
          return null;
        }
        return range.start.line;
      })
      .filter((line): line is number => line != null)
      .filter((line, index, arr) => arr.indexOf(line) === index)
      .sort((a, b) => a - b);

    if (selectionLines.length === 0) {
      return;
    }

    await vscode.commands.executeCommand(commandId, {
      selectionLines,
    });
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : `Unknown error: ${String(error)}`;
    outputChannel?.appendLine(
      `Failed to ${commandId === 'editor.fold' ? 'fold' : 'unfold'} docstrings: ${message}`,
    );
  }
}
