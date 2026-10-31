/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
import * as vscode from 'vscode';

function check(value: unknown, message: string): asserts value {
  if (!value) { throw new Error(message); }
}

/** Exercise the packaged extension in a real browser extension host. */
export async function run(): Promise<void> {
  const extension = vscode.extensions.getExtension('meta.pyrefly');
  check(extension, 'The packaged Pyrefly extension must be installed.');
  await extension.activate();
  const document = await vscode.workspace.openTextDocument({
    language: 'python', content: 's = "😀"; target: int = "bad"\nreveal_ty',
  });
  await vscode.window.showTextDocument(document);
  const deadline = Date.now() + 30000;
  while (!vscode.languages.getDiagnostics(document.uri).some(item => item.source === 'Pyrefly')) {
    check(Date.now() < deadline, 'The packaged WASM server did not publish diagnostics.');
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const hover = await vscode.commands.executeCommand<vscode.Hover[]>(
    'vscode.executeHoverProvider', document.uri, new vscode.Position(0, 11),
  );
  check(hover?.length, 'Hover must work after an astral character.');
  const list = await vscode.commands.executeCommand<vscode.CompletionList>(
    'vscode.executeCompletionItemProvider', document.uri, new vscode.Position(1, 9),
  );
  const item = list?.items.find(item => item.label === 'reveal_type');
  check(item, 'The client must convert the auto-import completion.');
  check(item.kind === vscode.CompletionItemKind.Function, 'The completion kind must be Function.');
  check(item.additionalTextEdits?.length, 'The completion must include its import edit.');
  const edits = new vscode.WorkspaceEdit();
  for (const edit of item.additionalTextEdits) { edits.replace(document.uri, edit.range, edit.newText); }
  edits.replace(document.uri, new vscode.Range(1, 0, 1, 9), 'reveal_type');
  check(await vscode.workspace.applyEdit(edits), 'The import edit must apply.');
  check(document.getText().includes('from typing import reveal_type'), 'The import must be inserted.');
  check(document.getText().endsWith('\nreveal_type'), 'The completed name must be inserted.');
}
