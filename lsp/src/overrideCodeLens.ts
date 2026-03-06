/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as path from 'path';
import * as vscode from 'vscode';

type MethodSymbol = {
  range: vscode.Range;
  position: vscode.Position;
};

type NavigationTarget = {
  name: string;
  uri: vscode.Uri;
  range: vscode.Range;
};

function codeLensDisabled(document: vscode.TextDocument): boolean {
  return vscode.workspace
    .getConfiguration('python.pyrefly', document.uri)
    .get<Record<string, boolean>>('disabledLanguageServices', {})
    .codeLens === true;
}

function rangesEqual(left: vscode.Range, right: vscode.Range): boolean {
  return left.start.isEqual(right.start) && left.end.isEqual(right.end);
}

function locationsEqual(
  left: vscode.Location,
  rightUri: vscode.Uri,
  rightRange: vscode.Range,
): boolean {
  return left.uri.toString() === rightUri.toString() && rangesEqual(left.range, rightRange);
}

function normalizeLocations(
  result: vscode.Location | vscode.Location[] | vscode.LocationLink[] | undefined,
): vscode.Location[] {
  if (!result) {
    return [];
  }

  const items = Array.isArray(result) ? result : [result];
  const locations = items.map(item =>
    'targetUri' in item
      ? new vscode.Location(item.targetUri, item.targetSelectionRange ?? item.targetRange)
      : item,
  );

  const seen = new Set<string>();
  return locations.filter(location => {
    const key = `${location.uri.toString()}:${location.range.start.line}:${location.range.start.character}:${location.range.end.line}:${location.range.end.character}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function collectMethodSymbols(
  symbols: readonly vscode.DocumentSymbol[],
  enclosingClasses: readonly string[] = [],
): MethodSymbol[] {
  const methods: MethodSymbol[] = [];
  for (const symbol of symbols) {
    if (symbol.kind === vscode.SymbolKind.Class) {
      methods.push(
        ...collectMethodSymbols(symbol.children, [...enclosingClasses, symbol.name]),
      );
      continue;
    }
    if (symbol.kind !== vscode.SymbolKind.Method || enclosingClasses.length === 0) {
      continue;
    }
    methods.push({
      range: symbol.selectionRange,
      position: symbol.selectionRange.start,
    });
  }
  return methods;
}

function findSymbolNameAtRange(
  symbols: readonly vscode.DocumentSymbol[],
  targetRange: vscode.Range,
  enclosingClasses: readonly string[] = [],
): string | undefined {
  for (const symbol of symbols) {
    const nextEnclosingClasses =
      symbol.kind === vscode.SymbolKind.Class
        ? [...enclosingClasses, symbol.name]
        : enclosingClasses;
    if (rangesEqual(symbol.selectionRange, targetRange)) {
      if (symbol.kind === vscode.SymbolKind.Method && nextEnclosingClasses.length > 0) {
        return `${nextEnclosingClasses.join('.')}.${symbol.name}`;
      }
      return symbol.name;
    }
    const nested = findSymbolNameAtRange(
      symbol.children,
      targetRange,
      nextEnclosingClasses,
    );
    if (nested) {
      return nested;
    }
  }
  return undefined;
}

async function getDocumentSymbols(
  uri: vscode.Uri,
  cache: Map<string, Promise<vscode.DocumentSymbol[]>>,
): Promise<vscode.DocumentSymbol[]> {
  const key = uri.toString();
  let symbols = cache.get(key);
  if (!symbols) {
    symbols = Promise.resolve(
      vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
        'vscode.executeDocumentSymbolProvider',
        uri,
      ),
    ).then(result => result ?? []);
    cache.set(key, symbols);
  }
  return await symbols;
}

async function describeLocation(
  location: vscode.Location,
  cache: Map<string, Promise<vscode.DocumentSymbol[]>>,
): Promise<string> {
  const symbols = await getDocumentSymbols(location.uri, cache);
  const symbolName = findSymbolNameAtRange(symbols, location.range);
  if (symbolName) {
    return symbolName;
  }
  return `${path.basename(location.uri.fsPath)}:${location.range.start.line + 1}`;
}

function singleTargetCommand(target: NavigationTarget): vscode.Command {
  return {
    title: '',
    command: 'pyrefly.navigateToOverrideTarget',
    arguments: [target],
  };
}

function multiTargetCommand(targets: NavigationTarget[]): vscode.Command {
  return {
    title: '',
    command: 'pyrefly.showOverrideTargets',
    arguments: [targets],
  };
}

function overrideLensTitle(targets: readonly NavigationTarget[]): string {
  return targets.length === 1
    ? `$(arrow-up) Overrides ${targets[0].name}`
    : `$(arrow-up) Overrides ${targets.length} parents`;
}

function implementationLensTitle(targets: readonly NavigationTarget[]): string {
  return targets.length === 1
    ? `$(arrow-down) Implemented by ${targets[0].name}`
    : `$(arrow-down) ${targets.length} implementations`;
}

export class OverrideCodeLensProvider implements vscode.CodeLensProvider {
  private readonly onDidChangeEmitter = new vscode.EventEmitter<void>();

  public readonly onDidChangeCodeLenses = this.onDidChangeEmitter.event;

  public refresh(): void {
    this.onDidChangeEmitter.fire();
  }

  public async provideCodeLenses(
    document: vscode.TextDocument,
    token: vscode.CancellationToken,
  ): Promise<vscode.CodeLens[]> {
    if (document.languageId !== 'python' || codeLensDisabled(document)) {
      return [];
    }

    const rootSymbols =
      (await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
        'vscode.executeDocumentSymbolProvider',
        document.uri,
      )) ?? [];
    if (rootSymbols.length === 0) {
      return [];
    }

    const symbolCache = new Map<string, Promise<vscode.DocumentSymbol[]>>();
    symbolCache.set(document.uri.toString(), Promise.resolve(rootSymbols));

    const methodSymbols = collectMethodSymbols(rootSymbols);
    const codeLenses = await Promise.all(
      methodSymbols.map(method =>
        this.getCodeLensesForMethod(document, method, symbolCache, token),
      ),
    );
    return codeLenses.flat();
  }

  private async getCodeLensesForMethod(
    document: vscode.TextDocument,
    method: MethodSymbol,
    symbolCache: Map<string, Promise<vscode.DocumentSymbol[]>>,
    token: vscode.CancellationToken,
  ): Promise<vscode.CodeLens[]> {
    if (token.isCancellationRequested) {
      return [];
    }

    const [declarationResult, implementationResult] = await Promise.all([
      vscode.commands.executeCommand<vscode.Location | vscode.Location[] | vscode.LocationLink[]>(
        'vscode.executeDeclarationProvider',
        document.uri,
        method.position,
      ),
      vscode.commands.executeCommand<vscode.Location | vscode.Location[] | vscode.LocationLink[]>(
        'vscode.executeImplementationProvider',
        document.uri,
        method.position,
      ),
    ]);

    const declarationTargets = normalizeLocations(declarationResult).filter(
      location => !locationsEqual(location, document.uri, method.range),
    );
    const implementationTargets = normalizeLocations(implementationResult).filter(
      location => !locationsEqual(location, document.uri, method.range),
    );

    const lenses: vscode.CodeLens[] = [];
    if (declarationTargets.length > 0) {
      const targets = await Promise.all(
        declarationTargets.map(async location => ({
          name: await describeLocation(location, symbolCache),
          uri: location.uri,
          range: location.range,
        })),
      );
      const command = targets.length === 1 ? singleTargetCommand(targets[0]) : multiTargetCommand(targets);
      command.title = overrideLensTitle(targets);
      lenses.push(new vscode.CodeLens(method.range, command));
    }

    if (implementationTargets.length > 0) {
      const targets = await Promise.all(
        implementationTargets.map(async location => ({
          name: await describeLocation(location, symbolCache),
          uri: location.uri,
          range: location.range,
        })),
      );
      const command =
        targets.length === 1 ? singleTargetCommand(targets[0]) : multiTargetCommand(targets);
      command.title = implementationLensTitle(targets);
      lenses.push(new vscode.CodeLens(method.range, command));
    }

    return lenses;
  }
}

export async function navigateToOverrideTarget(
  target: NavigationTarget,
): Promise<void> {
  const document = await vscode.workspace.openTextDocument(target.uri);
  const editor = await vscode.window.showTextDocument(document);
  editor.selection = new vscode.Selection(target.range.start, target.range.start);
  editor.revealRange(target.range, vscode.TextEditorRevealType.InCenter);
}

export async function showOverrideTargets(
  targets: NavigationTarget[],
): Promise<void> {
  if (targets.length === 0) {
    return;
  }
  if (targets.length === 1) {
    await navigateToOverrideTarget(targets[0]);
    return;
  }

  const pickedTarget = await vscode.window.showQuickPick(
    targets.map(target => ({
      label: target.name,
      description: path.basename(target.uri.fsPath),
      target,
    })),
    {placeHolder: 'Select a parent or implementation to navigate to'},
  );
  if (pickedTarget) {
    await navigateToOverrideTarget(pickedTarget.target);
  }
}
