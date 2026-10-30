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
import {LanguageClient, Location, LSPErrorCodes, ResponseError} from 'vscode-languageclient/node';

const MAX_METHOD_SYMBOLS = 200;
const REFRESH_DELAY_MS = 150;
const MAX_SERVER_RETRIES = 3;
const REFRESH_METHOD = 'pyrefly/overrideCodeLens/refresh';

type NamedLocation = Location & {name?: string};

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

function hasHierarchicalSymbols(
  symbols: vscode.DocumentSymbol[] | vscode.SymbolInformation[] | undefined,
): symbols is vscode.DocumentSymbol[] {
  return symbols !== undefined && (symbols.length === 0 || 'children' in symbols[0]);
}

function collectMethodRanges(
  symbols: readonly vscode.DocumentSymbol[],
  inClass = false,
): vscode.Range[] {
  const methods: vscode.Range[] = [];
  for (const symbol of symbols) {
    if (symbol.kind === vscode.SymbolKind.Method && inClass) {
      methods.push(symbol.selectionRange);
    } else {
      methods.push(
        ...collectMethodRanges(
          symbol.children,
          inClass || symbol.kind === vscode.SymbolKind.Class,
        ),
      );
    }
  }
  return methods;
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
  private readonly requests = new Map<string, vscode.CancellationTokenSource>();
  private cachedLenses = new WeakMap<vscode.TextDocument, {
    version: number;
    lenses: vscode.CodeLens[];
  }>();
  private readonly retries = new Map<string, number>();
  private refreshTimer: ReturnType<typeof setTimeout> | undefined;
  private refreshSubscription: vscode.Disposable | undefined;

  private readonly onDidChangeEmitter = new vscode.EventEmitter<void>();

  public readonly onDidChangeCodeLenses = this.onDidChangeEmitter.event;

  public constructor(private readonly getClient: () => LanguageClient) {}

  public connectClient(client: LanguageClient): void {
    this.refreshSubscription?.dispose();
    this.refreshSubscription = client.onNotification(REFRESH_METHOD, () => this.refresh());
    this.cachedLenses = new WeakMap();
    this.refresh();
  }

  public refresh(): void {
    for (const request of this.requests.values()) {
      request.cancel();
    }
    this.retries.clear();
    this.scheduleRefresh();
  }

  private scheduleRefresh(): void {
    if (this.refreshTimer !== undefined) {
      clearTimeout(this.refreshTimer);
    }
    this.refreshTimer = setTimeout(() => {
      this.refreshTimer = undefined;
      this.onDidChangeEmitter.fire();
    }, REFRESH_DELAY_MS);
  }

  public dispose(): void {
    for (const request of this.requests.values()) {
      request.cancel();
      request.dispose();
    }
    this.requests.clear();
    this.cachedLenses = new WeakMap();
    this.retries.clear();
    this.refreshSubscription?.dispose();
    if (this.refreshTimer !== undefined) {
      clearTimeout(this.refreshTimer);
    }
    this.onDidChangeEmitter.dispose();
  }

  public async provideCodeLenses(
    document: vscode.TextDocument,
    token: vscode.CancellationToken,
  ): Promise<vscode.CodeLens[]> {
    if (document.languageId !== 'python' || codeLensDisabled(document)) {
      return [];
    }

    const key = document.uri.toString();
    this.requests.get(key)?.cancel();
    const cancellation = new vscode.CancellationTokenSource();
    this.requests.set(key, cancellation);
    const subscription = token.onCancellationRequested(() => cancellation.cancel());
    if (token.isCancellationRequested) {
      cancellation.cancel();
    }
    token = cancellation.token;
    const version = document.version;
    try {
      const rootSymbolsResult =
        (await vscode.commands.executeCommand<
          vscode.DocumentSymbol[] | vscode.SymbolInformation[]
        >(
          'vscode.executeDocumentSymbolProvider',
          document.uri,
        )) ?? [];
      if (!hasHierarchicalSymbols(rootSymbolsResult) || rootSymbolsResult.length === 0) {
        return [];
      }
      const rootSymbols = rootSymbolsResult;

      const methodRanges = collectMethodRanges(rootSymbols);
      if (methodRanges.length > MAX_METHOD_SYMBOLS) {
        return [];
      }

      if (
        token.isCancellationRequested ||
        document.version !== version ||
        methodRanges.length === 0
      ) {
        return [];
      }
      const client = this.getClient();
      const result = await client.sendRequest<
        {parents: NamedLocation[]; implementations: NamedLocation[]}[]
      >(
        'pyrefly/textDocument/overrideCodeLens',
        {
          textDocument:
            client.code2ProtocolConverter.asVersionedTextDocumentIdentifier(document),
          refreshSupport: true,
          ranges: methodRanges.map(range =>
            client.code2ProtocolConverter.asRange(range),
          ),
        },
        token,
      );
      if (token.isCancellationRequested || document.version !== version) {
        return [];
      }
      const toTarget = (location: NamedLocation): NavigationTarget => {
        const target = client.protocol2CodeConverter.asLocation(location);
        return {
          name: location.name ??
            `${path.basename(target.uri.fsPath)}:${target.range.start.line + 1}`,
          uri: target.uri,
          range: target.range,
        };
      };
      const lenses = result.flatMap((targets, i) => {
        const methodLenses: vscode.CodeLens[] = [];
        for (const [locations, title] of [
          [targets.parents, overrideLensTitle],
          [targets.implementations, implementationLensTitle],
        ] as const) {
          if (locations.length === 0) {
            continue;
          }
          const navigationTargets = locations.map(toTarget);
          const command =
            navigationTargets.length === 1
              ? singleTargetCommand(navigationTargets[0])
              : multiTargetCommand(navigationTargets);
          command.title = title(navigationTargets);
          methodLenses.push(new vscode.CodeLens(methodRanges[i], command));
        }
        return methodLenses;
      });
      this.cachedLenses.set(document, {version, lenses});
      this.retries.delete(key);
      return lenses;
    } catch (error) {
      if (token.isCancellationRequested || document.version !== version) {
        return [];
      }
      if (
        error instanceof ResponseError &&
        [
          LSPErrorCodes.RequestCancelled,
          LSPErrorCodes.ServerCancelled,
          LSPErrorCodes.ContentModified,
        ].includes(error.code)
      ) {
        const attempts = this.retries.get(key) ?? 0;
        if (attempts < MAX_SERVER_RETRIES) {
          this.retries.set(key, attempts + 1);
          this.scheduleRefresh();
        }
        const cached = this.cachedLenses.get(document);
        return cached?.version === version ? cached.lenses : [];
      }
      throw error;
    } finally {
      subscription.dispose();
      cancellation.dispose();
      if (this.requests.get(key) === cancellation) {
        this.requests.delete(key);
      }
    }
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
