/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {LanguageClient, LSPErrorCodes, ResponseError} from 'vscode-languageclient/node';
import {OverrideCodeLensProvider} from '../overrideCodeLens';

type NamedLocation = vscode.Location & {name?: string};
type Targets = {parents: NamedLocation[]; implementations: NamedLocation[]};
type Batch = {textDocument: {version: number}; ranges: vscode.Range[]; refreshSupport: boolean};

suite('OverrideCodeLensProvider', () => {
  let symbols: vscode.DocumentSymbol[];
  let document: vscode.TextDocument;
  let provider: OverrideCodeLensProvider;
  let registrations: vscode.Disposable[];
  let requests: Batch[];
  let symbolQueries: number;
  let notifications: Map<string, () => void>;
  let client: LanguageClient;
  let respond: (batch: Batch, token: vscode.CancellationToken) => Promise<Targets[]>;

  function symbol(name: string, kind: vscode.SymbolKind, line: number) {
    const range = new vscode.Range(line, 0, line, name.length);
    return new vscode.DocumentSymbol(
      name, '', kind,
      kind === vscode.SymbolKind.Method ? range : new vscode.Range(line, 0, 209, 80),
      range,
    );
  }

  setup(async () => {
    symbols = [];
    requests = [];
    symbolQueries = 0;
    notifications = new Map();
    respond = async batch => batch.ranges.map(() => ({parents: [], implementations: []}));
    const scheme = 'override-code-lens-test';
    registrations = [
      vscode.workspace.registerTextDocumentContentProvider(scheme, {
        provideTextDocumentContent: () => ' '.repeat(80).concat('\n').repeat(210),
      }),
      vscode.languages.registerDocumentSymbolProvider({scheme}, {
        provideDocumentSymbols: () => { symbolQueries += 1; return symbols; },
      }),
    ];
    document = await vscode.languages.setTextDocumentLanguage(
      await vscode.workspace.openTextDocument(vscode.Uri.parse(`${scheme}:/methods.py`)),
      'python',
    );
    client = {
      onNotification: (method: string, handler: () => void) => {
        notifications.set(method, handler);
        return new vscode.Disposable(() => notifications.delete(method));
      },
      code2ProtocolConverter: {
        asVersionedTextDocumentIdentifier: (doc: vscode.TextDocument) => ({
          uri: doc.uri.toString(), version: doc.version,
        }),
        asRange: (range: vscode.Range) => range,
      },
      protocol2CodeConverter: {asLocation: (location: vscode.Location) => location},
      sendRequest: async (method: string, batch: Batch, token: vscode.CancellationToken) => {
        assert.strictEqual(method, 'pyrefly/textDocument/overrideCodeLens');
        requests.push(batch);
        return respond(batch, token);
      },
    } as unknown as LanguageClient;
    provider = new OverrideCodeLensProvider(() => client);
    provider.connectClient(client);
  });

  teardown(() => {
    provider.dispose();
    registrations.forEach(registration => registration.dispose());
  });

  test('finds methods inside comment sections and class sections', async () => {
    const method = symbol('f', vscode.SymbolKind.Method, 3);
    const cls = symbol('Child', vscode.SymbolKind.Class, 1);
    const section = symbol('Models', vscode.SymbolKind.String, 0);
    const classSection = symbol('Methods', vscode.SymbolKind.String, 2);
    classSection.children = [method];
    cls.children = [classSection];
    section.children = [cls];
    respond = async () => [{
      parents: [new vscode.Location(document.uri, new vscode.Range(5, 0, 5, 1))],
      implementations: [],
    }];
    const token = new vscode.CancellationTokenSource();
    try {
      for (const root of [cls, section]) {
        symbols = [root];
        const lenses = await provider.provideCodeLenses(document, token.token);
        assert.strictEqual(lenses.length, 1);
        assert.deepStrictEqual(lenses[0].range, method.selectionRange);
      }
    } finally {
      token.dispose();
    }
  });

  test('sends one cancellable batch for 200 methods without overrides', async () => {
    const cls = symbol('C', vscode.SymbolKind.Class, 0);
    cls.children = Array.from({length: 200}, (_, i) =>
      symbol(`method_${i}`, vscode.SymbolKind.Method, i + 1),
    );
    symbols = [cls];
    const token = new vscode.CancellationTokenSource();
    try {
      assert.deepStrictEqual(await provider.provideCodeLenses(document, token.token), []);
      assert.strictEqual(requests.length, 1);
      assert.strictEqual(requests[0].ranges.length, 200);
      assert.strictEqual(requests[0].textDocument.version, document.version);
      assert.strictEqual(requests[0].refreshSupport, true);
    } finally {
      token.dispose();
    }
  });

  test('uses batch names without requesting symbols for 500 targets', async () => {
    const cls = symbol('Base', vscode.SymbolKind.Class, 0);
    cls.children = [symbol('f', vscode.SymbolKind.Method, 1)];
    symbols = [cls];
    respond = async () => [{parents: [], implementations: Array.from({length: 500}, (_, i) =>
      Object.assign(new vscode.Location(
        vscode.Uri.parse(`override-code-lens-test:/child_${i}.py`), new vscode.Range(0, 0, 0, 1),
      ), {name: `Child${i}.f`}),
    )}];
    const token = new vscode.CancellationTokenSource();
    try {
      const lenses = await provider.provideCodeLenses(document, token.token);
      assert.strictEqual(lenses.length, 1);
      assert.strictEqual(lenses[0].command?.title, '$(arrow-down) 500 implementations');
      assert.strictEqual(lenses[0].command?.arguments?.[0][499].name, 'Child499.f');
      assert.strictEqual(requests.length, 1);
      assert.strictEqual(symbolQueries, 1);
      token.cancel();
    } finally {
      token.dispose();
    }
  });

  test('retains lenses and retries server cancellation', async () => {
    const cls = symbol('Child', vscode.SymbolKind.Class, 0);
    cls.children = [symbol('f', vscode.SymbolKind.Method, 1)];
    symbols = [cls];
    const target = Object.assign(
      new vscode.Location(document.uri, new vscode.Range(5, 0, 5, 1)), {name: 'Base.f'},
    );
    const success = async () => [{parents: [target], implementations: []}];
    const token = new vscode.CancellationTokenSource();
    try {
      respond = success;
      const lenses = await provider.provideCodeLenses(document, token.token);
      for (const code of [LSPErrorCodes.RequestCancelled, LSPErrorCodes.ServerCancelled,
        LSPErrorCodes.ContentModified]) {
        const refresh = new Promise<void>(resolve => {
          const subscription = provider.onDidChangeCodeLenses(() => {
            subscription.dispose(); resolve();
          });
        });
        respond = async () => { throw new ResponseError(code, 'Analysis changed'); };
        assert.deepStrictEqual(await provider.provideCodeLenses(document, token.token), lenses);
        await refresh;
        respond = success;
        assert.deepStrictEqual(await provider.provideCodeLenses(document, token.token), lenses);
      }
    } finally {
      token.dispose();
    }
  });

  test('limits retries when the server continues to cancel', async () => {
    const cls = symbol('C', vscode.SymbolKind.Class, 0);
    cls.children = [symbol('f', vscode.SymbolKind.Method, 1)];
    symbols = [cls];
    respond = async () => { throw new ResponseError(LSPErrorCodes.ServerCancelled, 'Rechecking'); };
    let refreshCount = 0;
    const subscription = provider.onDidChangeCodeLenses(() => { refreshCount += 1; });
    const token = new vscode.CancellationTokenSource();
    try {
      for (let i = 0; i < 3; i += 1) {
        const refreshed = new Promise<void>(resolve => {
          const event = provider.onDidChangeCodeLenses(() => { event.dispose(); resolve(); });
        });
        assert.deepStrictEqual(await provider.provideCodeLenses(document, token.token), []);
        await refreshed;
      }
      assert.deepStrictEqual(await provider.provideCodeLenses(document, token.token), []);
      await new Promise(resolve => setTimeout(resolve, 200));
      assert.strictEqual(requests.length, 4);
      assert.strictEqual(refreshCount, 3);
    } finally {
      subscription.dispose();
      token.dispose();
    }
  });

  test('refreshes on server invalidation and reconnects the notification handler', async () => {
    const cls = symbol('C', vscode.SymbolKind.Class, 0);
    cls.children = [symbol('f', vscode.SymbolKind.Method, 1)];
    symbols = [cls];
    let started!: () => void;
    const entered = new Promise<void>(resolve => { started = resolve; });
    let requestToken: vscode.CancellationToken | undefined;
    respond = async (_, token) => {
      requestToken = token;
      started();
      return new Promise(resolve => token.onCancellationRequested(() => resolve([])));
    };
    provider.connectClient(client);
    assert.strictEqual(notifications.size, 1);
    const token = new vscode.CancellationTokenSource();
    const pending = provider.provideCodeLenses(document, token.token);
    await entered;
    const refreshed = new Promise<void>(resolve => {
      const subscription = provider.onDidChangeCodeLenses(() => {
        subscription.dispose(); resolve();
      });
    });
    notifications.get('pyrefly/overrideCodeLens/refresh')!();
    assert.strictEqual(requestToken?.isCancellationRequested, true);
    assert.deepStrictEqual(await pending, []);
    await refreshed;
    token.dispose();
  });

  test('propagates cancellation and cancels superseded batches', async () => {
    const cls = symbol('C', vscode.SymbolKind.Class, 0);
    cls.children = [symbol('f', vscode.SymbolKind.Method, 1)];
    symbols = [cls];
    for (const cancel of ['token', 'refresh', 'supersede'] as const) {
      let started!: () => void;
      const entered = new Promise<void>(resolve => { started = resolve; });
      let requestToken: vscode.CancellationToken | undefined;
      respond = async (_, token) => {
        requestToken = token;
        started();
        return new Promise(resolve => token.onCancellationRequested(() => resolve([])));
      };
      const token = new vscode.CancellationTokenSource();
      const pending = provider.provideCodeLenses(document, token.token);
      await entered;
      if (cancel === 'token') {
        token.cancel();
      } else if (cancel === 'refresh') {
        provider.refresh();
      } else {
        respond = async () => [];
        await provider.provideCodeLenses(document, token.token);
      }
      assert.strictEqual(requestToken?.isCancellationRequested, true);
      assert.deepStrictEqual(await pending, []);
      token.dispose();
    }
  });
});
