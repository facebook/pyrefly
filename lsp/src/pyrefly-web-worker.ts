/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Browser LSP transport for the WASM type checker.
 * Each workspace has its own state, original document URIs, and import cache.
 * Rust records unresolved modules before diagnostic suppression. The host reads
 * candidate files through workspace.fs. Playground scalar positions are converted
 * here. LSP completion edits and semantic tokens already use UTF-16.
 * Untitled and external files use isolated sessions. Notebook cells are excluded.
 */
import {
  BrowserMessageReader, BrowserMessageWriter, createProtocolConnection,
  Diagnostic, DiagnosticSeverity, InitializeParams, InitializeResult,
  TextDocumentSyncKind, FileChangeType,
  DidOpenTextDocumentParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
  DidChangeWatchedFilesParams, TextDocumentPositionParams, SemanticTokensParams,
  SemanticTokensRangeParams, InlayHintParams,
} from 'vscode-languageserver-protocol/browser';
import init, {State} from '../wasm/pyrefly_wasm';
import {
  ProjectOptions, WorkspaceFile, WebSession, READ_FILE,
  SUPPORTED_SCHEMES, CONFIG_FILES, relativeFilename, fileUri, toScalar, toLsp, toLspRange,
} from './web-session';

const scope = self as DedicatedWorkerGlobalScope;
const connection = createProtocolConnection(
  new BrowserMessageReader(scope), new BrowserMessageWriter(scope),
);
const projects: WebSession[] = [];
const documents = new Map<string, {session: WebSession; filename: string}>();
const pending = new Set<WebSession>();
let defaultVersion = '3.12';
let work: Promise<void> = Promise.resolve();
let timer: ReturnType<typeof setTimeout> | undefined;

function newSession(options: ProjectOptions): WebSession {
  let state: State;
  try {
    state = new State(options.pythonVersion);
  } catch (error) {
    throw new Error('Invalid Pyrefly Python version for ' + options.uri + ': ' + String(error));
  }
  return new WebSession(options, state, params =>
    connection.sendRequest<WorkspaceFile | null>(READ_FILE, params),
  );
}

function publish(session: WebSession): void {
  const diagnostics = new Map<string, Diagnostic[]>();
  for (const item of session.state.getErrors()) {
    const text = session.files.get(item.filename);
    if (text === undefined) {
      continue;
    }
    // WASM uses Monaco MarkerSeverity values.
    const severity = new Map([
      [8, DiagnosticSeverity.Error], [4, DiagnosticSeverity.Warning],
      [2, DiagnosticSeverity.Information], [1, DiagnosticSeverity.Hint],
    ]).get(item.severity);
    if (severity === undefined) {
      throw new Error('Unknown WASM diagnostic severity: ' + item.severity);
    }
    const list = diagnostics.get(item.filename) ?? [];
    list.push({
      range: toLspRange(item, text),
      message: item.message_header + (item.message_details ? '\n' + item.message_details : ''),
      code: item.kind, source: 'Pyrefly', severity,
    });
    diagnostics.set(item.filename, list);
  }
  const published = new Set([...session.opened, ...CONFIG_FILES]);
  for (const filename of published) {
    const uri = session.uris.get(filename) ||
      (session.options.uri ? fileUri(session.options.uri, filename) : undefined);
    if (!uri) { continue; }
    connection.sendNotification('textDocument/publishDiagnostics', {
      uri, diagnostics: diagnostics.get(filename) ?? [],
    });
  }
}

/** Serialize WASM work and combine document changes received within 40 ms. */
function flush(): Promise<void> {
  if (timer !== undefined) {
    clearTimeout(timer);
    timer = undefined;
  }
  // A failed request already reports its error to the client. Later edits can retry.
  work = work.catch(() => undefined).then(async () => {
    while (pending.size > 0) {
      const batch = Array.from(pending);
      pending.clear();
      for (const session of batch) {
        await session.rebuild();
        publish(session);
      }
    }
  });
  return work;
}

function schedule(session: WebSession): void {
  pending.add(session);
  if (timer === undefined) {
    timer = setTimeout(() => {
      void flush().catch(error => {
        connection.sendNotification('window/showMessage', {
          type: 1, message: 'Pyrefly web: ' + String(error),
        });
        // Report the failure while permitting the next document change to retry.
        work = Promise.resolve();
      });
    }, 40);
  }
}

connection.onRequest('initialize', async (params: InitializeParams): Promise<InitializeResult> => {
  const options = params.initializationOptions as {
    wasmBytes: Uint8Array; projects: ProjectOptions[]; pythonVersion: string;
  };
  try {
    await init({module_or_path: options.wasmBytes});
  } catch (error) {
    throw new Error('Cannot initialize Pyrefly WASM: ' + String(error) +
      '. Run pyrefly_wasm/build.sh and rebuild the extension.');
  }
  defaultVersion = options.pythonVersion;
  for (const project of options.projects) {
    projects.push(newSession(project));
  }
  const probe = new State('3.12');
  return {capabilities: {
    positionEncoding: 'utf-16',
    textDocumentSync: TextDocumentSyncKind.Full,
    completionProvider: {resolveProvider: false},
    hoverProvider: true, definitionProvider: true, inlayHintProvider: true,
    semanticTokensProvider: {
      legend: probe.semanticTokensLegend(), full: true, range: true,
    },
  }};
});

connection.onNotification('textDocument/didOpen', (params: DidOpenTextDocumentParams) => {
  const {uri, text} = params.textDocument;
  const scheme = new URL(uri).protocol.slice(0, -1);
  if (scheme === 'vscode-notebook-cell' || (!SUPPORTED_SCHEMES.includes(scheme) &&
      !projects.some(project => new URL(project.options.uri).protocol === scheme + ':'))) {
    return;
  }
  const match = projects
    .map(session => ({session, filename: relativeFilename(session.options.uri, uri)}))
    .filter((item): item is {session: WebSession; filename: string} => item.filename !== undefined)
    .sort((a, b) => b.session.options.uri.length - a.session.options.uri.length)[0];
  const session = match?.session ?? newSession({
    uri: '', pythonVersion: defaultVersion, files: {},
  });
  const filename = match?.filename ?? 'document.py';
  documents.set(uri, {session, filename});
  session.opened.add(filename);
  session.files.set(filename, text);
  session.uris.set(filename, uri);
  schedule(session);
});

connection.onNotification('textDocument/didChange', (params: DidChangeTextDocumentParams) => {
  const document = documents.get(params.textDocument.uri);
  if (!document) {
    return;
  }
  const content = params.contentChanges[params.contentChanges.length - 1];
  if (content) {
    document.session.files.set(document.filename, content.text);
    schedule(document.session);
  }
});

connection.onNotification('textDocument/didClose', (params: DidCloseTextDocumentParams) => {
  const document = documents.get(params.textDocument.uri);
  if (!document) {
    return;
  }
  documents.delete(params.textDocument.uri);
  document.session.opened.delete(document.filename);
  document.session.invalidate(document.filename);
  connection.sendNotification('textDocument/publishDiagnostics', {
    uri: params.textDocument.uri, diagnostics: [],
  });
  schedule(document.session);
});

connection.onNotification('workspace/didChangeWatchedFiles', (params: DidChangeWatchedFilesParams) => {
  for (const change of params.changes) {
    for (const session of projects) {
      const filename = relativeFilename(session.options.uri, change.uri);
      if (filename === undefined) {
        continue;
      }
      session.invalidate(filename);
      if (CONFIG_FILES.includes(filename) && change.type !== FileChangeType.Deleted) {
        // Configuration reads use the same serialized path as module reads.
        work = work.then(async () => {
          const file = await connection.sendRequest<WorkspaceFile | null>(READ_FILE, {
            rootUri: session.options.uri, filenames: [filename],
          });
          if (file && !session.opened.has(filename)) {
            session.files.set(filename, file.content);
            session.uris.set(filename, file.uri);
          }
        });
      }
      schedule(session);
    }
  }
});

/** Unknown documents return no result and cannot reuse another active file. */
async function select(uri: string) {
  await flush();
  const document = documents.get(uri);
  if (!document) {
    return undefined;
  }
  const text = document.session.files.get(document.filename);
  if (text === undefined) {
    throw new Error('An open document must have source text.');
  }
  document.session.state.setActiveFile(document.filename);
  return {...document, text, state: document.session.state};
}

connection.onRequest('textDocument/hover', async (params: TextDocumentPositionParams) => {
  const doc = await select(params.textDocument.uri);
  if (!doc) { return null; }
  const pos = toScalar(params.position, doc.text);
  const hover = doc.state.hover(pos.lineNumber, pos.column);
  return hover?.contents[0] ? {contents: hover.contents[0]} : null;
});

connection.onRequest('textDocument/completion', async (params: TextDocumentPositionParams) => {
  const doc = await select(params.textDocument.uri);
  if (!doc) { return []; }
  const pos = toScalar(params.position, doc.text);
  return doc.state.autoCompleteLsp(pos.lineNumber, pos.column);
});

connection.onRequest('textDocument/definition', async (params: TextDocumentPositionParams) => {
  const doc = await select(params.textDocument.uri);
  if (!doc) { return []; }
  const pos = toScalar(params.position, doc.text);
  return doc.state.gotoDefinitionLocations(pos.lineNumber, pos.column).flatMap(result => {
    const uri = doc.session.uris.get(result.filename);
    const text = doc.session.files.get(result.filename);
    return uri && text !== undefined ? [{uri, range: toLspRange(result.range, text)}] : [];
  });
});

connection.onRequest('textDocument/inlayHint', async (params: InlayHintParams) => {
  const doc = await select(params.textDocument.uri);
  if (!doc) { return []; }
  return doc.state.inlayHint(false).map(hint => ({
    label: hint.label, position: toLsp(hint.position, doc.text),
  }));
});

connection.onRequest('textDocument/semanticTokens/full', async (params: SemanticTokensParams) => {
  const doc = await select(params.textDocument.uri);
  return doc?.state.semanticTokens(null) ?? null;
});

connection.onRequest('textDocument/semanticTokens/range', async (params: SemanticTokensRangeParams) => {
  const doc = await select(params.textDocument.uri);
  if (!doc) { return null; }
  const start = toScalar(params.range.start, doc.text);
  const end = toScalar(params.range.end, doc.text);
  return doc.state.semanticTokens({
    startLineNumber: start.lineNumber, startColumn: start.column,
    endLineNumber: end.lineNumber, endColumn: end.column,
  });
});

connection.onRequest('shutdown', () => null);
connection.listen();
