/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {
  BrowserMessageReader,
  BrowserMessageWriter,
  CompletionItem,
  createProtocolConnection,
  DefinitionParams,
  DidChangeTextDocumentParams,
  DidCloseTextDocumentParams,
  DidOpenTextDocumentParams,
  Diagnostic,
  DiagnosticSeverity,
  CompletionParams,
  Hover,
  HoverParams,
  InitializeParams,
  InitializeResult,
  InlayHintParams,
  InlayHint,
  MessageType,
  Position as LspPosition,
  Range as LspRange,
  SemanticTokens,
  SemanticTokensParams,
  SemanticTokensRangeParams,
  SemanticTokensLegend,
  TextDocumentSyncKind,
} from 'vscode-languageserver-protocol/browser';
import * as pyreflyWasm from '../wasm/pyrefly_wasm';

type PyreflyWasmModule = {
  default: (input?: RequestInfo | URL | Response | BufferSource) => Promise<void>;
  State: new (version: string) => PyreflyState;
};

type PyreflyState = {
  updateSandboxFiles: (
    files: Record<string, string>,
    forceUpdate: boolean,
  ) => string | null;
  updateSingleFile: (filename: string, content: string) => void;
  setActiveFile: (filename: string) => void;
  getErrors: () => PyreflyDiagnostic[];
  hover: (line: number, column: number) => PyreflyHover | null;
  semanticTokens: (range: PyreflyRange | null) => SemanticTokens | null;
  semanticTokensLegend: () => SemanticTokensLegend;
  gotoDefinition: (line: number, column: number) => PyreflyRange[];
  autoComplete: (line: number, column: number) => CompletionItem[];
  inlayHint: () => PyreflyInlayHint[];
};

type PyreflyDiagnostic = {
  startLineNumber: number;
  startColumn: number;
  endLineNumber: number;
  endColumn: number;
  message_header: string;
  message_details: string;
  kind: string;
  severity: number;
  filename: string;
};

type PyreflyHover = {
  contents: unknown[];
};

type PyreflyRange = {
  startLineNumber: number;
  startColumn: number;
  endLineNumber: number;
  endColumn: number;
};

type PyreflyPosition = {
  lineNumber: number;
  column: number;
};

type PyreflyInlayHint = {
  label: string;
  position: PyreflyPosition;
};

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const connection = createProtocolConnection(
  new BrowserMessageReader(workerScope),
  new BrowserMessageWriter(workerScope),
);

const files = new Map<string, string>();
let workspaceRoots: Array<{uri: string; path: string}> = [];
const wasmModule = pyreflyWasm as unknown as PyreflyWasmModule;
let wasmState: PyreflyState | null = null;
let wasmLoadNotified = false;
let wasmResourceUri: string | undefined;

async function ensureWasmState(): Promise<PyreflyState | null> {
  if (wasmState) {
    return wasmState;
  }
  try {
    const wasmUrl = wasmResourceUri
      ? new URL(wasmResourceUri)
      : new URL('pyrefly_wasm_bg.wasm', self.location.href);
    await wasmModule.default(wasmUrl);
    wasmState = new wasmModule.State('3.12');
    return wasmState;
  } catch (error) {
    if (!wasmLoadNotified) {
      wasmLoadNotified = true;
      connection.sendNotification('window/showMessage', {
        type: MessageType.Error,
        message: `Pyrefly web failed to load the WASM bundle: ${String(
          error,
        )}. Run pyrefly_wasm/build.sh and rebuild the extension.`,
      });
    }
    console.error(String(error));
    return null;
  }
}

function setWorkspaceRoots(params: InitializeParams): void {
  workspaceRoots = [];
  if (params.workspaceFolders) {
    for (const folder of params.workspaceFolders) {
      workspaceRoots.push({
        uri: folder.uri,
        path: getPathFromUri(folder.uri),
      });
    }
  } else if (params.rootUri) {
    workspaceRoots.push({
      uri: params.rootUri,
      path: getPathFromUri(params.rootUri),
    });
  }
}

function getPathFromUri(uri: string): string {
  try {
    return new URL(uri).pathname;
  } catch {
    return uri;
  }
}

function uriToFilename(uri: string): string {
  const parsedPath = getPathFromUri(uri);
  for (const root of workspaceRoots) {
    if (parsedPath.startsWith(root.path)) {
      let rel = parsedPath.slice(root.path.length);
      if (rel.startsWith('/')) {
        rel = rel.slice(1);
      }
      return rel;
    }
  }
  return parsedPath.startsWith('/') ? parsedPath.slice(1) : parsedPath;
}

function filenameToUri(filename: string): string | null {
  const root = workspaceRoots[0];
  if (!root) {
    return null;
  }
  const trimmed = filename.startsWith('/') ? filename.slice(1) : filename;
  const url = new URL(root.uri);
  const base = root.path.endsWith('/') ? root.path : `${root.path}/`;
  url.pathname = `${base}${trimmed}`;
  return url.toString();
}

function toLspPosition(line: number, column: number): LspPosition {
  return {
    line: Math.max(0, line - 1),
    character: Math.max(0, column - 1),
  };
}

function toLspRange(range: PyreflyRange): LspRange {
  return {
    start: toLspPosition(range.startLineNumber, range.startColumn),
    end: toLspPosition(range.endLineNumber, range.endColumn),
  };
}

function toPyreflyRange(range: LspRange): PyreflyRange {
  return {
    startLineNumber: range.start.line + 1,
    startColumn: range.start.character + 1,
    endLineNumber: range.end.line + 1,
    endColumn: range.end.character + 1,
  };
}

function toDiagnosticSeverity(severity: number): DiagnosticSeverity {
  switch (severity) {
    case 8:
      return DiagnosticSeverity.Error;
    case 4:
      return DiagnosticSeverity.Warning;
    case 2:
      return DiagnosticSeverity.Information;
    case 1:
      return DiagnosticSeverity.Hint;
    default:
      return DiagnosticSeverity.Error;
  }
}

async function publishDiagnostics(state: PyreflyState): Promise<void> {
  const allDiagnostics = state.getErrors();
  const byFile = new Map<string, Diagnostic[]>();

  for (const diag of allDiagnostics) {
    const entry = byFile.get(diag.filename) ?? [];
    const details = diag.message_details
      ? `\n${diag.message_details}`
      : '';
    entry.push({
      range: toLspRange(diag),
      message: `${diag.message_header}${details}`,
      severity: toDiagnosticSeverity(diag.severity),
      source: 'Pyrefly',
      code: diag.kind,
    });
    byFile.set(diag.filename, entry);
  }

  for (const filename of files.keys()) {
    const uri = filenameToUri(filename);
    if (!uri) {
      continue;
    }
    connection.sendNotification('textDocument/publishDiagnostics', {
      uri,
      diagnostics: byFile.get(filename) ?? [],
    });
  }
}

async function rebuildAll(): Promise<void> {
  const state = await ensureWasmState();
  if (!state) {
    return;
  }
  state.updateSandboxFiles(Object.fromEntries(files), true);
  await publishDiagnostics(state);
}

connection.onRequest(
  'initialize',
  async (params: InitializeParams): Promise<InitializeResult> => {
    setWorkspaceRoots(params);
    const initOptions = params.initializationOptions as
      | {wasmUri?: string}
      | undefined;
    if (initOptions?.wasmUri) {
      wasmResourceUri = initOptions.wasmUri;
    }
    const state = await ensureWasmState();
    const legend = state
      ? state.semanticTokensLegend()
      : {tokenTypes: [], tokenModifiers: []};

    return {
      capabilities: {
        textDocumentSync: TextDocumentSyncKind.Full,
        completionProvider: {resolveProvider: false},
        hoverProvider: true,
        definitionProvider: true,
        inlayHintProvider: true,
        semanticTokensProvider: {
          legend,
          full: true,
          range: true,
        },
      },
    };
  },
);

connection.onRequest('shutdown', () => {
  return null;
});

connection.onRequest(
  'pyrefly/setWorkspaceFiles',
  async (params: {files: Record<string, string>}) => {
    files.clear();
    for (const [filename, content] of Object.entries(params.files)) {
      files.set(filename, content);
    }
    await rebuildAll();
    return null;
  },
);

connection.onNotification(
  'textDocument/didOpen',
  async (openParams: DidOpenTextDocumentParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return;
  }
  const filename = uriToFilename(openParams.textDocument.uri);
  const isNewFile = !files.has(filename);
  files.set(filename, openParams.textDocument.text);
  if (isNewFile) {
    await rebuildAll();
  } else {
    state.updateSingleFile(filename, openParams.textDocument.text);
    await publishDiagnostics(state);
  }
  },
);

connection.onNotification(
  'textDocument/didChange',
  async (changeParams: DidChangeTextDocumentParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return;
  }
  const filename = uriToFilename(changeParams.textDocument.uri);
  const content = changeParams.contentChanges[0]?.text;
  if (content == null) {
    return;
  }
  const isNewFile = !files.has(filename);
  files.set(filename, content);
  if (isNewFile) {
    await rebuildAll();
  } else {
    state.updateSingleFile(filename, content);
    await publishDiagnostics(state);
  }
  },
);

connection.onNotification(
  'textDocument/didClose',
  async (closeParams: DidCloseTextDocumentParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return;
  }
  const filename = uriToFilename(closeParams.textDocument.uri);
  const content = files.get(filename);
  if (!content) {
    return;
  }
  state.updateSingleFile(filename, content);
  await publishDiagnostics(state);
  },
);

connection.onRequest(
  'textDocument/hover',
  async (hoverParams: HoverParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return null;
  }
  const filename = uriToFilename(hoverParams.textDocument.uri);
  state.setActiveFile(filename);
  const hover = state.hover(
    hoverParams.position.line + 1,
    hoverParams.position.character + 1,
  );
  if (!hover || hover.contents.length === 0) {
    return null;
  }
  const contents = hover.contents[0] as Hover['contents'];
  return {contents};
  },
);

connection.onRequest(
  'textDocument/completion',
  async (completionParams: CompletionParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return null;
  }
  const filename = uriToFilename(completionParams.textDocument.uri);
  state.setActiveFile(filename);
  return state.autoComplete(
    completionParams.position.line + 1,
    completionParams.position.character + 1,
  );
  },
);

connection.onRequest('textDocument/definition', async (params: DefinitionParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return null;
  }
  const filename = uriToFilename(params.textDocument.uri);
  state.setActiveFile(filename);
  const ranges = state.gotoDefinition(
    params.position.line + 1,
    params.position.character + 1,
  );
  if (!ranges.length) {
    return null;
  }
  return ranges.map(range => ({
    uri: params.textDocument.uri,
    range: toLspRange(range),
  }));
});

connection.onRequest(
  'textDocument/semanticTokens/full',
  async (fullParams: SemanticTokensParams) => {
    const state = await ensureWasmState();
    if (!state) {
      return null;
    }
    const filename = uriToFilename(fullParams.textDocument.uri);
    state.setActiveFile(filename);
    return state.semanticTokens(null);
  },
);

connection.onRequest(
  'textDocument/semanticTokens/range',
  async (rangeParams: SemanticTokensRangeParams) => {
    const state = await ensureWasmState();
    if (!state) {
      return null;
    }
    const filename = uriToFilename(rangeParams.textDocument.uri);
    state.setActiveFile(filename);
    return state.semanticTokens(toPyreflyRange(rangeParams.range));
  },
);

connection.onRequest(
  'textDocument/inlayHint',
  async (inlayParams: InlayHintParams) => {
  const state = await ensureWasmState();
  if (!state) {
    return null;
  }
  const filename = uriToFilename(inlayParams.textDocument.uri);
  state.setActiveFile(filename);
  const hints = state.inlayHint();
  return hints.map(hint => ({
    label: hint.label,
    position: toLspPosition(hint.position.lineNumber, hint.position.column),
  })) as InlayHint[];
  },
);

connection.listen();
