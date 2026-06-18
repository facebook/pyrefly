/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Web-worker "server" used by the Pyrefly VS Code Web extension (vscode.dev / github.dev).
 *
 * Architecture:
 * - `lsp/src/extension-web.ts` spins up this Worker.
 * - This worker speaks LSP over `postMessage` using the browser JSON-RPC transport from
 *   `vscode-languageserver-protocol/browser`.
 * - Typechecking is powered by the Pyrefly WASM (`pyrefly_wasm`), which exposes a small,
 *   editor-oriented API (diagnostics, hover, completion, etc).
 * - We keep an in-memory map of opened or lazily loaded file contents (`files`). On
 *   open/change/close we update the WASM state, then publish diagnostics back to the client.
 *
 * Limitations:
 * - Cross-file features are limited to opened files and imports that can be resolved by asking
 *   `lsp/src/extension-web.ts` to read specific candidate paths from the workspace.
 * - No subprocesses / interpreter probing in the browser sandbox.
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
  default: (
    input?:
      | RequestInfo
      | URL
      | Response
      | BufferSource
      | {module_or_path?: RequestInfo | URL | Response | BufferSource},
  ) => Promise<void>;
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
  semanticTokens: (range: PyreflyRange | null) => unknown | null;
  semanticTokensLegend: () => SemanticTokensLegend;
  gotoDefinition: (line: number, column: number) => PyreflyRange[];
  gotoDefinitionLocations: (
    line: number,
    column: number,
  ) => PyreflyDefinitionLocation[];
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

type PyreflyDefinitionLocation = {
  filename: string;
  range: PyreflyRange;
};

type PyreflyPosition = {
  lineNumber: number;
  column: number;
};

type PyreflyInlayHint = {
  label: string;
  position: PyreflyPosition;
};

type WorkspaceFileResponse = {
  filename: string;
  content: string;
};

const READ_WORKSPACE_FILE_REQUEST = 'pyrefly/readWorkspaceFile';
const MAX_MISSING_IMPORT_RESOLUTION_PASSES = 3;

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
let wasmResourceBytes: Uint8Array | undefined;
let pythonVersion = '3.12';

const DEFAULT_SEMANTIC_TOKENS_LEGEND: SemanticTokensLegend = {
  // Keep in sync with `pyrefly/lib/state/semantic_tokens.rs` legends.
  tokenTypes: [
    'namespace',
    'type',
    'class',
    'enum',
    'interface',
    'struct',
    'typeParameter',
    'parameter',
    'variable',
    'property',
    'enumMember',
    'event',
    'function',
    'method',
    'macro',
    'keyword',
    'modifier',
    'comment',
    'string',
    'number',
    'regexp',
    'operator',
    'decorator',
  ],
  tokenModifiers: [
    'declaration',
    'definition',
    'readonly',
    'static',
    'deprecated',
    'abstract',
    'async',
    'modification',
    'documentation',
    'defaultLibrary',
    'selfParameter',
  ],
};

async function ensureWasmState(): Promise<PyreflyState | null> {
  if (wasmState) {
    return wasmState;
  }
  try {
    if (wasmResourceBytes) {
      await wasmModule.default({module_or_path: wasmResourceBytes});
    } else {
      const wasmUrl = wasmResourceUri
        ? new URL(wasmResourceUri)
        : new URL('pyrefly_wasm_bg.wasm', self.location.href);
      await wasmModule.default({module_or_path: wasmUrl});
    }
    wasmState = new wasmModule.State(pythonVersion);
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

function normalizeSemanticTokens(value: unknown): SemanticTokens | null {
  if (value == null || typeof value !== 'object') {
    return null;
  }
  const anyValue = value as Record<string, unknown>;
  // `lsp_types::SemanticTokens` serializes as `{ data: number[], resultId?: string }`.
  if (Array.isArray(anyValue.data)) {
    return value as SemanticTokens;
  }
  // Some runtimes may surface `data` as a typed array.
  if (ArrayBuffer.isView(anyValue.data)) {
    const v = value as SemanticTokens & {data: ArrayLike<number>};
    return {
      ...v,
      data: Array.from(v.data),
    };
  }
  // Be robust to enum-wrapped shapes such as `{ Tokens: { data: [...] } }`.
  const tokens = anyValue.Tokens ?? anyValue.tokens;
  if (tokens && typeof tokens === 'object') {
    const anyTokens = tokens as Record<string, unknown>;
    if (Array.isArray(anyTokens.data)) {
      return tokens as SemanticTokens;
    }
    if (ArrayBuffer.isView(anyTokens.data)) {
      const t = tokens as SemanticTokens & {data: ArrayLike<number>};
      return {
        ...t,
        data: Array.from(t.data),
      };
    }
  }
  return null;
}

function toDiagnosticSeverity(severity: number): DiagnosticSeverity {
  // Pyrefly's WASM playground reports Monaco MarkerSeverity numeric values:
  // - 8: Error
  // - 4: Warning
  // - 2: Info
  // - 1: Hint
  // See `pyrefly/lib/playground.rs` (and Monaco MarkerSeverity docs).
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

function publishDiagnosticsForFiles(allDiagnostics: PyreflyDiagnostic[]): void {
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

function missingImportCandidates(moduleName: string, importerFilename: string): string[] {
  const path = moduleName.replace(/\./g, '/');
  const importerDir = importerFilename.split('/').slice(0, -1).join('/');
  const candidates = [
    ...(importerDir
      ? [
          `${importerDir}/${path}.pyi`,
          `${importerDir}/${path}.py`,
          `${importerDir}/${path}/__init__.pyi`,
          `${importerDir}/${path}/__init__.py`,
        ]
      : []),
    `${path}.pyi`,
    `${path}.py`,
    `${path}/__init__.pyi`,
    `${path}/__init__.py`,
  ];
  return [...new Set(candidates)];
}

async function loadMissingImportFiles(
  diagnostics: PyreflyDiagnostic[],
): Promise<boolean> {
  const modules = new Set<string>();
  for (const diagnostic of diagnostics) {
    const moduleName = diagnostic.message_header.match(
      /^Cannot find module `([^`]+)`/,
    )?.[1];
    if (moduleName) {
      modules.add(`${diagnostic.filename}\0${moduleName}`);
    }
  }

  let loadedAny = false;
  for (const key of modules) {
    const [importerFilename, moduleName] = key.split('\0', 2);
    const filenames = missingImportCandidates(moduleName, importerFilename).filter(
      filename => !files.has(filename),
    );
    if (filenames.length === 0) {
      continue;
    }
    const response = (await connection.sendRequest(
      READ_WORKSPACE_FILE_REQUEST,
      {filenames},
    )) as WorkspaceFileResponse | null;
    if (!response || files.has(response.filename)) {
      continue;
    }
    files.set(response.filename, response.content);
    loadedAny = true;
  }
  return loadedAny;
}

async function publishDiagnostics(state: PyreflyState): Promise<void> {
  for (let i = 0; i < MAX_MISSING_IMPORT_RESOLUTION_PASSES; i++) {
    const diagnostics = state.getErrors();
    publishDiagnosticsForFiles(diagnostics);
    if (!(await loadMissingImportFiles(diagnostics))) {
      return;
    }
    state.updateSandboxFiles(Object.fromEntries(files), true);
  }
  publishDiagnosticsForFiles(state.getErrors());
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
      | {wasmUri?: string; wasmBytes?: Uint8Array; pythonVersion?: string}
      | undefined;
    if (initOptions?.wasmUri) {
      wasmResourceUri = initOptions.wasmUri;
    }
    if (initOptions?.wasmBytes) {
      wasmResourceBytes = initOptions.wasmBytes;
    }
    if (initOptions?.pythonVersion) {
      pythonVersion = initOptions.pythonVersion;
    }
    const state = await ensureWasmState();
    const legend = state
      ? state.semanticTokensLegend()
      : DEFAULT_SEMANTIC_TOKENS_LEGEND;

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
    if (!files.delete(filename)) {
      return;
    }
    state.updateSandboxFiles(Object.fromEntries(files), true);
    connection.sendNotification('textDocument/publishDiagnostics', {
      uri: closeParams.textDocument.uri,
      diagnostics: [],
    });
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
  const results =
    state.gotoDefinitionLocations(
      params.position.line + 1,
      params.position.character + 1,
    ) ?? [];

  const locations = results
    .filter(result => files.has(result.filename))
    .map(result => {
      const uri = filenameToUri(result.filename);
      if (!uri) {
        return null;
      }
      return {
        uri,
        range: toLspRange(result.range),
      };
    })
    .filter((x): x is {uri: string; range: LspRange} => x != null);

  return locations.length ? locations : null;
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
    return normalizeSemanticTokens(state.semanticTokens(null));
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
    return normalizeSemanticTokens(
      state.semanticTokens(toPyreflyRange(rangeParams.range)),
    );
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
