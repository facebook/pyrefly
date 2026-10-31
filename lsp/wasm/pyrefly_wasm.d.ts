import {CompletionItem, Hover, SemanticTokens, SemanticTokensLegend} from 'vscode-languageserver-protocol';

// This is the worker's contract with pyrefly_wasm/lib.rs.
// Position and range columns count Unicode scalar values and start at one.
// autoCompleteLsp and semanticTokens already return LSP data with UTF-16 columns.
export interface ScalarPosition {
  lineNumber: number;
  column: number;
}
export interface ScalarRange {
  startLineNumber: number;
  startColumn: number;
  endLineNumber: number;
  endColumn: number;
}
export interface Diagnostic extends ScalarRange {
  message_header: string;
  message_details: string;
  kind: string;
  severity: number;
  filename: string;
}
export default function init(input: {module_or_path: Uint8Array}): Promise<unknown>;
export class State {
  constructor(version: string);
  updateSandboxFiles(files: Record<string, string>, forceUpdate: boolean): string | undefined;
  updateSingleFile(filename: string, content: string): void;
  setActiveFile(filename: string): void;
  pendingImports(): string[];
  getErrors(): Diagnostic[];
  hover(line: number, column: number): {contents: Hover['contents'][]} | null;
  semanticTokens(range: ScalarRange | null): SemanticTokens | null;
  semanticTokensLegend(): SemanticTokensLegend;
  gotoDefinitionLocations(line: number, column: number): {filename: string; range: ScalarRange}[];
  autoCompleteLsp(line: number, column: number): CompletionItem[];
  inlayHint(callArgumentNames: boolean): {label: string; position: ScalarPosition}[];
}
