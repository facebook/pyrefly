declare module '../wasm/pyrefly_wasm' {
  export default function init(input?: unknown): Promise<void>;

  export class State {
    constructor(version: string);
    updateSandboxFiles(
      files: Record<string, string>,
      forceUpdate: boolean,
    ): string | null;
    updateSingleFile(filename: string, content: string): void;
    setActiveFile(filename: string): void;
    getErrors(): Array<{
      startLineNumber: number;
      startColumn: number;
      endLineNumber: number;
      endColumn: number;
      message_header: string;
      message_details: string;
      kind: string;
      severity: number;
      filename: string;
    }>;
    hover(line: number, column: number): {contents: unknown[]} | null;
    semanticTokens(range: unknown | null): unknown | null;
    semanticTokensLegend(): {
      tokenTypes: string[];
      tokenModifiers: string[];
    };
    gotoDefinition(line: number, column: number): Array<{
      startLineNumber: number;
      startColumn: number;
      endLineNumber: number;
      endColumn: number;
    }>;
    autoComplete(line: number, column: number): Array<{
      label: string;
      detail?: string;
      kind?: number;
      sortText?: string;
    }>;
    inlayHint(): Array<{
      label: string;
      position: {lineNumber: number; column: number};
    }>;
  }
}
