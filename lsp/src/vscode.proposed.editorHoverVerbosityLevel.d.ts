declare module 'vscode' {
  export class VerboseHover extends Hover {
    canIncreaseVerbosity?: boolean;
    canDecreaseVerbosity?: boolean;

    constructor(
      contents: MarkdownString | MarkedString | Array<MarkdownString | MarkedString>,
      range?: Range,
      canIncreaseVerbosity?: boolean,
      canDecreaseVerbosity?: boolean,
    );
  }

  export interface HoverContext {
    readonly verbosityDelta?: number;
    readonly previousHover?: Hover;
  }
}
