/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as monaco from 'monaco-editor';
import { default as MonacoEditor, loader } from '@monaco-editor/react';

type CompletionItem = monaco.languages.CompletionItem;
type Range = monaco.IRange;
type Hover = monaco.languages.Hover;
type InlayHint = monaco.languages.InlayHint;
type SemanticTokens = monaco.languages.SemanticTokens;
type SemanticTokensLegend = monaco.languages.SemanticTokensLegend;

type AutoCompleteFunction = (line: number, column: number) => CompletionItem[];
type GetDefFunction = (
    line: number,
    column: number
) => ReadonlyArray<Range> | null;
type HoverFunction = (line: number, column: number) => Hover | null;
type InlayHintFunction = () => InlayHint[];
type SemanticTokensFunction = (range: Range | null) => SemanticTokens | null;

interface Box<T> {
  contents: T;
}

const defaultAutoCompleteFunctionForMonaco: AutoCompleteFunction = (
    _line: number,
    _column: number
): CompletionItem[] => {
    throw 'not implemented';
};

const autoCompleteFunctionsForMonaco = new Map<
    monaco.editor.ITextModel,
    AutoCompleteFunction
>();

function setAutoCompleteFunction(
    model: monaco.editor.ITextModel,
    f: AutoCompleteFunction
): void {
    autoCompleteFunctionsForMonaco.set(model, f);
}

const defaultGetDefFunctionForMonaco: GetDefFunction = (
    _l: number,
    _c: number
): ReadonlyArray<Range> | null => null;
const getDefFunctionsForMonaco = new Map<
    monaco.editor.ITextModel,
    GetDefFunction
>();

function setGetDefFunction(
    model: monaco.editor.ITextModel,
    f: GetDefFunction
): void {
    getDefFunctionsForMonaco.set(model, f);
}

const defaultHoverFunctionForMonaco: HoverFunction = (
    _l: number,
    _c: number
): Hover | null => null;
const hoverFunctionsForMonaco = new Map<
    monaco.editor.ITextModel,
    HoverFunction
>();

function setHoverFunctionForMonaco(
    model: monaco.editor.ITextModel,
    f: HoverFunction
): void {
    hoverFunctionsForMonaco.set(model, f);
}

const defaultInlayHintFunctionForMonaco: InlayHintFunction =
    (): InlayHint[] => [];
const inlayHintFunctionsForMonaco = new Map<
    monaco.editor.ITextModel,
    InlayHintFunction
>();

function setInlayHintFunctionForMonaco(
    model: monaco.editor.ITextModel,
    f: InlayHintFunction
): void {
    inlayHintFunctionsForMonaco.set(model, f);
}

const defaultSemanticTokensFunctionForMonaco: SemanticTokensFunction =
    (range: Range | null): SemanticTokens | null => null;
const semanticTokensFunctionsForMonaco = new Map<
    monaco.editor.ITextModel,
    SemanticTokensFunction
>();

function setSemanticTokensFunctionForMonaco(
    model: monaco.editor.ITextModel,
    f: SemanticTokensFunction
): void {
    semanticTokensFunctionsForMonaco.set(model, f);
}

const semanticTokensLegendsForMonaco = {
    contents: () => ({ tokenTypes: [], tokenModifiers: [] })
};

function setSemanticTokensLegendForMonaco(
    f: () => SemanticTokensLegend
): void {
    semanticTokensLegendsForMonaco.contents = f;
}

monaco.languages.register({
    id: 'python',
    extensions: ['.py'],
    aliases: ['Python'],
});

monaco.languages.registerCompletionItemProvider('python', {
    triggerCharacters: [
        '.',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '[',
        '"',
        "'",
    ],

    provideCompletionItems(model, position) {
        try {
          const f =
            autoCompleteFunctionsForMonaco.get(model) ??
            defaultAutoCompleteFunctionForMonaco;
          const result = f(position.lineNumber, position.column);
          return {
            suggestions: result.map((r) => ({
              ...r,
              insertText: r.label.toString(),
            })),
          };
        } catch (e) {
          console.error(e);
          return null;
        }
      },
    });

monaco.languages.registerDefinitionProvider('python', {
    provideDefinition(model, position) {
        try {
            const f =
                getDefFunctionsForMonaco.get(model) ??
                defaultGetDefFunctionForMonaco;
            const ranges = f(position.lineNumber, position.column);
            if (!ranges || ranges.length === 0) {
                return null;
            }
            if (ranges.length === 1) {
                return { uri: model.uri, range: ranges[0] };
            }
            return ranges.map((range) => ({ uri: model.uri, range }));
        } catch (e) {
            console.error(e);
            return null;
        }
    },
});

monaco.languages.registerHoverProvider('python', {
    provideHover(model, position) {
        const f =
            hoverFunctionsForMonaco.get(model) ?? defaultHoverFunctionForMonaco;
        const result = f(position.lineNumber, position.column);
        return result;
    },
});

monaco.languages.registerInlayHintsProvider('python', {
    provideInlayHints(model) {
        const f =
            inlayHintFunctionsForMonaco.get(model) ??
            defaultInlayHintFunctionForMonaco;
        const hints = f();
        return { hints, dispose: () => {} };
    },
});

monaco.languages.registerDocumentSemanticTokensProvider('python', {
    provideDocumentSemanticTokens(model, _lastResultId, _token) {
        const f =
            semanticTokensFunctionsForMonaco.get(model) ??
            defaultSemanticTokensFunctionForMonaco;
        return f(null);
    },
    releaseDocumentSemanticTokens(_) {},
    getLegend() {
        return semanticTokensLegendsForMonaco.contents();
    }
})

monaco.languages.registerDocumentRangeSemanticTokensProvider('python', {
    provideDocumentRangeSemanticTokens(model, range, _token) {
        const f =
            semanticTokensFunctionsForMonaco.get(model) ??
            defaultSemanticTokensFunctionForMonaco;
        return f(range);
    },
    getLegend() {
        return semanticTokensLegendsForMonaco.contents();
    }
})

// Based on VS Code's Dark+ theme
monaco.editor.defineTheme("pyreflyDark", {
    base: "vs-dark",
    inherit: true,
    colors: {},
    rules: [
        { token: "comment", foreground: "#6A9955", fontStyle: "italic" },
        { token: "keyword", foreground: "#569cd6" },
        { token: "operator", foreground: "#D4D4D4" },
        { token: "namespace", foreground: "#4EC9B0" },
        { token: "type", foreground: "#4EC9B0" },
        { token: "struct", foreground: "#4EC9B0" },
        { token: "class", foreground: "#4EC9B0", fontStyle: "bold" },
        { token: "interface", foreground: "#4EC9B0", fontStyle: "bold" },
        { token: "enum", foreground: "#4FC1FF", fontStyle: "bold" },
        { token: "typeParameter", foreground: "#4EC9B0" },
        { token: "function", foreground: "#DCDCAA" },
        { token: "member", foreground: "#DCDCAA" },
        { token: "macro", foreground: "#569cd6" },
        { token: "variable", foreground: "#9CDCFE" },
        { token: "variable.readonly", foreground: "#4EC9B0" },
        { token: "parameter", foreground: "#9CDCFE" },
        { token: "property", foreground: "#9CDCFE" },
        { token: "label", foreground: "#C8C8C8" },
        { token: "type.static", fontStyle: "bold" },
        { token: "class.static", foreground: "#ff0000", fontStyle: "bold" },
    ],
});

// Based on VS Code's Light+ theme
monaco.editor.defineTheme("pyreflyLight", {
	base: "vs",
	inherit: true,
	colors: {},
	rules: [
		{ token: "comment", foreground: "#008000", fontStyle: "italic" },
		{ token: "keyword", foreground: "#0000ff" },
		{ token: "operator", foreground: "#000000" },
		{ token: "namespace", foreground: "#267f99" },
		{ token: "type", foreground: "#267f99" },
		{ token: "struct", foreground: "#267f99" },
		{ token: "class", foreground: "#267f99", fontStyle: "bold" },
		{ token: "interface", foreground: "#267f99", fontStyle: "bold" },
		{ token: "enum", foreground: "#0070C1", fontStyle: "bold" },
		{ token: "typeParameter", foreground: "#267f99" },
		{ token: "function", foreground: "#795E26" },
		{ token: "member", foreground: "#795E26" },
		{ token: "macro", foreground: "#615a60" },
		{ token: "variable", foreground: "#001080" },
		{ token: "variable.readonly", foreground: "#267f99" },
		{ token: "parameter", foreground: "#001080" },
		{ token: "property", foreground: "#001080" },
		{ token: "label", foreground: "#000000" },
		{ token: "type.static", fontStyle: "bold" },
		{ token: "class.static", foreground: "#ff0000", fontStyle: "bold" },
	],
});

// Monaco editor types are now properly handled
loader.config({ monaco });

export {
    monaco,
    setAutoCompleteFunction,
    setGetDefFunction,
    setHoverFunctionForMonaco,
    setInlayHintFunctionForMonaco,
    setSemanticTokensFunctionForMonaco,
    setSemanticTokensLegendForMonaco
};
