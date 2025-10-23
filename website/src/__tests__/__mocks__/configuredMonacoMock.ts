/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mock implementation of configured-monaco module for testing

import { SANDBOX_FILE_NAME } from '../../pages/sandbox';

/**
 * Mock editor model interface
 */
interface MockEditorModel {
    uri: { path: string };
    getValue: () => string;
    setValue: (value: string) => void;
    updateOptions: (options: any) => void;
}

/**
 * Monaco editor mock
 */
const mockModels = new Map<string, MockEditorModel>();

const monaco = {
    editor: {
        onDidCreateModel: jest.fn(),
        setModelMarkers: jest.fn(),
        getModels: (): MockEditorModel[] => Array.from(mockModels.values()),
        createModel: (content: string, language: string, uri?: any): MockEditorModel => {
            const path = uri?.path || '/mock-file.py';
            const model: MockEditorModel = {
                uri: { path },
                getValue: () => content,
                setValue: jest.fn((value: string) => { content = value; }),
                updateOptions: jest.fn(),
            };
            mockModels.set(path, model);
            return model;
        },
    },
    Uri: {
        file: (path: string) => ({ path }),
    },
    languages: {
        register: jest.fn(),
        registerCompletionItemProvider: jest.fn(),
        registerDefinitionProvider: jest.fn(),
        registerHoverProvider: jest.fn(),
        registerInlayHintsProvider: jest.fn(),
        registerDocumentSemanticTokensProvider: jest.fn(),
        registerDocumentSemanticTokensLegendProvider: jest.fn(),
    },
};

/**
 * Sets the auto-complete function for Monaco
 * @param _model The editor model
 * @param _autoCompleteFunction The auto-complete function
 */
function setAutoCompleteFunction(
    _model: MockEditorModel,
    _autoCompleteFunction: (line: number, column: number) => any
): void {}

/**
 * Sets the go-to-definition function for Monaco
 * @param _model The editor model
 * @param _getDefFunction The go-to-definition function
 */
function setGetDefFunction(
    _model: MockEditorModel,
    _getDefFunction: (line: number, column: number) => any
): void {}

/**
 * Sets the hover function for Monaco
 * @param _model The editor model
 * @param _hoverFunction The hover function
 */
function setHoverFunctionForMonaco(
    _model: MockEditorModel,
    _hoverFunction: (line: number, column: number) => any
): void {}

/**
 * Sets the inlay hint function for Monaco
 * @param _model The editor model
 * @param _inlayHintFunction The inlay hint function
 */
function setInlayHintFunctionForMonaco(
    _model: MockEditorModel,
    _inlayHintFunction: () => any
): void {}

/**
 * Sets the semantic tokens function for Monaco
 * @param _model The editor model
 * @param _semanticTokensFunction The semantic tokens function
 */
function setSemanticTokensFunctionForMonaco(
    _model: MockEditorModel,
    _semanticTokensFunction: (range: any) => any
): void {}

/**
 * Sets the semantic tokens legend function for Monaco
 * @param _semanticTokensLegendFunction The semantic tokens legend function
 */
function setSemanticTokensLegendForMonaco(
    _semanticTokensLegendFunction: () => any
): void {}

module.exports = {
    __esModule: true,
    monaco,
    setAutoCompleteFunction,
    setGetDefFunction,
    setHoverFunctionForMonaco,
    setInlayHintFunctionForMonaco,
    setSemanticTokensFunctionForMonaco,
    setSemanticTokensLegendForMonaco
};
