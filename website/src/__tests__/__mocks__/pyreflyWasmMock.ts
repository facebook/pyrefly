/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mock implementation of pyrefly_wasm module for testing
// Provides immediate resolution to avoid async issues

/**
 * Mock State class for Pyrefly WASM module
 */
class MockState {
    /**
     * Updates the source code to be analyzed
     * @param source The source code
     */
    updateSource(source?: string): void {}

    /**
     * Updates multiple sandbox files
     * @param files Dictionary of filename to content
     */
    updateSandboxFiles(files: Record<string, string>): void {}

    /**
     * Updates a single file
     * @param filename The file name
     * @param content The file content
     */
    updateSingleFile(filename: string, content: string): void {}

    /**
     * Sets the active file
     * @param filename The file name
     */
    setActiveFile(filename: string): void {}

    /**
     * Gets the errors from the analysis
     * @returns An empty array of errors
     */
    getErrors(): any[] {
        return [];
    }

    /**
     * Provides auto-completion suggestions
     * @param line Line number
     * @param column Column number
     * @returns Empty auto-completion result
     */
    autoComplete(line?: number, column?: number): Record<string, any> {
        return {};
    }

    /**
     * Provides definition location
     * @param line Line number
     * @param column Column number
     * @returns Empty definition result
     */
    gotoDefinition(_line?: number, _column?: number): Record<string, any>[] {
        return [];
    }

    /**
     * Provides type information
     * @param line Line number
     * @param column Column number
     * @returns Empty type information
     */
    queryType(line?: number, column?: number): Record<string, any> {
        return {};
    }

    /**
     * Provides inlay hints
     * @returns Empty inlay hints
     */
    inlayHint(): Record<string, any> {
        return {};
    }
}

module.exports = {
    __esModule: true,
    default: (): Promise<void> => Promise.resolve(),
    State: MockState,
};
