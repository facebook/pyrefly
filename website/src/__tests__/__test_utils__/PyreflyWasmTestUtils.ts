/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

'use strict';

import { PyreflyState } from '../../sandbox/Sandbox';
import { PyreflyErrorMessage } from '../../sandbox/SandboxResults';

/**
 * Check if the pyrefly_wasm module is available (built and copied to test directory)
 */
export async function isPyreflyWasmAvailable(): Promise<boolean> {
    try {
        const pyreflyWasmModule = await import(
            '../wasm/pyrefly_wasm_for_testing'
        );
        // Check if State constructor exists (not just a stub)
        return typeof (pyreflyWasmModule as { State?: unknown }).State === 'function';
    } catch {
        return false;
    }
}

/**
 * Create and initialize the pyrefly_wasm module
 * Note: The wasm module must be built and copied to the test directory for tests to work.
 * See src/__tests__/wasm/pyrefly_wasm_for_testing.js for details.
 */
export async function createPyreflyWasmModule(): Promise<{
    State: new (pythonVersion: string) => PyreflyState;
}> {
    try {
        // Import the pyrefly_wasm module for testing
        // Use type assertion since the stub file doesn't have proper types
        const pyreflyWasmModule = (await import(
            '../wasm/pyrefly_wasm_for_testing'
        )) as unknown as { State: new (pythonVersion: string) => PyreflyState };
        return pyreflyWasmModule;
    } catch (error) {
        console.error('Error initializing pyrefly_wasm:', error);
        throw error;
    }
}

/**
 * Create a new PyreflyState instance
 */
export async function createPyreflyState(): Promise<PyreflyState> {
    const pyreflyWasmModule = await createPyreflyWasmModule();
    return new pyreflyWasmModule.State('3.12');
}

/**
 * Find a specific error in the errors array
 */
export function findError(
    errors: ReadonlyArray<PyreflyErrorMessage>,
    errorText: string
): PyreflyErrorMessage | undefined {
    return errors.find((error) => error.message_header.includes(errorText));
}
