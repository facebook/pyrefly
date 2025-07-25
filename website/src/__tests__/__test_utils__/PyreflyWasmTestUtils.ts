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
 * Create and initialize the pyrefly_wasm module
 */
export async function createPyreflyWasmModule() {
    try {
        // Import the pyrefly_wasm module for testing
        const pyreflyWasmModule = await import(
            '../wasm/pyrefly_wasm_for_testing'
        );
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
    return new pyreflyWasmModule.State();
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
