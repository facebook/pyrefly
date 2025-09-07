/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import '@testing-library/jest-dom';
import { PyreflyState } from '../sandbox/Sandbox';
import { DEFAULT_SANDBOX_PROGRAM } from '../sandbox/DefaultSandboxProgram';
import {
    createPyreflyState,
    findError,
} from './__test_utils__/PyreflyWasmTestUtils';

describe('pyrefly_wasm', () => {
    let pyreService: PyreflyState;

    beforeAll(async () => {
        try {
            // Create a new PyreflyState instance using our test utility
            pyreService = await createPyreflyState();
        } catch (error) {
            console.error('Error initializing pyrefly_wasm:', error);
            throw error;
        }
    });

    beforeEach(() => {
        // Initialize sandbox files with both main.py and utils.py
        const utilsContent = `
def format_number(x: int) -> str:
    return str(x)
`;
        pyreService.updateSandboxFiles({
            'main.py': DEFAULT_SANDBOX_PROGRAM,
            'utils.py': utilsContent.trim()
        });
    });

    describe('getErrors', () => {
        it('simple python program, checks for errors for reveal type, bad assignment, parse error', () => {
            const programWithError = `
x: int = ""
import
`;
            pyreService.updateSingleFile('main.py',
                DEFAULT_SANDBOX_PROGRAM + programWithError
            );
            const errors = pyreService.getErrors();

            // Should have at least one error for reveal_type
            expect(errors.length).toBeGreaterThan(0);

            // Find the reveal_type error
            const revealTypeError = findError(errors, 'revealed type:');
            expect(revealTypeError).toBeDefined();
            expect(revealTypeError.kind).toEqual('reveal-type');
            // The revealed type should be 'str'
            const match = revealTypeError.message_header.match(
                /revealed type: ([^\s]+)/
            );
            const revealedType = match[1];
            expect(revealedType).toBe('str');

            const badAssignmentError = findError(
                errors,
                "`Literal['']` is not assignable to `int`"
            );
            expect(badAssignmentError).toBeDefined();
            expect(badAssignmentError.kind).toEqual('bad-assignment');

            const parseError = findError(
                errors,
                'Parse error: Expected one or more symbol names after import'
            );
            expect(parseError).toBeDefined();
            expect(parseError.kind).toEqual('parse-error');
        });

        it('complex python program, error with typedDict', () => {
            // Update source with a complete program
            pyreService.updateSingleFile('main.py',
                `
from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int

movie: Movie = {'name': 'Blade Runner',
                'year': '1982'}
`.trimStart()
            );

            // Test getErrors
            const errors = pyreService.getErrors();
            expect(errors.length).toEqual(1);

            // Should have revealed types for result and difference
            const typedDictError = findError(
                errors,
                "`Literal['1982']` is not assignable to TypedDict key `year` with type `int`"
            );
            expect(typedDictError).toBeDefined();
            expect(typedDictError.kind).toBe('typed-dict-key-error');
        });
    });

    describe('gotoDefinition', () => {
        it('should return definition location for function call', () => {
            // Position of "test" in "test(42)" 
            const definition = pyreService.gotoDefinition(18, 13);

            // Should return a location object
            expect(definition).toBeDefined();

            // expect that location is correct
            expect(definition.startLineNumber).toBe(13);
            expect(definition.startColumn).toBe(5);
            expect(definition.endLineNumber).toBe(13);
            expect(definition.endColumn).toBe(9);
        });
    });

    describe('autoComplete', () => {
        it('should return completion items for function name', () => {
            const typingForAutocomplete = `
tes
`;
            pyreService.updateSingleFile('main.py',
                DEFAULT_SANDBOX_PROGRAM + typingForAutocomplete
            );

            const completions = pyreService.autoComplete(19, 4);
            expect(completions.length).toBeGreaterThan(0);

            // Check that 'test' function appears in completions
            const testCompletion = completions.find(c => c.label === 'test');
            expect(testCompletion).toBeDefined();
            expect(testCompletion.detail).toContain('(x: int) -> str');
            expect(testCompletion.kind).toBe(3); // Function kind
        });
    });

    describe('hover', () => {
        it('should return type information for expressions', () => {
            // Set active file to main.py
            pyreService.setActiveFile('main.py');
            // Position of "test(42)" in reveal_type 
            const hoverInfo = pyreService.queryType(18, 13);

            expect(hoverInfo).toBeDefined();
            expect(hoverInfo.contents).toBeDefined();
            expect(hoverInfo.contents).toHaveLength(1);

            const hoverInfoContent = hoverInfo.contents[0];
            expect(hoverInfoContent.language).toEqual('python');
            expect(hoverInfoContent.value).toEqual('(x: int) -> str');
        });
    });

    describe('inlayHint', () => {
        it('should return inlay hints', () => {
            const hints = pyreService.inlayHint();
            expect(hints).toBeDefined();
            expect(hints.length).toBeGreaterThan(0);

            // Check the first hint
            const firstHint = hints[0];
            expect(firstHint.position).toEqual({ lineNumber: 13, column: 17 });
            expect(firstHint.label).toEqual(' -> str');
        });
    });
});
