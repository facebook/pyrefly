/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import '@testing-library/jest-dom';
import { act } from 'react';
import { fireEvent, render } from '@testing-library/react';
import Sandbox from '../sandbox/Sandbox';
import { SANDBOX_FILE_NAME } from '../pages/sandbox';
import { DEFAULT_SANDBOX_PROGRAM } from '../sandbox/DefaultSandboxProgram';

describe('Sandbox Component', () => {
    const pyreflyCommit = '123456789abcdef';

    beforeAll(() => {
        process.env.PYREFLY_COMMIT = pyreflyCommit;
    });

    afterAll(() => {
        delete process.env.PYREFLY_COMMIT;
    });

    test('render sandbox correctly', async () => {
        const container = await act(async () => {
            const { container } = render(
                <Sandbox sampleFilename={SANDBOX_FILE_NAME} />
            );

            await Promise.resolve(); // Let any promises from timers resolve
            return container;
        });

        expectMonacoEditorLoadedWithContent(
            container,
            SANDBOX_FILE_NAME,
            DEFAULT_SANDBOX_PROGRAM,
            false
        );
        const buildLink = container.querySelector('#pyrefly-build');
        expect(buildLink).toHaveTextContent(
            'Pyrefly 1.2.3-test (commit 123456789)'
        );
        expect(buildLink).toHaveAttribute(
            'href',
            `https://github.com/facebook/pyrefly/commit/${pyreflyCommit}`
        );

        // Run test with --update-snapshot to update the snapshot if the test is failing after
        // you made a intentional change to the home page
        expect(container).toMatchSnapshot();
    });

    test('renders in code snippet mode without error panel', async () => {
        const fileName = 'snippet.py';
        const programContent = 'def hello(): pass';
        const container = await act(async () => {
            const { container } = render(
                <Sandbox
                    sampleFilename={fileName}
                    isCodeSnippet={true}
                    codeSample={programContent}
                />
            );

            await Promise.resolve(); // Let any promises from timers resolve
            return container;
        });

        expectMonacoEditorLoadedWithContent(
            container,
            fileName,
            programContent,
            true
        );

        // Run test with --update-snapshot to update the snapshot if the test is failing after
        // you made a intentional change to the home page
        expect(container).toMatchSnapshot();
    });

    test('toggles parameter name hints', async () => {
        const { getByLabelText } = await act(async () => {
            const result = render(
                <Sandbox sampleFilename={SANDBOX_FILE_NAME} />
            );
            await Promise.resolve();
            return result;
        });
        const toggle = getByLabelText('Parameter hints');

        expect(toggle).not.toBeChecked();
        fireEvent.click(toggle);
        expect(toggle).toBeChecked();
    });
    test('omits the commit for local builds', async () => {
        delete process.env.PYREFLY_COMMIT;
        try {
            const container = await act(async () => {
                const { container } = render(
                    <Sandbox sampleFilename={SANDBOX_FILE_NAME} />
                );

                await Promise.resolve();
                return container;
            });

            expect(
                container.querySelector('#pyrefly-build')
            ).not.toBeInTheDocument();
        } finally {
            process.env.PYREFLY_COMMIT = pyreflyCommit;
        }
    });

    function expectMonacoEditorLoadedWithContent(
        container: HTMLElement,
        fileName: string,
        programContent: string,
        isCodeSnippet: boolean
    ) {
        const sandboxEditorElement = container.querySelector('#sandbox-editor');
        expect(sandboxEditorElement).toBeInTheDocument();

        // Verify that the code editor container is a child of sandbox-editor
        const codeEditorContainer = sandboxEditorElement?.querySelector(
            '#sandbox-code-editor-container'
        );
        expect(codeEditorContainer).toBeInTheDocument();

        // Verify that the results container is a child of sandbox-editor
        const resultsContainer = sandboxEditorElement?.querySelector(
            '#sandbox-results-container'
        );
        if (isCodeSnippet) {
            expect(resultsContainer).not.toBeInTheDocument();
        } else {
            expect(resultsContainer).toBeInTheDocument();
        }

        // Verify that the Monaco editor is rendered
        const monacoEditor =
            codeEditorContainer.querySelector('#monaco-editor');
        expect(monacoEditor).toBeInTheDocument();

        // Verify that the editor has the correct path
        expect(monacoEditor.textContent).toContain(
            `Monaco Editor (Path: ${fileName})`
        );

        // Verify that the editor textarea is rendered
        const editorTextarea = monacoEditor.querySelector('#editor-textarea');
        expect(editorTextarea).toBeInTheDocument();

        // Verify that the editor content exactly matches the default Python program
        expect(editorTextarea).toHaveValue(programContent);

        // Verify that the share URL button exists
        const shareUrlButton =
            codeEditorContainer.querySelector('#share-url-button');
        if (isCodeSnippet) {
            expect(shareUrlButton).not.toBeInTheDocument();
        } else {
            expect(shareUrlButton).toBeInTheDocument();
        }
    }
});
