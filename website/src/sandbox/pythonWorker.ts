/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import { PyodideStatus } from './PyodideStatus';

/*
 * This file is a Web Worker that runs Python code using Pyodide. It is used by the
 * Pyrefly sandbox to run Python code in the browser. We use a web worker to run
 * Python code in a separate thread, so that the main thread can continue to
 * render the UI while the Python code is running. This is especially useful in cases
 * where the Python code takes a long time to run.
 *
 */
import { loadPyodide, PyodideInterface } from 'pyodide';

export interface MultiFilePayload {
    activeFile: string;
    allFiles: Record<string, string>;
}
export type WorkerResponse =
    | stdoutResponseOrStderrResponse
    | RunPythonResponse
    | PyodideStatusResponse;

type stdoutResponseOrStderrResponse = {
    type: 'stdout' | 'stderr';
    output: string;
};

type RunPythonResponse = {
    type: 'runPython';
    success: boolean;
    error?: string;
};

type PyodideStatusResponse = {
    type: 'pyodideStatusUpdate';
    status: PyodideStatus;
};

let pyodideInstance: PyodideInterface = null;

const initPyodide = async () => {
    // Notify the main thread that we're initializing Pyodide
    self.postMessage({
        type: 'pyodideStatusUpdate',
        status: PyodideStatus.INITIALIZING,
    });

    const py = await loadPyodide({
        indexURL: `${self.location.origin}/pyodide`,
    });

    // Notify the main thread that Pyodide has been initialized
    self.postMessage({
        type: 'pyodideStatusUpdate',
        status: PyodideStatus.RUNNING,
    });

    // Set up stdout and stderr capture
    py.setStdout({
        batched: (output: string) => {
            self.postMessage({
                type: 'stdout',
                output,
            });
        },
    });

    py.setStderr({
        batched: (output: string) => {
            self.postMessage({
                type: 'stderr',
                output,
            });
        },
    });

    return py;
};

// Handle messages from the main thread
self.onmessage = async (event: MessageEvent<string | MultiFilePayload>) => {
    try {
        if (!pyodideInstance) {
            pyodideInstance = await initPyodide();
        }

        // Check for multi-file payload
        if (typeof event.data === 'string') {
            // If not then Single file execution 
            const code = event.data;
            pyodideInstance.runPython(code);
        } else {
            // Else Multi-file execution
            const { activeFile, allFiles } = event.data as MultiFilePayload;
            
            // Write all the files to Pyodide's filesystem
            for (const [filename, content] of Object.entries(allFiles)) {
                pyodideInstance.FS.writeFile(`/${filename}`, new TextEncoder().encode(content));
            }
            
            // Add root directory to Python path and invalidate caches
            pyodideInstance.runPython(`
import sys
import importlib
# Add root directory to Python path if not already there
if '/' not in sys.path:
    sys.path.insert(0, '/')
# Invalidate import caches so Python can find the modules
importlib.invalidate_caches()
            `);
            
            // Execute the active (currently selected) file
            const activeFileContent = allFiles[activeFile];
            if (activeFileContent) {
                pyodideInstance.runPython(activeFileContent);
            } else {
                throw new Error(`Active file '${activeFile}' not found in provided files`);
            }
        }

        // Send successful response back to the main thread
        const response: WorkerResponse = {
            type: 'runPython',
            success: true,
        };
        self.postMessage(response);
    } catch (error) {
        console.error('Python error:', error);

        // Send error response back to the main thread
        const response: WorkerResponse = {
            type: 'runPython',
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
        };
        self.postMessage(response);
    }
};
