/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import * as docusaurusTheme from '@docusaurus/theme-common';
import MonacoEditorButton, {
    BUTTON_HEIGHT,
    runOnClickForAtLeastOneSecond,
} from './MonacoEditorButton';
import RunPythonButton from './RunPythonButton';
import { PyodideStatus } from './PyodideStatus';
import Editor from '@monaco-editor/react';
import * as LZString from 'lz-string';
import * as stylex from '@stylexjs/stylex';
import SandboxResults from './SandboxResults';
import {
    monaco,
    setAutoCompleteFunction,
    setGetDefFunction,
    setHoverFunctionForMonaco,
    setInlayHintFunctionForMonaco,
} from './configured-monaco';
import type { editor } from 'monaco-editor';
import type { PyreflyErrorMessage } from './SandboxResults';
import { DEFAULT_SANDBOX_PROGRAM } from './DefaultSandboxProgram';
import { DEFAULT_UTILS_PROGRAM } from './DefaultUtilsProgram';
import { usePythonWorker } from './usePythonWorker';
import PythonVersionSelector from './PythonVersionSelector';

// Import type for Pyrefly State
export interface PyreflyState {
    updateSandboxFiles: (files: Record<string, string>) => void;
    updateSingleFile: (filename: string, content: string) => void;
    setActiveFile: (filename: string) => void;
    getErrors: () => ReadonlyArray<PyreflyErrorMessage>;
    autoComplete: (line: number, column: number) => any;
    gotoDefinition: (line: number, column: number) => DefinitionResult | null;
    queryType: (line: number, column: number) => any;
    inlayHint: () => any;
}

// Lazy initialization function that will only be called when needed
export async function initializePyreflyWasm(): Promise<any> {
    const pyreflyWasmUninitializedPromise =
        typeof window !== 'undefined'
            ? import('../sandbox/pyrefly_wasm')
            : new Promise<any>((_resolve) => { });

    try {
        const mod = await pyreflyWasmUninitializedPromise;
        await mod.default();
        return await mod;
    } catch (e) {
        console.error(e);
        throw e;
    }
}

// Types for Pyrefly responses
export interface DefinitionResult {
    range: {
        startLineNumber: number;
        startColumn: number;
        endLineNumber: number;
        endColumn: number;
    };
    filename: string;
}

// This will be used in the component
let pyreflyWasmInitializedPromise: Promise<any> | null = null;

interface SandboxProps {
    sampleFilename: string;
    isCodeSnippet?: boolean;
    codeSample?: string;
    isInViewport?: boolean;
}

export default function Sandbox({
    sampleFilename,
    isCodeSnippet = false,
    codeSample = DEFAULT_SANDBOX_PROGRAM,
    isInViewport = true,
}: SandboxProps): React.ReactElement {
    const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
    const [errors, setErrors] =
        useState<ReadonlyArray<PyreflyErrorMessage> | null>([]);
    const [internalError, setInternalError] = useState('');
    const [loading, setLoading] = useState(true);
    const [pyreService, setPyreService] = useState<PyreflyState | null>(null);
    const [editorHeightforCodeSnippet, setEditorHeightforCodeSnippet] =
        useState<number | null>(null);
    const [models, setModels] = useState<Map<string, editor.ITextModel>>(new Map());
    const [activeFileName, setActiveFileName] = useState<string>('sandbox.py');
    const [renamingFile, setRenamingFile] = useState<string | null>(null);
    const [renameInputValue, setRenameInputValue] = useState<string>('');
    const [pyodideStatus, setPyodideStatus] = useState<PyodideStatus>(
        PyodideStatus.NOT_INITIALIZED
    );
    const [pythonOutput, setPythonOutput] = useState<string>('');
    const [activeTab, setActiveTab] = useState<string>('errors');
    const [isHovered, setIsHovered] = useState(false);
    const [pythonVersion, setPythonVersion] = useState('3.12');
    const model = models.get(activeFileName) || null;

    // File management functions
    const createNewFile = useCallback((fileName: string, content: string = '') => {

        // Lets see if model already exists in Monaco
        const existingModel = monaco.editor.getModels().find(m => m.uri.path === `/${fileName}`);

        let newModel;
        if (existingModel) {
            // File exists, update its content
            existingModel.setValue(content);
            newModel = existingModel;
        } else {
            // Create new file from scratch
            newModel = monaco.editor.createModel(content, 'python', monaco.Uri.file(`/${fileName}`));
        }

        setModels(prev => new Map(prev).set(fileName, newModel));
        setActiveFileName(fileName);
    }, []);

    // Switch to a different file
    const switchToFile = useCallback((fileName: string) => {
        if (models.has(fileName)) {
            setActiveFileName(fileName);
            // If editor exists, immediately switch the model
            const editor = editorRef.current;
            const targetModel = models.get(fileName);
            if (editor && targetModel) {
                editor.setModel(targetModel);
            }
            if (pyreService) {
                pyreService.setActiveFile(fileName);
            }
        }
    }, [models, pyreService]);

    // Create a new temporary file
    const createNewTempFile = useCallback(() => {
        // Prevent creating new file if already renaming one
        if (renamingFile) {
            return;
        }

        let counter = 1;
        let fileName = 'untitled.py';

        // Find next available untitled filename
        while (models.has(fileName)) {
            fileName = `untitled${counter}.py`;
            counter++;
        }

        // Create the file
        createNewFile(fileName, '# New Python file\n');

        // Start renaming mode with empty input
        setRenamingFile(fileName);
        setRenameInputValue('');
    }, [models, createNewFile, renamingFile]);


    // Rename a file
    const renameFile = useCallback((oldName: string, newName: string) => {
        if (!newName.trim() || models.has(newName)) return false;

        // Ensure .py extension
        const finalName = newName.endsWith('.py') ? newName : `${newName}.py`;
        if (models.has(finalName)) return false;

        const oldModel = models.get(oldName);
        if (!oldModel) return false;

        // Create new model with new name
        const newModel = monaco.editor.createModel(
            oldModel.getValue(),
            'python',
            monaco.Uri.file(`/${finalName}`)
        );

        // If this is the active file, switch the editor to the new model BEFORE disposing the old one
        const editor = editorRef.current;
        if (activeFileName === oldName && editor) {
            editor.setModel(newModel);
        }

        // Update models map
        setModels(prev => {
            const newMap = new Map(prev);
            newMap.delete(oldName);
            newMap.set(finalName, newModel);
            return newMap;
        });


        // Update active filename if needed
        if (activeFileName === oldName) {
            setActiveFileName(finalName);
        }

        // Dispose old model AFTER switching
        setTimeout(() => {
            oldModel.dispose();
        }, 100);

        return true;
    }, [models, activeFileName]);

    // Check if filename is valid (not empty and not duplicate)
    const isValidFilename = useCallback((inputValue: string, currentFileName: string) => {
        if (!inputValue.trim()) {
            return false;
        }

        const finalName = inputValue.trim().endsWith('.py')
            ? inputValue.trim()
            : `${inputValue.trim()}.py`;

        // Allow saving with the same name (no change)
        if (finalName === currentFileName) {
            return true;
        }

        // Check if another file with this name already exists
        return !models.has(finalName);
    }, [models]);

    // Handle rename save
    const handleRenameSave = useCallback(() => {
        if (!renamingFile) {
            return;
        }

        // Don't save if input is invalid
        if (!isValidFilename(renameInputValue, renamingFile)) {
            return;
        }

        const success = renameFile(renamingFile, renameInputValue.trim());
        if (success) {
            setRenamingFile(null);
            setRenameInputValue('');
        }
    }, [renamingFile, renameInputValue, renameFile, isValidFilename]);


    // Delete a file
    const deleteFile = useCallback((fileName: string) => {
        // Prevent deleting sandbox.py (one file must always exist)
        if (fileName === 'sandbox.py') {
            return false;
        }

        const modelToDelete = models.get(fileName);
        if (!modelToDelete) return false;

        // Remove from models map
        setModels(prev => {
            const newMap = new Map(prev);
            newMap.delete(fileName);
            return newMap;
        });


        // If this was the active file, switch to sandbox.py
        if (activeFileName === fileName) {
            setActiveFileName('sandbox.py');
            const editor = editorRef.current;
            const sandboxModel = models.get('sandbox.py');
            if (editor && sandboxModel) {
                editor.setModel(sandboxModel);
            }
        }

        // If this file is currently being renamed, reset the rename state
        if (renamingFile === fileName) {
            setRenamingFile(null);
            setRenameInputValue('');
        }

        // Dispose the model
        setTimeout(() => {
            modelToDelete.dispose();
        }, 100);

        return true;
    }, [models, activeFileName, renamingFile]);

    const TabBar = () => (
        <div {...stylex.props(styles.tabBar)}>
            {Array.from(models.keys()).map(fileName => (
                <div key={fileName} {...stylex.props(styles.tabContainer)}>
                    {renamingFile === fileName ? (
                        <div {...stylex.props(
                            styles.renameContainer,
                            !isValidFilename(renameInputValue, fileName) && styles.invalidInput
                        )}>
                            <input
                                {...stylex.props(styles.renameInput)}
                                value={renameInputValue}
                                onChange={(e) => setRenameInputValue(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' || (e.ctrlKey && e.key === 's')) {
                                        e.preventDefault();
                                        handleRenameSave();
                                    } else if (e.key === 'Escape') {
                                        setRenamingFile(null);
                                        setRenameInputValue('');
                                    }
                                }}
                                onBlur={handleRenameSave}
                                placeholder="Enter filename"
                                autoFocus
                            />
                            <span {...stylex.props(styles.extensionLabel)}>.py</span>
                        </div>
                    ) : (
                        <button
                            onClick={() => switchToFile(fileName)}
                            {...stylex.props(
                                styles.tabButton,
                                fileName === activeFileName && styles.tabButtonActive
                            )}
                        >
                            {fileName}
                        </button>
                    )}
                </div>
            ))}
            <div {...stylex.props(styles.buttonGroup)}>
                <button
                    onClick={() => deleteFile(activeFileName)}
                    {...stylex.props(
                        styles.actionButton,
                        styles.deleteButton,
                        activeFileName === 'sandbox.py' && styles.disabledButton
                    )}
                    title={activeFileName === 'sandbox.py' ? 'Cannot delete sandbox.py' : renamingFile ? 'Cancel' : 'Delete file'}
                    disabled={activeFileName === 'sandbox.py'}
                    onMouseDown={(e) => {
                        e.preventDefault();
                    }}
                >
                    ×
                </button>
                {renamingFile ? (
                    <button
                        onClick={handleRenameSave}
                        {...stylex.props(
                           styles.actionButton,
                            styles.saveButton,
                            !isValidFilename(renameInputValue, renamingFile) && styles.disabledButton
                        )}
                        title={isValidFilename(renameInputValue, renamingFile) ? "Save file name" : "Invalid filename"}
                        disabled={!isValidFilename(renameInputValue, renamingFile)}
                    >
                        ✓
                    </button>
                ) : (
                    <button
                        onClick={createNewTempFile}
                        {...stylex.props(styles.actionButton)}
                        title="Add new file"
                    >
                        +
                    </button>
                )}
            </div>
        </div>
    );

    useEffect(() => {
        if (models.size === 0) {
            const restored = restoreProjectFromURL(createNewFile, setActiveFileName, setModels);

            if (!restored) {
                if (isCodeSnippet) {
                    // For code snippets, use the provided filename and code sample
                    createNewFile(sampleFilename, codeSample);
                    setActiveFileName(sampleFilename);
                } else {
                    // For sandbox mode, create the default files
                    createNewFile('sandbox.py', DEFAULT_SANDBOX_PROGRAM);
                    createNewFile('utils.py', DEFAULT_UTILS_PROGRAM);
                    setActiveFileName('sandbox.py');
                }
            }
        }
    }, [createNewFile, models.size, isCodeSnippet, sampleFilename, codeSample]);

    // Initialize Python version from URL on component mount
    useEffect(() => {
        const versionFromURL = getVersionFromURL();
        if (versionFromURL) {
            setPythonVersion(versionFromURL);
        }
    }, []);

    // Initialize WebAssembly only when the component is in the viewport
    useEffect(() => {
        // Skip initialization if not in viewport
        if (!isInViewport) {
            return;
        }

        setLoading(true);
        // Initialize the WebAssembly module only when the component is mounted
        if (!pyreflyWasmInitializedPromise) {
            pyreflyWasmInitializedPromise = initializePyreflyWasm();
        }

        pyreflyWasmInitializedPromise
            .then((pyrefly) => {
                try {
                    setPyreService(new pyrefly.State(pythonVersion));
                    setLoading(false);
                    setInternalError('');
                } catch (e) {
                    setLoading(false);
                    setInternalError(`Failed to initialize with Python ${pythonVersion}: ${e}`);
                }
            })
            .catch((e) => {
                setLoading(false);
                setInternalError(JSON.stringify(e));
            });
    }, [isInViewport, pythonVersion]); // Re-run when isInViewport or pythonVersion changes

    // Need to add createModel handler in case monaco model was not created at mount time
    monaco.editor.onDidCreateModel((_newModel) => {
        const curModel = fetchCurMonacoModelAndTriggerUpdate(activeFileName);
        if (curModel) {
            setModels(prev => new Map(prev).set(activeFileName, curModel));
        }
    });

    // Ensure Monaco editor model is synced with active file
    useEffect(() => {
        const editor = editorRef.current;
        const targetModel = models.get(activeFileName);

        if (editor && targetModel && editor.getModel() !== targetModel) {
            editor.setModel(targetModel);
        }
    }, [activeFileName, models]);


    // Recheck when pyre service or model changes
    useEffect(() => {
        forceRecheck();
    }, [pyreService, model]);

    // Initial type check when models and pyreService are ready
    useEffect(() => {
        if (models.size > 0 && pyreService && model) {
            const allFiles: Record<string, string> = {};
            models.forEach((model, filename) => {
                allFiles[filename] = model.getValue();
            });

            if (Object.keys(allFiles).length > 0) {
                pyreService.updateSandboxFiles(allFiles);
                pyreService.setActiveFile(activeFileName);
            }

            setTimeout(() => forceRecheck(), 100);
        }
    }, [models.size, pyreService, model, activeFileName]);

    function forceRecheck() {
        if (model == null || pyreService == null) return;

        setAutoCompleteFunction(model, (l: number, c: number) =>
            pyreService.autoComplete(l, c)
        );
        setGetDefFunction(model, (l: number, c: number) => {
            const result = pyreService.gotoDefinition(l, c);

            if (result && result.filename !== activeFileName) {
                setTimeout(() => {
                    switchToFile(result.filename);
                }, 100);
            }
            return result;
        });
        setHoverFunctionForMonaco(model, (l: number, c: number) =>
            pyreService.queryType(l, c)
        );
        setInlayHintFunctionForMonaco(model, () => pyreService?.inlayHint() || []);

        // typecheck on edit
        try {
            if (models.size > 1) {
                const currentFileContent = model.getValue();
                pyreService.updateSingleFile(activeFileName, currentFileContent);
                pyreService.setActiveFile(activeFileName);
            } else {
                const value = model.getValue();
                pyreService.updateSingleFile(activeFileName, value);
            }

            const errors =
                pyreService.getErrors() as ReadonlyArray<PyreflyErrorMessage>;

            models.forEach((model) => {
                monaco.editor.setModelMarkers(model, 'default', []);
            });

            const errorsByFile = new Map<string, PyreflyErrorMessage[]>();
            errors.forEach(error => {
                const filename = error.filename || activeFileName;
                if (!errorsByFile.has(filename)) {
                    errorsByFile.set(filename, []);
                }
                errorsByFile.get(filename)!.push(error);
            });

            errorsByFile.forEach((fileErrors, filename) => {
                const fileModel = models.get(filename);
                if (fileModel) {
                    monaco.editor.setModelMarkers(
                        fileModel,
                        'default',
                        mapPyreflyErrorsToMarkerData(fileErrors)
                    );
                }
            });

            setInternalError('');
            setErrors(errors);

        } catch (e) {
            console.error(e);
            setInternalError(JSON.stringify(e));
            setErrors([]);
        }
    }

    const { runPython, runMultiFilePython } = usePythonWorker({
        setPythonOutput,
        setPyodideStatus,
    });

    // Create a function to run Python code that can be passed a model
    const runPythonCodeFunction = createRunPythonCodeFunction(
        setActiveTab,
        runPython,
        runMultiFilePython,
        models,
        activeFileName
    );
    const runPythonCodeCallback = useCallback(async () => {
        if (!model) return;

        // Set status to running before executing Python code
        setPyodideStatus(PyodideStatus.RUNNING);
        await runPythonCodeFunction(model);
    }, [model, setActiveTab, setPyodideStatus]);

    function onEditorMount(editor: editor.IStandaloneCodeEditor) {
        // Use activeFileName instead of sampleFilename
        const curModel = fetchCurMonacoModelAndTriggerUpdate(activeFileName);
        if (curModel) {
            setModels(prev => new Map(prev).set(activeFileName, curModel));
        }

        if (isCodeSnippet) {
            // Add extra space for buttons on mobile devices
            const extraSpace = isMobile() ? BUTTON_HEIGHT + 5 : 0; // Add 5px as buffer on top of button height
            setEditorHeightforCodeSnippet(
                Math.max(50, editor.getContentHeight() + extraSpace)
            );
        }
        editorRef.current = editor;

        // Add keyboard shortcut for Command+Enter (Mac) or Ctrl+Enter (Windows/Linux)
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
            if (
                !isCodeSnippet &&
                pyodideStatus !== PyodideStatus.INITIALIZING &&
                pyodideStatus !== PyodideStatus.RUNNING
            ) {
                // Use the model from the editor directly
                const editorModel = editor.getModel();
                if (editorModel) {
                    runOnClickForAtLeastOneSecond(
                        (running) =>
                            setPyodideStatus(
                                running
                                    ? PyodideStatus.RUNNING
                                    : PyodideStatus.FINISHED_RUNNING
                            ),
                        async () => {
                            await runPythonCodeFunction(editorModel);
                        }
                    );
                }
            }
        });
    }

    const handleGoToDefFromErrors = (
        startLineNumber: number,
        startColumn: number,
        endLineNumber: number,
        endColumn: number
    ) => {
        const editor = editorRef.current;
        if (editor === null) {
            return;
        }
        const range = {
            startLineNumber,
            startColumn,
            endLineNumber,
            endColumn,
        };

        editor.revealRange(range);
        editor.setSelection(range);
    };

    const handleVersionChange = (newVersion: string) => {
        setPythonVersion(newVersion);
        if (models.size > 0 && !isCodeSnippet) {
            const allFiles: Record<string, string> = {};
            models.forEach((model, filename) => {
                allFiles[filename] = model.getValue();
            });
            updateURL(allFiles, activeFileName, newVersion);
        }
    };

    const buttons = getMonacoButtons(
        isCodeSnippet,
        model,
        runPythonCodeCallback,
        pyodideStatus,
        setPyodideStatus,
        forceRecheck,
        codeSample,
        pythonVersion,
        handleVersionChange,
        loading,
        models,
        activeFileName
    );
    return (
        <div
            id="sandbox-editor"
            {...stylex.props(
                styles.tryEditor,
                !isCodeSnippet && !isMobile() && styles.sandboxPadding
            )}
        >
            {!isCodeSnippet && <TabBar />}
            <div
                id="sandbox-code-editor-container"
                {...stylex.props(
                    styles.codeEditorContainer,
                    isCodeSnippet && styles.codeEditorContainerWithRadius,
                    !isCodeSnippet && styles.codeEditorContainerWithTabs
                )}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
            >
                {getPyreflyEditor(
                    isCodeSnippet,
                    activeFileName,
                    model?.getValue() || codeSample,
                    forceRecheck,
                    onEditorMount,
                    editorHeightforCodeSnippet,
                    pythonVersion,
                    activeFileName,
                    models
                )}
                {
                    <div
                        {...stylex.props(
                            styles.buttonContainerBase,
                            isMobile()
                                ? isCodeSnippet
                                    ? styles.mobileButtonContainerCodeSnippet
                                    : styles.mobileButtonContainerSandbox
                                : styles.desktopButtonContainer,
                            // show button if it's in sandbox or if it's hovered or if it's mobile
                            // We only want to hide this if it's a code snippet not hovered on mobile
                            !isCodeSnippet || isHovered || isMobile()
                                ? styles.visibleButtonContainer
                                : styles.hiddenButtonContainer
                        )}
                    >
                        {buttons}
                    </div>
                }
            </div>
            {!isCodeSnippet && (
                <SandboxResults
                    loading={loading}
                    goToDef={handleGoToDefFromErrors}
                    errors={errors}
                    internalError={internalError}
                    pythonOutput={pythonOutput}
                    pyodideStatus={pyodideStatus}
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                />
            )}
        </div>
    );
}

interface ProjectState {
    files: Record<string, string>;
    activeFile: string;
}

function updateURL(allFiles: Record<string, string>, activeFile: string, version?: string): void {
    const projectState: ProjectState = {
        files: allFiles,
        activeFile: activeFile
    };
    const compressed = LZString.compressToEncodedURIComponent(JSON.stringify(projectState));
    const params = new URLSearchParams();
    params.set('project', compressed);
    if (version) {
        params.set('version', version);
    }
    const newURL = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newURL);
}

function getProjectFromURL(): ProjectState | null {
    if (typeof window === 'undefined') return null;
    const params = new URLSearchParams(window.location.search);

    const project = params.get('project');
    if (project) {
        try {
            const decompressed = LZString.decompressFromEncodedURIComponent(project);
            return decompressed ? JSON.parse(decompressed) : null;
        } catch (e) {
            console.error('Failed to parse project from URL:', e);
        }
    }

    const code = params.get('code');
    if (code) {
        const decompressed = LZString.decompressFromEncodedURIComponent(code);
        if (decompressed) {
            return {
                files: { 'sandbox.py': decompressed },
                activeFile: 'sandbox.py'
            };
        }
    }

    return null;
}

function getVersionFromURL(): string | null {
    if (typeof window === 'undefined') return null;
    const params = new URLSearchParams(window.location.search);
    return params.get('version');
}

function fetchCurMonacoModelAndTriggerUpdate(
    fileName: string
): editor.ITextModel | null {
    const model = monaco.editor
        .getModels()
        .filter((model) => model?.uri?.path === `/${fileName}`)[0];

    if (model == null) {
        return null;
    }

    // Force update to trigger initial inlay hint
    model.setValue(model.getValue());

    // Ensure tab size is correctly set
    model.updateOptions({ tabSize: 4, insertSpaces: true });

    return model;
}

function restoreProjectFromURL(
    createNewFile: (fileName: string, content: string) => void,
    setActiveFileName: (fileName: string) => void,
    setModels: React.Dispatch<React.SetStateAction<Map<string, editor.ITextModel>>>
): boolean {
    const projectState = getProjectFromURL();
    if (!projectState) return false;

    monaco.editor.getModels().forEach(model => model.dispose());
    setModels(new Map());

    Object.entries(projectState.files).forEach(([fileName, content]) => {
        createNewFile(fileName, content);
    });

    if (projectState.activeFile && projectState.files[projectState.activeFile]) {
        setActiveFileName(projectState.activeFile);
    }

    return true;
}

function isMobile(): boolean {
    return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
}

function getPyreflyEditor(
    isCodeSnippet: boolean,
    fileName: string,
    codeSample: string,
    forceRecheck: () => void,
    onEditorMount: (editor: editor.IStandaloneCodeEditor) => void,
    editorHeightforCodeSnippet: number | null,
    pythonVersion: string,
    activeFileName: string,
    models: Map<string, editor.ITextModel>
): React.ReactElement {
    const { colorMode } = docusaurusTheme.useColorMode();

    const editorTheme = colorMode === 'dark' ? 'vs-dark' : 'vs-light';
    if (isCodeSnippet) {
        return (
            <Editor
                defaultPath={fileName}
                defaultValue={codeSample}
                defaultLanguage="python"
                theme={editorTheme}
                onChange={forceRecheck}
                onMount={onEditorMount}
                keepCurrentModel={true}
                height={editorHeightforCodeSnippet}
                options={{
                    readOnly: isMobile(),
                    domReadOnly: isMobile(),
                    minimap: { enabled: false },
                    hover: { enabled: true, above: false },
                    scrollBeyondLastLine: false,
                    overviewRulerBorder: false,
                }}
            />
        );
    } else {
        // TODO (T217559369): Instead of manually calculating the sandbox height, we should
        // use flexbox behavior to make the sandbox height to be 75% of the screen
        // This doesn't seem to work with the monaco editor currently.
        const screenHeight = window.innerHeight;
        const navbarElement = document.querySelector(
            '.navbar'
        ) as HTMLElement | null; // Replace with your navbar selector
        const navbarHeight = navbarElement?.offsetHeight || 0;

        const sandboxHeight = ((screenHeight - navbarHeight) * 75) / 100;

        return (
            <Editor
                defaultPath={fileName}
                defaultValue={codeSample}
                defaultLanguage="python"
                theme={editorTheme}
                onChange={(value) => {
                    forceRecheck();
                    if (typeof value === 'string' && !isCodeSnippet) {
                        const allFiles: Record<string, string> = {};
                        models.forEach((model, filename) => {
                            if (filename === activeFileName) {
                                allFiles[filename] = value;
                            } else {
                                allFiles[filename] = model.getValue();
                            }
                        });
                        updateURL(allFiles, activeFileName, pythonVersion);
                    }
                }}
                onMount={onEditorMount}
                keepCurrentModel={true}
                height={sandboxHeight}
                options={{
                    readOnly: false,
                    domReadOnly: false,
                    minimap: { enabled: false },
                    hover: { enabled: true, above: false },
                    scrollBeyondLastLine: false,
                    overviewRulerBorder: false,
                }}
            />
        );
    }
}

function getMonacoButtons(
    isCodeSnippet: boolean,
    model: editor.ITextModel,
    runPythonCodeCallback: () => Promise<void>,
    pyodideStatus: PyodideStatus,
    setPyodideStatus: React.Dispatch<React.SetStateAction<PyodideStatus>>,
    forceRecheck: () => void,
    codeSample: string,
    pythonVersion: string,
    handleVersionChange: (version: string) => void,
    loading: boolean,
    models: Map<string, editor.ITextModel>,
    activeFileName: string
): ReadonlyArray<React.ReactElement> {
    let buttons: ReadonlyArray<React.ReactElement> = [];
    if (isCodeSnippet) {
        buttons = [
            <OpenSandboxButton model={model} />,
            getCopyButton(model),
            /* Hide reset button if it's readonly, which is when it's a code snippet on mobile */
            !isMobile()
                ? getResetButton(model, forceRecheck, codeSample, isCodeSnippet, pythonVersion, models, activeFileName)
                : null,
        ].filter(Boolean);
    } else {
        buttons = [
            getRunPythonButton(
                runPythonCodeCallback,
                pyodideStatus,
                setPyodideStatus
            ),
            getShareUrlButton(),
            getResetButton(model, forceRecheck, codeSample, isCodeSnippet, pythonVersion, models, activeFileName),
            getGitHubIssuesButton(model, pythonVersion),
            <PythonVersionSelector
                selectedVersion={pythonVersion}
                onVersionChange={handleVersionChange}
                loading={loading}
            />,
        ];
    }

    // react requires a unique key for each element in an array
    // Apply sandboxMobileButton style to buttons when they're in the sandbox (not code snippet) on mobile
    return buttons.map((button, index) =>
        React.cloneElement(button, {
            key: `button-${index}`,
            // Apply additional style for sandbox buttons on mobile
            ...(!isCodeSnippet && isMobile()
                ? { className: stylex.props(styles.sandboxMobileButton) }
                : {}),
        })
    );
}

// Monaco Editor Buttons
function getShareUrlButton(): React.ReactElement {
    return (
        <MonacoEditorButton
            id="share-url-button"
            onClick={() => {
                const currentURL = window.location.href;
                return navigator.clipboard.writeText(currentURL);
            }}
            defaultLabel="📋 Share URL"
            runningLabel="✓ URL Copied!" // we reuse the running label to indicate that the URL has been copied
            ariaLabel="share URL button"
        />
    );
}

function getGitHubIssuesButton(
    model: editor.ITextModel | null,
    pythonVersion: string
): React.ReactElement {
    return (
        <MonacoEditorButton
            id="github-issues-button"
            onClick={() => {
                const sandboxURL = window.location.href;
                const code = model ? model.getValue() : '';

                const title = 'Bug Report';
                // Prefill issue form fields by query param names matching field IDs in issue_template.yml
                const description = [
                    '```python',
                    code,
                    '```',
                    '',
                    `Python: ${pythonVersion}`,
                    '',
                    '<!-- Describe your bug -->',
                ].join('\n');

                const baseIssueURL =
                    'https://github.com/facebook/pyrefly/issues/new' +
                    `?template=issue_template.yml` +
                    `&labels=bug` +
                    `&title=${encodeURIComponent(title)}` +
                    `&sandbox=${encodeURIComponent(sandboxURL)}`;

                const issueURLWithDesc =
                    baseIssueURL + `&description=${encodeURIComponent(description)}`;

                // Guardrail: if URL too long, fall back to a concise placeholder description
                const MAX_URL_LEN = 1800;
                if (issueURLWithDesc.length <= MAX_URL_LEN) {
                    window.open(issueURLWithDesc, '_blank', 'noopener,noreferrer');
                    return Promise.resolve();
                } else {
                    const placeholderDescription =
                        '<!-- Code is too long to include here. Please paste the smallest reproducible snippet (≤30 lines), include your Python version, and briefly describe expected vs actual behavior. -->';
                    const fallbackURL =
                        baseIssueURL +
                        `&description=${encodeURIComponent(placeholderDescription)}`;
                    window.open(fallbackURL, '_blank', 'noopener,noreferrer');
                    return Promise.resolve();
                }
            }}
            defaultLabel="⚠️ Report Issue"
            runningLabel="⚠️ Report Issue"
            ariaLabel="report issue on GitHub"
        />
    );
}

function OpenSandboxButton({
    model,
}: {
    model: editor.ITextModel;
}): React.ReactElement {
    // This call is a react hook that must be called inside the function body rather than the return statement
    const sandboxBaseUrl = useBaseUrl('sandbox/');

    return (
        <MonacoEditorButton
            id="open-sandbox-button"
            onClick={async () => {
                if (model) {
                    const currentCode = model.getValue();
                    const compressed =
                        LZString.compressToEncodedURIComponent(currentCode);
                    // Navigate to the sandbox URL with the compressed code as a query parameter
                    const sandboxURL = sandboxBaseUrl + `?code=${compressed}`;
                    window.location.href = sandboxURL;
                }

                return Promise.resolve();
            }}
            defaultLabel="🧪 Open in Sandbox"
            runningLabel="⏳ Opening Sandbox..."
            ariaLabel="open in sandbox"
        />
    );
}

function getRunPythonButton(
    runPython: () => Promise<void>,
    pyodideStatus: PyodideStatus,
    setPyodideStatus: React.Dispatch<React.SetStateAction<PyodideStatus>>
): React.ReactElement {
    return (
        <RunPythonButton
            runPython={runPython}
            pyodideStatus={pyodideStatus}
            setPyodideStatus={setPyodideStatus}
        />
    );
}

function getCopyButton(model: editor.ITextModel): React.ReactElement {
    return (
        <MonacoEditorButton
            id="copy-code-button"
            onClick={async () => {
                if (model) {
                    const currentCode = model.getValue();
                    await navigator.clipboard.writeText(currentCode);
                }
                return Promise.resolve();
            }}
            defaultLabel="📋 Copy"
            runningLabel="✓ Copied!"
            ariaLabel="copy code to clipboard"
        />
    );
}

function getResetButton(
    model: editor.ITextModel | null,
    forceRecheck: () => void,
    codeSample: string,
    isCodeSnippet: boolean,
    pythonVersion: string,
    models: Map<string, editor.ITextModel>,
    activeFileName: string
): React.ReactElement {
    return (
        <MonacoEditorButton
            id="reset-button"
            onClick={async () => {
                if (model) {
                    model.setValue(codeSample);
                    if (!isCodeSnippet) {
                        const allFiles: Record<string, string> = {};
                        models.forEach((model, filename) => {
                            if (filename === activeFileName) {
                                allFiles[filename] = codeSample;
                            } else {
                                allFiles[filename] = model.getValue();
                            }
                        });
                        updateURL(allFiles, activeFileName, pythonVersion);
                    }
                    forceRecheck();
                }
            }}
            defaultLabel="🔄 Reset"
            runningLabel="✓ Reset!"
            ariaLabel="reset to default code"
        />
    );
}

// Reusable function to create a Python code runner
export function createRunPythonCodeFunction(
    onActiveTabChange: React.Dispatch<React.SetStateAction<string>>,
    runPython: (code: string) => Promise<void>,
    runMultiFilePython: (activeFile: string, allFiles: Record<string, string>) => Promise<void>,
    models: Map<string, editor.ITextModel>,
    activeFileName: string
) {
    // Return a function that takes a model parameter
    return async (model: editor.ITextModel | null) => {
        if (!model) return;

        // Switch to output tab
        onActiveTabChange('output');

        // Check if this is a multi-file project
        if (models.size > 1) {
            const allFiles: Record<string, string> = {};
            models.forEach((model, filename) => {
                allFiles[filename] = model.getValue();
            });

            await runMultiFilePython(activeFileName, allFiles);
        } else {
            const code = model.getValue();
            await runPython(code);
        }
    };
}

/**
 * Maps PyreflyErrorMessage array to Monaco editor IMarkerData array
 */
function mapPyreflyErrorsToMarkerData(
    errors: ReadonlyArray<PyreflyErrorMessage>
): editor.IMarkerData[] {
    return errors.map((error) => {
        const message = error.message_details
            ? `${error.message_header}\n${error.message_details}`
            : error.message_header;
        return {
            startLineNumber: error.startLineNumber,
            startColumn: error.startColumn,
            endLineNumber: error.endLineNumber,
            endColumn: error.endColumn,
            message: message,
            severity: error.severity,
        };
    });
}

// Styles for Sandbox component
const styles = stylex.create({
    tryEditor: {
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
    },
    sandboxPadding: {
        paddingHorizontal: '10px',
    },
    codeEditorContainer: {
        position: 'relative',
        display: 'flex',
        overflow: 'visible',
        borderBottom: '10px',
        background: 'var(--color-background)',
        height: '100%',
    },
    codeEditorContainerWithRadius: {
        border: '1px solid var(--color-background-secondary)',
        borderRadius: '0.25rem',
    },
    codeEditorContainerWithTabs: {
        border: '1px solid var(--color-border)',
        borderTop: 'none',
        borderBottomLeftRadius: '0.25rem',
        borderBottomRightRadius: '0.25rem',
        paddingTop: '4px',
    },
    buttonContainerBase: {
        position: 'absolute',
        display: 'flex',
        zIndex: 10, // used to ensure it's beneath the navbar
        gap: '8px',
    },
    mobileButtonContainerCodeSnippet: {
        // Position at bottom right for mobile
        bottom: '16px',
        right: '16px',
    },
    mobileButtonContainerSandbox: {
        // Position at bottom right for mobile
        bottom: '16px',
        right: '16px',
        flexDirection: 'column', // Buttons stack vertically on mobile for sandbox
    },
    // Style for desktop buttons (hidden by default, visible on hover)
    desktopButtonContainer: {
        // Position at top right for desktop
        top: '20px',
        right: '20px',
    },
    // Style for hidden button Container
    hiddenButtonContainer: {
        opacity: 0,
        visibility: 'hidden',
        transition: 'opacity 0.2s ease-in-out, visibility 0.2s ease-in-out',
    },
    // Style for visible button Container
    visibleButtonContainer: {
        // Visible when hovered
        opacity: 1,
        visibility: 'visible',
        transition: 'opacity 0.2s ease-in-out, visibility 0.2s ease-in-out',
    },
    // Style for sandbox mobile buttons
    sandboxMobileButton: {
        '@media (max-width: 768px)': {
            margin: '0', // No margin needed as gap is handled by the container
            width: '100%', // Make buttons full width on mobile for sandbox
        },
    },
    // TabBar styles matching SandboxResults tabs
    tabBar: {
        display: 'flex',
        background: 'var(--color-background)',
        borderBottom: '1px solid var(--color-background-secondary)',
        fontSize: '14px',
        borderTopLeftRadius: '0.25rem',
        borderTopRightRadius: '0.25rem',
        alignItems: 'center',
    },
    tabButton: {
        borderRight: '1px solid var(--color-background-secondary)',
        cursor: 'pointer',
        fontWeight: 'bold',
        padding: '7px 15px',
        border: 'none',
        backgroundColor: 'transparent',
        color: 'var(--color-text)',
        fontSize: '14px',
        ':last-child': {
            borderRight: 'none',
        },
    },
    tabButtonActive: {
        background: 'var(--color-background)',
        borderBottom: '2px solid var(--color-text)',
        marginBottom: '-1px', // cover up container bottom border
    },
    buttonGroup: {
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        marginLeft: '8px', // Small gap from last tab
    },
    actionButton: {
        border: 'none',
        background: 'transparent',
        color: 'var(--color-text)',
        cursor: 'pointer',
        padding: '7px 10px',
        fontSize: '16px',
        fontWeight: 'bold',
        borderRadius: '4px',
        minWidth: '32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        ':hover': {
            backgroundColor: 'var(--color-background-secondary)',
        },
    },
    deleteButton: {
        fontSize: '18px',
        ':hover': {
            color: 'var(--color-danger, #d32f2f)',
        },
    },
    saveButton: {
        ':hover': {
            color: 'var(--color-success, #2e7d32)',
        },
    },
    disabledButton: {
        opacity: 0.4,
        cursor: 'not-allowed',
        ':hover': {
            backgroundColor: 'transparent',
            color: 'var(--color-text)',
        },
    },
    tabContainer: {
        display: 'flex',
        alignItems: 'center',
    },
    renameContainer: {
        display: 'flex',
        alignItems: 'center',
        padding: '4px 8px',
        backgroundColor: 'var(--color-background)',
        border: '1px solid var(--color-primary)',
        borderRadius: '3px',
        margin: '3px',
    },
    invalidInput: {
        borderColor: 'var(--color-danger, #d32f2f)',
    },
    renameInput: {
        border: 'none',
        background: 'transparent',
        color: 'var(--color-text)',
        fontSize: '14px',
        fontWeight: 'bold',
        outline: 'none',
        minWidth: '80px',
        maxWidth: '150px',
    },
    extensionLabel: {
        color: 'var(--color-text-secondary)',
        fontSize: '14px',
        fontWeight: 'bold',
        marginLeft: '2px',
    },
});
