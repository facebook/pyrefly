/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import * as stylex from '@stylexjs/stylex';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { PyodideStatus } from './PyodideStatus';

export interface PyreflyErrorMessage {
    startLineNumber: number;
    startColumn: number;
    endLineNumber: number;
    endColumn: number;
    message_header: string;
    message_details: string;
    kind: string;
    severity: number;
    filename: string;
}

export type GoToDefFromError = (
    startLineNumber: number,
    startColumn: number,
    endLineNumber: number,
    endColumn: number
) => void;

interface ErrorMessageProps {
    error: PyreflyErrorMessage;
    goToDef: GoToDefFromError;
}

interface SandboxResultsProps {
    loading: boolean;
    goToDef: GoToDefFromError;
    errors?: ReadonlyArray<PyreflyErrorMessage> | null;
    internalError: string;
    pythonOutput: string;
    pyodideStatus: PyodideStatus;
    activeTab: string;
    setActiveTab: (tab: string) => void;
    height?: number; // Dynamic height in pixels (from resizable splitter)
}

function ErrorMessage({
    error,
    goToDef,
}: ErrorMessageProps): React.ReactElement {
    // This logic is meant to be an exact match of how we output errors in the cli defined here:
    // - https://fburl.com/code/e9lqk0h2
    // - https://fburl.com/code/hwhe60zt
    // TODO (T217247871): expose full error message from Pyrefly binary and use it directly here instead of duplicating the logic
    const { startLineNumber, startColumn, endLineNumber, endColumn } = error;

    let rangeStr: string;
    if (startLineNumber === endLineNumber) {
        if (startColumn === endColumn) {
            rangeStr = `${startLineNumber}:${startColumn}`;
        } else {
            rangeStr = `${startLineNumber}:${startColumn}-${endColumn}`;
        }
    } else {
        rangeStr = `${startLineNumber}:${startColumn}-${endLineNumber}:${endColumn}`;
    }

    const errorKindUrl = (
        <a
            href={useBaseUrl(`en/docs/error-kinds/#${error.kind}`)}
            target="_blank"
            // adding this to address security considerations: https://www.dhiwise.com/post/html-open-link-in-new-tab-improve-user-navigation
            rel="noopener noreferrer"
        >
            {error.kind}
        </a>
    );

    return (
        <span
            {...stylex.props(styles.msgType)}
            onClick={() =>
                goToDef(startLineNumber, startColumn, endLineNumber, endColumn)
            }
        >
            {
                // We intentionally add a leading space to INFO and WARN so they visually line up with ERROR
                error.severity == 2 ? (
                    <span {...stylex.props(styles.ErrorMessageInfo)}>
                        {' '}
                        INFO{' '}
                    </span>
                ) : error.severity == 4 ? (
                    <span {...stylex.props(styles.ErrorMessageWarning)}>
                        {' '}
                        WARN{' '}
                    </span>
                ) : (
                    <span {...stylex.props(styles.ErrorMessageError)}>
                        ERROR{' '}
                    </span>
                )
            }
            {`${error.filename}:${rangeStr}: ${error.message_header} `}
            {'['}
            {errorKindUrl}
            {']'}
            {error.message_details && '\n' + error.message_details}
        </span>
    );
}

function LoadingAnimation(): React.ReactElement {
    return (
        <div>
            <div {...stylex.props(styles.loader)}>
                <div {...stylex.props(styles.loaderDot, styles.bounce1)}></div>
                <div {...stylex.props(styles.loaderDot, styles.bounce2)}></div>
                <div {...stylex.props(styles.loaderDot)}></div>
            </div>
        </div>
    );
}
export default function SandboxResults({
    loading,
    goToDef,
    errors,
    internalError,
    pythonOutput,
    pyodideStatus,
    activeTab,
    setActiveTab = () => {},
    height,
}: SandboxResultsProps): React.ReactElement {
    const activeToolbarTab = activeTab;

    const hasTypecheckErrors =
        errors !== undefined && errors !== null && errors.length > 0;

    // Use dynamic height if provided, otherwise fall back to default CSS
    const containerStyle = height ? { height: `${height}px` } : undefined;

    return (
        <div
            id="sandbox-results-container"
            {...stylex.props(styles.resultsContainer)}
            style={containerStyle}
        >
            <div {...stylex.props(styles.resultsToolbar)}>
                <ul {...stylex.props(styles.tabs, styles.tabsEnabled)}>
                    <li
                        {...stylex.props(
                            styles.tab,
                            activeToolbarTab === 'errors' && styles.selectedTab
                        )}
                        onClick={() => setActiveTab('errors')}
                    >
                        Type Checking Errors
                    </li>
                    <li
                        {...stylex.props(
                            styles.tab,
                            activeToolbarTab === 'output' && styles.selectedTab
                        )}
                        onClick={() => setActiveTab('output')}
                    >
                        Runtime Output
                    </li>
                </ul>
                {/* TODO (T217536145): Add JSON tab to sandbox */}
            </div>
            <div {...stylex.props(styles.results)}>
                {loading && LoadingAnimation()}
                {!loading && activeToolbarTab === 'errors' && (
                    <pre {...stylex.props(styles.resultBody)}>
                        <ul {...stylex.props(styles.errorsList)}>
                            {hasTypecheckErrors ? (
                                errors
                                    .filter((error) => error.severity > 1)
                                    .map((error, i) => (
                                        <li
                                            key={i}
                                            {...stylex.props(
                                                i > 0 && styles.errorItemSibling
                                            )}
                                        >
                                            <ErrorMessage
                                                key={i}
                                                error={error}
                                                goToDef={goToDef}
                                            />
                                        </li>
                                    ))
                            ) : (
                                <li>
                                    {internalError
                                        ? `Pyrefly encountered an internal error: ${internalError}.`
                                        : errors === undefined ||
                                            errors === null
                                          ? 'Pyrefly failed to fetch errors.'
                                          : errors?.length === 0
                                            ? 'No errors!'
                                            : null}
                                </li>
                            )}
                        </ul>
                    </pre>
                )}
                {activeToolbarTab === 'output' && (
                    <pre {...stylex.props(styles.resultBody)}>
                        <span {...stylex.props(styles.PyodideDisclaimer)}>
                            {`Note: The Python runtime currently only supports features up to Python version 3.12, features from newer versions are not available. \n\n`}
                        </span>

                        {pyodideStatus === PyodideStatus.NOT_INITIALIZED
                            ? 'No output, please press the ▶️ Run button.'
                            : pyodideStatus === PyodideStatus.INITIALIZING
                              ? 'Loading Python interpreter...'
                              : pyodideStatus === PyodideStatus.RUNNING
                                ? 'Running...'
                                : pythonOutput.trimStart()}
                    </pre>
                )}
                {/* TODO (T217536145): Add JSON tab to sandbox */}
            </div>
        </div>
    );
}

// Define keyframes for animations
const skBounceDelayKeyframes = stylex.keyframes({
    '0%, 80%, 100%': { transform: 'scale(0)' },
    '40%': { transform: 'scale(1)' },
});

// Styles for SandboxResults component
const styles = stylex.create({
    resultsContainer: {
        height: 'calc(25vh - var(--ifm-navbar-height) / 4)', // 25% of screen height - nav bar
        position: 'relative',
        fontSize: '12px',
        background: 'var(--color-background)',
        borderLeft: '1px solid var(--color-background)',
    },
    resultsToolbar: {
        display: 'flex',
        background: 'var(--color-background)',
        borderBottom: '1px solid var(--color-background-secondary)',
        fontSize: '14px',
    },
    results: {
        overflow: 'auto',
        height: '80%',
    },
    resultBody: {
        paddingVertical: '7px',
        paddingHorizontal: '10px',
        marginBottom: 0,
        background: 'var(--color-background)',
        width: '100%',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        display: 'block',
    },
    tabs: {
        display: 'flex',
        listStyle: 'none',
        margin: 0,
        padding: 0,
        pointerEvents: 'none',
    },
    tabsEnabled: {
        pointerEvents: 'auto',
    },
    tab: {
        borderRight: '1px solid var(--color-background-secondary)',
        cursor: 'pointer',
        fontWeight: 'bold',
        padding: '7px 15px',
    },
    selectedTab: {
        background: 'var(--color-background)',
        borderBottom: '2px solid var(--color-text)',
        marginBottom: '-1px', // cover up container bottom border
    },
    loader: {
        display: 'flex',
        justifyContent: 'center',
        marginTop: '10px',
    },
    loaderDot: {
        width: '14px',
        height: '14px',
        backgroundColor: 'var(--color-background)',
        borderRadius: '100%',
        animationName: skBounceDelayKeyframes,
        animationDuration: '1.4s',
        animationIterationCount: 'infinite',
        animationTimingFunction: 'ease-in-out',
        animationFillMode: 'both',
    },
    bounce1: {
        animationDelay: '-320ms',
    },
    bounce2: {
        animationDelay: '-160ms',
    },
    errorsList: {
        listStyle: 'none',
        margin: 0,
        padding: 0,
    },
    errorItemSibling: {
        marginTop: '10px',
        paddingTop: '10px',
        borderTop: 'solid var(--color-background-secondary) 1px',
        background: 'var(--color-background)',
    },
    errorNestedItem: {
        padding: 'inherit',
        paddingLeft: '20px',
        margin: 'inherit',
        border: 'none',
    },
    msgHighlight: {
        cursor: 'pointer',
    },
    msgType: {
        cursor: 'pointer',
    },
    ErrorMessageError: {
        color: '#ed0a0a', // red
    },
    ErrorMessageWarning: {
        color: '#ffde21', // yellow
    },
    ErrorMessageInfo: {
        color: '#00aa00', // green
    },
    PyodideDisclaimer: {
        color: 'var(--color-primary)',
    },
});
