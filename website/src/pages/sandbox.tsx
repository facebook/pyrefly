/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import * as stylex from '@stylexjs/stylex';
import { resetPersistedSandboxState } from '../sandbox/persistedSandboxState';

const Sandbox = React.lazy(() => import('../sandbox/Sandbox'));

export const SANDBOX_FILE_NAME = 'sandbox.py';

export default function SandboxPage(): React.JSX.Element {
    const { siteConfig = {} } = useDocusaurusContext();
    return (
        <Layout
            title="Try Pyrefly: the Pyrefly Sandbox"
            description={siteConfig.description}
            noFooter
        >
            {
                // TODO (T222948083) Re-enable internal sandbox after we fix issues with sandbox on static docs
                process.env.INTERNAL_STATIC_DOCS === '1' ? (
                    <header {...stylex.props(styles.title)}>
                        Sandbox isn't currently available Internally, please use
                        the{' '}
                        <a
                            href="https://pyrefly.org/sandbox"
                            {...stylex.props(styles.hyperlink)}
                        >
                            {' '}
                            public sandbox
                        </a>
                    </header>
                ) : (
                    <BrowserOnly>
                        {() => (
                            <React.Suspense fallback={<div>Loading...</div>}>
                                <SandboxErrorBoundary>
                                    <Sandbox
                                        sampleFilename={SANDBOX_FILE_NAME}
                                    />
                                </SandboxErrorBoundary>
                            </React.Suspense>
                        )}
                    </BrowserOnly>
                )
            }
        </Layout>
    );
}

class SandboxErrorBoundary extends React.Component<
    { children: React.ReactNode },
    { error: Error | null }
> {
    state: { error: Error | null } = { error: null };

    static getDerivedStateFromError(error: Error): { error: Error } {
        return { error };
    }

    componentDidCatch(error: Error, info: React.ErrorInfo): void {
        console.error('Sandbox crashed:', error, info.componentStack);
    }

    render(): React.ReactNode {
        if (this.state.error) {
            return (
                <SandboxCrashFallback
                    error={this.state.error}
                    tryAgain={() => this.setState({ error: null })}
                    resetSandbox={() => {
                        resetPersistedSandboxState();
                        this.setState({ error: null });
                    }}
                />
            );
        }
        return this.props.children;
    }
}

function SandboxCrashFallback({
    error,
    resetSandbox,
    tryAgain,
}: {
    error: Error;
    resetSandbox: () => void;
    tryAgain: () => void;
}): React.JSX.Element {
    return (
        <div role="alert" {...stylex.props(styles.crashContainer)}>
            <h1 {...stylex.props(styles.crashTitle)}>The sandbox crashed.</h1>
            <p {...stylex.props(styles.crashText)}>
                Try again keeps your files and reruns the sandbox. Reset clears
                your saved sandbox files and starts from the default sandbox.
            </p>
            <div {...stylex.props(styles.crashActions)}>
                <button
                    type="button"
                    {...stylex.props(styles.primaryButton)}
                    onClick={tryAgain}
                >
                    Try again
                </button>
                <button
                    type="button"
                    {...stylex.props(styles.secondaryButton)}
                    onClick={resetSandbox}
                >
                    Reset sandbox
                </button>
            </div>
            <pre {...stylex.props(styles.crashError)}>
                {error.stack ?? String(error)}
            </pre>
        </div>
    );
}

const styles = stylex.create({
    title: {
        marginTop: '10px',
        marginLeft: '10px',
        fontSize: 32,
    },
    hyperlink: {
        textDecoration: 'underline',
        color: '#337ab7',
    },
    crashContainer: {
        margin: '48px auto',
        maxWidth: '720px',
        padding: '0 20px',
    },
    crashTitle: {
        fontSize: '32px',
        marginBottom: '12px',
    },
    crashText: {
        fontSize: '16px',
        lineHeight: 1.5,
        marginBottom: '20px',
    },
    crashActions: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: '12px',
        marginBottom: '24px',
    },
    primaryButton: {
        background: 'var(--ifm-color-primary)',
        border: '1px solid var(--ifm-color-primary)',
        borderRadius: '4px',
        color: '#fff',
        cursor: 'pointer',
        fontWeight: 600,
        padding: '10px 16px',
    },
    secondaryButton: {
        background: 'var(--ifm-background-color)',
        border: '1px solid var(--ifm-color-primary)',
        borderRadius: '4px',
        color: 'var(--ifm-color-primary)',
        cursor: 'pointer',
        fontWeight: 600,
        padding: '10px 16px',
    },
    crashError: {
        background: 'var(--ifm-code-background)',
        borderRadius: '4px',
        fontSize: '12px',
        overflow: 'auto',
        padding: '12px',
        whiteSpace: 'pre-wrap',
    },
});
