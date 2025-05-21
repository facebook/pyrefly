/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import BrowserOnly from '@docusaurus/BrowserOnly';
import * as React from 'react';
import * as stylex from '@stylexjs/stylex';

// Lazy load the Sandbox component
const Sandbox = React.lazy(() => import('./Sandbox'));

interface CodeSnippetProps {
    sampleFilename: string;
    isCodeSnippet?: boolean;
    codeSample?: string;
}

export default function CodeSnippet({
    sampleFilename,
    codeSample = '',
}: CodeSnippetProps): JSX.Element {
    if (sampleFilename == null) {
        throw "Missing sampleFilename. IDE services won't work properly.";
    }

    // State to track visibility
    const [isVisible, setIsVisible] = React.useState(false);
    const [hasBeenVisible, setHasBeenVisible] = React.useState(false);
    const preRef = React.useRef(null);

    // Use Intersection Observer to detect when the code snippet is visible
    React.useEffect(() => {
        if (typeof window === 'undefined') return;
        if (!preRef.current) return;

        const observer = new IntersectionObserver(
            (entries) => {
                const entry = entries[0];
                setIsVisible(entry.isIntersecting);
                if (entry.isIntersecting) {
                    setHasBeenVisible(true);
                }
            },
            {
                rootMargin: '500px 0px', // Load when within 500px of viewport
                threshold: 0.1,
            }
        );

        observer.observe(preRef.current);
        return () => observer.disconnect();
    }, []);

    // Only render the Sandbox component when the code snippet has been visible
    return (
        <div ref={preRef} {...stylex.props(styles.wrapper)}>
            {!hasBeenVisible && (
                <div {...stylex.props(styles.placeholder)}>
                    <div {...stylex.props(styles.placeholderContent)}>
                        {codeSample}
                    </div>
                </div>
            )}

            {hasBeenVisible && (
                <pre {...stylex.props(styles.pre)}>
                    <BrowserOnly>
                        {() => (
                            <React.Suspense fallback={<div>Loading...</div>}>
                                <Sandbox
                                    sampleFilename={sampleFilename}
                                    isCodeSnippet={true}
                                    codeSample={codeSample}
                                    isInViewport={isVisible}
                                />
                            </React.Suspense>
                        )}
                    </BrowserOnly>
                </pre>
            )}
        </div>
    );
}

const styles = stylex.create({
    wrapper: {
        padding: 0,
        margin: 0,
    },
    pre: {
        backgroundColor: 'var(--code-snippet-background)',
        overflow: 'visible', // Allow hover results to overflow
        position: 'relative',
        minHeight: '50px',
    },
    placeholder: {
        width: '100%',
        height: '100%',
        whiteSpace: 'pre-wrap',
        overflow: 'auto',
        padding: '16px',

        color: 'var(--code-snippet-placeholder-text-color)',
        backgroundColor: 'var(--code-snippet-placeholder-background)',
    },
    placeholderContent: {
        overflow: 'hidden',
        position: 'relative',
        fontSize: '14px',
        fontFamily:
            'SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    },
});
