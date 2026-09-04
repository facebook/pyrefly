/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as docusaurusTheme from '@docusaurus/theme-common';
import { Highlight, type Token } from 'prism-react-renderer';
import * as React from 'react';

import styles from './InlayCodeBlock.module.css';

export interface InlayHint {
    /** One-based line number. */
    line: number;
    /** Source text after which the hint should be displayed. */
    after: string;
    label: string;
}

interface ResolvedInlayHint extends InlayHint {
    column: number;
}

interface InlayCodeBlockProps {
    code: string;
    hints?: readonly InlayHint[];
}

export default function InlayCodeBlock({
    code,
    hints = [],
}: InlayCodeBlockProps): React.ReactElement {
    const theme = docusaurusTheme.usePrismTheme();
    const sourceLines = code.split('\n');
    const hintsByLine = new Map<number, ResolvedInlayHint[]>();

    for (const hint of hints) {
        const sourceLine = sourceLines[hint.line - 1];
        const afterIndex = sourceLine?.indexOf(hint.after) ?? -1;
        if (afterIndex < 0) {
            throw new Error(
                `Could not find ${JSON.stringify(hint.after)} on line ${hint.line} of code block`
            );
        }
        const lineHints = hintsByLine.get(hint.line) ?? [];
        lineHints.push({
            ...hint,
            column: afterIndex + hint.after.length,
        });
        hintsByLine.set(hint.line, lineHints);
    }
    for (const lineHints of hintsByLine.values()) {
        lineHints.sort((left, right) => left.column - right.column);
    }

    return (
        <Highlight theme={theme} code={code} language="python">
            {({ className, style, tokens, getLineProps, getTokenProps }) => (
                <div className={`theme-code-block ${styles.codeBlock}`}>
                    <pre className={className} style={style} tabIndex={0}>
                        <code>
                            {tokens.map((line, lineIndex) => {
                                const lineHints =
                                    hintsByLine.get(lineIndex + 1) ?? [];
                                let column = 0;
                                let hintIndex = 0;
                                const children: React.ReactNode[] = [];

                                const addHintsAtColumn = () => {
                                    while (
                                        hintIndex < lineHints.length &&
                                        lineHints[hintIndex].column === column
                                    ) {
                                        const hint = lineHints[hintIndex];
                                        children.push(
                                            <span
                                                aria-label={`Inferred type ${hint.label.replace(/^:\s*/, '')}`}
                                                className={styles.inlayHint}
                                                key={`hint-${hintIndex}`}
                                            >
                                                {hint.label}
                                            </span>
                                        );
                                        hintIndex++;
                                    }
                                };

                                addHintsAtColumn();
                                line.forEach(
                                    (token: Token, tokenIndex: number) => {
                                        const tokenStart = column;
                                        const tokenEnd =
                                            tokenStart + token.content.length;
                                        let tokenOffset = 0;
                                        while (
                                            hintIndex < lineHints.length &&
                                            lineHints[hintIndex].column <=
                                                tokenEnd
                                        ) {
                                            const splitAt =
                                                lineHints[hintIndex].column -
                                                tokenStart;
                                            if (splitAt > tokenOffset) {
                                                children.push(
                                                    <span
                                                        {...getTokenProps({
                                                            token: {
                                                                ...token,
                                                                content:
                                                                    token.content.slice(
                                                                        tokenOffset,
                                                                        splitAt
                                                                    ),
                                                            },
                                                        })}
                                                        key={`token-${tokenIndex}-${tokenOffset}`}
                                                    />
                                                );
                                            }
                                            tokenOffset = splitAt;
                                            column =
                                                lineHints[hintIndex].column;
                                            addHintsAtColumn();
                                        }
                                        if (
                                            tokenOffset < token.content.length
                                        ) {
                                            children.push(
                                                <span
                                                    {...getTokenProps({
                                                        token: {
                                                            ...token,
                                                            content:
                                                                token.content.slice(
                                                                    tokenOffset
                                                                ),
                                                        },
                                                    })}
                                                    key={`token-${tokenIndex}-${tokenOffset}`}
                                                />
                                            );
                                        }
                                        column = tokenEnd;
                                    }
                                );
                                addHintsAtColumn();

                                return (
                                    <span
                                        {...getLineProps({ line })}
                                        className={styles.line}
                                        key={lineIndex}
                                    >
                                        {children}
                                        {'\n'}
                                    </span>
                                );
                            })}
                        </code>
                    </pre>
                </div>
            )}
        </Highlight>
    );
}
