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
import typography from './typography';
import DelayedComponent from '../../utils/DelayedComponent';
interface QuoteCardProps {
    quote: string;
    author: string;
    project: string;
}

export default function QuoteCard({
    quote,
    author,
    project,
}: QuoteCardProps): React.ReactElement {
    return (
        <DelayedComponent delayInSeconds={0}>
            {(isLoaded) => (
                <div
                    {...stylex.props(
                        styles.quoteCard,
                        isLoaded && styles.quoteCardVisible
                    )}
                >
                    <p {...stylex.props(styles.quoteText, typography.p)}>
                        {quote}
                    </p>
                    <div {...stylex.props(styles.quoteAuthor, typography.p)}>
                        <strong {...stylex.props(styles.authorName)}>
                            {author}
                        </strong>
                        <span {...stylex.props(styles.projectName)}>
                            {project}
                        </span>
                    </div>
                </div>
            )}
        </DelayedComponent>
    );
}

const styles = stylex.create({
    quoteCard: {
        padding: '2.25rem',
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow:
            '0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
        transform: 'translateY(20px)',
        opacity: 0,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        ':hover': {
            transform: 'translateY(-5px)',
            boxShadow:
                '0 10px 20px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.1)',
            background: 'rgba(255, 255, 255, 0.08)',
        },
    },
    quoteCardVisible: {
        opacity: 1,
        transform: 'translateY(0)',
    },
    quoteText: {
        lineHeight: '1.6',
        marginBottom: '1.5rem',
        fontStyle: 'italic',
        fontSize: '1rem',
        flex: 1,
        color: 'var(--color-text)',
    },
    quoteAuthor: {
        display: 'flex',
        flexDirection: 'column',
        gap: '0.25rem',
    },
    authorName: {
        fontWeight: 700,
        color: 'var(--color-primary)',
    },
    projectName: {
        color: 'var(--color-text)',
        opacity: 0.7,
    },
});
