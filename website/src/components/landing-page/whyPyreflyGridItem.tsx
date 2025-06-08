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
import { landingPageCardStyles } from './landingPageCardStyles';
import Tooltip from './Tooltip';
import DelayedComponent from '../../utils/DelayedComponent';

interface LinkProps {
    text: string;
    url: string;
}

interface ContentWithLinkProps {
    text?: string;
    link?: LinkProps;
    beforeText?: string;
    afterText?: string;
}

interface WhyPyreflyGridItemProps {
    title: string;
    content?: string;
    index: number;
    contentWithLink?: ContentWithLinkProps;
    footnote?: string;
}

export default function WhyPyreflyGridItem({
    title,
    content,
    contentWithLink,
    index,
    footnote,
}: WhyPyreflyGridItemProps): React.ReactElement {
    // Calculate delay in seconds (100ms per index, 0.5 is the time it takes for the Why Pyrefly section to load)
    const delayInSeconds = index * 0.1 + 0.5;

    return (
        <DelayedComponent delayInSeconds={delayInSeconds}>
            {(isVisible) => (
                <div
                    {...stylex.props(
                        landingPageCardStyles.card,
                        styles.hidden,
                        isVisible && styles.visible
                    )}
                    style={{
                        // Apply dynamic delay based on index
                        transitionDelay: `${index * 0.05}s`, // Reduced from 0.1s to 0.05s
                    }}
                >
                    <h3 {...stylex.props(typography.h5, styles.cardTitle)}>
                        {title}
                    </h3>
                    <p {...stylex.props(typography.p, styles.contentText)}>
                        {content && (
                            <>
                                {footnote ? (
                                    <>
                                        {/* Split content to keep last part with footnote */}
                                        {content
                                            .split(' ')
                                            .slice(0, -2)
                                            .join(' ')}{' '}
                                        <span
                                            {...stylex.props(
                                                styles.inlineContent
                                            )}
                                        >
                                            {content
                                                .split(' ')
                                                .slice(-2)
                                                .join(' ')}
                                            <sup
                                                {...stylex.props(
                                                    styles.footnoteSupElement
                                                )}
                                            >
                                                <Tooltip content={footnote} />
                                            </sup>
                                        </span>
                                    </>
                                ) : (
                                    content
                                )}
                            </>
                        )}
                        {contentWithLink && (
                            <>
                                {contentWithLink.beforeText}
                                <a
                                    href={contentWithLink.link.url}
                                    target="_blank"
                                    {...stylex.props(styles.link)}
                                >
                                    {contentWithLink.link.text}
                                </a>
                                {contentWithLink.afterText}
                            </>
                        )}
                    </p>
                </div>
            )}
        </DelayedComponent>
    );
}

const styles = stylex.create({
    // Animation styles
    hidden: {
        transform: 'rotateX(15deg) translateY(20px)',
        opacity: 0,
        filter: 'blur(6px)',
    },
    visible: {
        opacity: 1,
        transform: 'rotateX(0deg) translateY(0)',
        filter: 'blur(0px)',
    },
    // Text styles
    cardTitle: {
        fontWeight: 700,
        marginBottom: '0.75rem',
        color: 'var(--color-text)',
    },
    contentText: {
        fontSize: '1rem',
        lineHeight: '1.6',
        marginBottom: '0rem',
        flex: 1,
        color: 'var(--color-text)',
    },
    inlineContent: {
        whiteSpace: 'nowrap',
        display: 'inline-block',
    },
    // Add a new style for the sup element
    footnoteSupElement: {
        marginLeft: '-2px',
    },
    // Style for links
    link: {
        color: 'var(--color-primary)',
        ':hover': {
            color: 'var(--color-primary-hover)',
        },
    },
});
