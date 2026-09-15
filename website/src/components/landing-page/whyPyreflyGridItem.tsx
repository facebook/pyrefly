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
import typography from './typography';
import { landingPageCardStyles } from './landingPageCardStyles';
import Tooltip from './Tooltip';
import DelayedComponent from '../../utils/DelayedComponent';
import { getWhyPyreflyGridItemDelay } from '../../utils/componentAnimationDelay';

interface LinkProps {
    text: string;
    url: string;
}

interface ContentWithLinkProps {
    text?: string;
    link?: LinkProps;
    beforeText?: string;
    afterText?: string;
    onClick?: () => void;
}

interface WhyPyreflyGridItemProps {
    title: string;
    content?: string;
    index: number;
    contentWithLink?: ContentWithLinkProps;
    footnote?: string;
    linkTo?: string;
    href?: string;
    ctaText?: string;
    onClick?: () => void;
}

export default function WhyPyreflyGridItem({
    title,
    content,
    contentWithLink,
    index,
    footnote,
    linkTo,
    href,
    ctaText,
    onClick,
}: WhyPyreflyGridItemProps): React.ReactElement {
    const delayInSeconds = getWhyPyreflyGridItemDelay(index);
    const isClickable = linkTo != null || href != null;
    const hasCtaLink = contentWithLink?.link != null && content == null;
    const resolvedHref = useBaseUrl(href ?? '');

    const handleClick = () => {
        onClick?.();
        if (linkTo != null) {
            document
                .querySelector(linkTo)
                ?.scrollIntoView({ behavior: 'smooth' });
        } else if (href != null) {
            window.location.href = resolvedHref;
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        // Activate the card on Enter or Space, matching native link/button behavior
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick();
        }
    };

    const handleCtaLinkClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        contentWithLink?.onClick?.();
    };

    return (
        <DelayedComponent delayInSeconds={delayInSeconds}>
            {(isVisible) => (
                <div
                    {...stylex.props(
                        landingPageCardStyles.card,
                        styles.hidden,
                        isVisible && styles.visible,
                        isClickable && styles.clickable
                    )}
                    style={{
                        transitionDelay: `${index * 0.05}s`,
                    }}
                    onClick={isClickable ? handleClick : undefined}
                    onKeyDown={isClickable ? handleKeyDown : undefined}
                    tabIndex={isClickable ? 0 : undefined}
                    role={isClickable ? 'link' : undefined}
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
                        {contentWithLink && content != null && (
                            <>
                                {contentWithLink.beforeText}
                                <a
                                    href={contentWithLink.link.url}
                                    target="_blank"
                                    onClick={contentWithLink.onClick}
                                    {...stylex.props(styles.link)}
                                >
                                    {contentWithLink.link.text}
                                </a>
                                {contentWithLink.afterText}
                            </>
                        )}
                        {contentWithLink && content == null && (
                            <>{contentWithLink.beforeText}</>
                        )}
                    </p>
                    {isClickable && (
                        <span {...stylex.props(styles.cta)}>
                            {ctaText ?? '→'}
                        </span>
                    )}
                    {hasCtaLink && (
                        <a
                            href={contentWithLink.link.url}
                            target="_blank"
                            onClick={handleCtaLinkClick}
                            {...stylex.props(styles.cta)}
                        >
                            {contentWithLink.link.text}
                        </a>
                    )}
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
    clickable: {
        cursor: 'pointer',
    },
    cta: {
        color: 'var(--color-primary)',
        fontSize: '0.95rem',
        fontWeight: 600,
        marginTop: '0.5rem',
        transition: 'transform 0.2s ease, color 0.2s ease',
        display: 'inline-block',
    },
    link: {
        color: 'var(--color-primary)',
        ':hover': {
            color: 'var(--color-primary-hover)',
        },
    },
});
