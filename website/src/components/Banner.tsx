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

interface BannerProps {
    /** Text content for the banner */
    text?: string;
    /** Background color variant */
    variant?: 'primary' | 'secondary' | 'tertiary' | 'info';
    /** Whether to show a close button */
    dismissible?: boolean;
    /** Callback when banner is dismissed */
    onDismiss?: () => void;
    /** CTA button configuration */
    cta?: {
        text: string;
        href?: string;
        onClick?: () => void;
        external?: boolean;
    };
    /** Additional links to display */
    links?: Array<{
        text: string;
        href?: string;
        onClick?: () => void;
        external?: boolean;
    }>;
    /** Custom styles to apply */
    className?: string;
    /** Additional content to render */
    children?: React.ReactNode;
}

export default function Banner({
    text,
    variant = 'primary',
    dismissible = false,
    onDismiss,
    cta,
    links = [],
    className,
    children,
}: BannerProps): React.ReactElement {
    const [isVisible, setIsVisible] = React.useState(true);

    const handleDismiss = () => {
        setIsVisible(false);
        if (onDismiss) {
            onDismiss();
        }
    };

    const handleLinkClick = (link: typeof cta | typeof links[0]) => {
        if (link?.onClick) {
            link.onClick();
        }
        if (link?.href && link.external) {
            window.open(link.href, '_blank', 'noopener,noreferrer');
        } else if (link?.href) {
            window.location.href = link.href;
        }
    };

    if (!isVisible) {
        return <></>;
    }

    return (
        <div
            {...stylex.props(
                styles.banner,
                styles[variant],
                className && { className }
            )}
        >
            <div {...stylex.props(styles.content)}>
                {text && (
                    <span {...stylex.props(styles.text)}>{text}</span>
                )}
                {children}

                {(cta || links.length > 0) && (
                    <div {...stylex.props(styles.actions)}>
                        {cta && (
                            <button
                                {...stylex.props(styles.ctaButton)}
                                onClick={() => handleLinkClick(cta)}
                            >
                                {cta.text}
                            </button>
                        )}
                        {links.map((link, index) => (
                            <button
                                key={index}
                                {...stylex.props(styles.link)}
                                onClick={() => handleLinkClick(link)}
                            >
                                {link.text}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {dismissible && (
                <button
                    {...stylex.props(styles.dismissButton)}
                    onClick={handleDismiss}
                    aria-label="Dismiss banner"
                >
                    ✕
                </button>
            )}
        </div>
    );
}

const styles = stylex.create({
    banner: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 20px',
        minHeight: '48px',
        width: '100%',
        position: 'relative',
        transition: 'all 0.3s ease',
        fontSize: '16px',
        lineHeight: '1.4',
    },
    primary: {
        backgroundColor: 'var(--color-primary)',
        color: 'var(--color-background)',
        borderBottom: '1px solid var(--color-primary-hover)',
    },
    secondary: {
        backgroundColor: 'var(--color-background-secondary, #f5f5f5)',
        color: 'var(--color-text)',
        borderBottom: '1px solid var(--color-border, #e0e0e0)',
    },
    tertiary: {
        backgroundColor: '#fff3cd',
        color: '#856404',
        borderBottom: '1px solid #ffeaa7',

    },
    info: {
        backgroundColor: '#d1ecf1',
        color: '#0c5460',
        borderBottom: '1px solid #bee5eb',
    },
    content: {
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        flex: 1,
        flexWrap: 'wrap',
    },
    text: {
        margin: 0,
        fontWeight: '500',
    },
    actions: {
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        flexWrap: 'wrap',
    },
    ctaButton: {
        padding: '6px 16px',
        borderRadius: '4px',
        border: '1px solid currentColor',
        backgroundColor: 'transparent',
        color: 'inherit',
        fontWeight: 'bold',
        cursor: 'pointer',
        fontSize: '13px',
        transition: 'all 0.2s ease',
        ':hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            transform: 'translateY(-1px)',
        },
        ':active': {
            transform: 'translateY(0)',
        },
    },
    link: {
        background: 'none',
        border: 'none',
        color: 'inherit',
        textDecoration: 'underline',
        cursor: 'pointer',
        fontSize: '13px',
        padding: '4px 8px',
        transition: 'opacity 0.2s ease',
        ':hover': {
            opacity: 0.8,
        },
    },
    dismissButton: {
        background: 'none',
        border: 'none',
        color: 'inherit',
        cursor: 'pointer',
        fontSize: '16px',
        fontWeight: 'bold',
        padding: '4px 8px',
        marginLeft: '12px',
        borderRadius: '50%',
        width: '28px',
        height: '28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.2s ease',
        ':hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
        },
    },
});
