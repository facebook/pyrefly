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
import Firefly from './firefly';
import typography from './typography';
import useBaseUrl from '@docusaurus/useBaseUrl';
import PipInstallPyrefly from './pipInstallPyrefly';
import ThemedImage from '@theme/ThemedImage';
import { landingPageCardStyles } from './landingPageCardStyles';
import { log, LoggingEvent } from '../../utils/LoggingUtils';
import DelayedComponent from '../../utils/DelayedComponent';
import { animationDelaySeconds } from '../../utils/componentAnimationDelay';

export default function PresetLandingPageHeader(): React.ReactElement {
    const lightLogoUrl = useBaseUrl('img/Pyrefly-Brandmark.svg');
    const darkLogoUrl = useBaseUrl('img/Pyrefly-Brandmark-Invert.svg');

    return (
        <header {...stylex.props(styles.featureHero)}>
            <div {...stylex.props(styles.columns)}>
                <DelayedComponent
                    delayInSeconds={
                        animationDelaySeconds['LandingPageHeader.Logo']
                    }
                >
                    {(isLoaded) => (
                        <section
                            {...stylex.props(
                                styles.leftColumn,
                                isLoaded && styles.leftColumnVisible
                            )}
                        >
                            <ThemedImage
                                alt="Pyrefly Logo"
                                sources={{
                                    light: lightLogoUrl,
                                    dark: darkLogoUrl,
                                }}
                                {...stylex.props(styles.logo)}
                            />
                            <h1
                                {...stylex.props(
                                    styles.headline,
                                    typography.h3
                                )}
                            >
                                Catch bugs before you run your code
                            </h1>
                        </section>
                    )}
                </DelayedComponent>

                <DelayedComponent
                    delayInSeconds={
                        animationDelaySeconds['LandingPageHeader.ButtonGroup']
                    }
                >
                    {(isLoaded) => (
                        <section
                            {...stylex.props(
                                styles.rightColumn,
                                isLoaded && styles.rightColumnVisible
                            )}
                        >
                            {/* A static rendering of a real diagnostic rather
                                than a screenshot, so it stays legible at any
                                width and follows the active colour theme. */}
                            <div
                                {...stylex.props(
                                    landingPageCardStyles.card,
                                    styles.demo
                                )}
                                // `aria-label` is only honoured on a generic
                                // element once it has a role, and `img`
                                // presents the snippet and its diagnostic as
                                // the single picture they are meant to be.
                                role="img"
                                aria-label="Pyrefly reporting that the json module has no attribute lods, suggesting loads"
                            >
                                <pre {...stylex.props(styles.demoCode)}>
                                    <code>
                                        <span {...stylex.props(styles.keyword)}>
                                            import
                                        </span>{' '}
                                        json{'\n\n'}
                                        data = json.
                                        <span
                                            {...stylex.props(styles.squiggle)}
                                        >
                                            lods
                                        </span>
                                        (response)
                                    </code>
                                </pre>
                                <p {...stylex.props(styles.diagnostic)}>
                                    <span {...stylex.props(styles.errorMark)}>
                                        ✕
                                    </span>{' '}
                                    Module `json` has no attribute `lods`. Did
                                    you mean `loads`?
                                </p>
                            </div>

                            <PipInstallPyrefly />
                            <div {...stylex.props(styles.buttonRow)}>
                                <a
                                    href="https://marketplace.visualstudio.com/items?itemName=meta.pyrefly"
                                    target="_blank"
                                    onClick={() =>
                                        log(LoggingEvent.CLICK, {
                                            button_id: 'get_vscode_extension',
                                        })
                                    }
                                    {...stylex.props(
                                        styles.primaryButton,
                                        typography.p
                                    )}
                                >
                                    Get VSCode Extension
                                </a>
                                <a
                                    href="https://open-vsx.org/extension/meta/pyrefly"
                                    target="_blank"
                                    onClick={() =>
                                        log(LoggingEvent.CLICK, {
                                            button_id: 'get_openvsx_extension',
                                        })
                                    }
                                    {...stylex.props(
                                        styles.secondaryButton,
                                        typography.p
                                    )}
                                >
                                    Get OpenVSX Extension
                                </a>
                            </div>
                        </section>
                    )}
                </DelayedComponent>
            </div>

            <section>
                <Firefly />
                <Firefly />
                <Firefly />
                <Firefly />
                <Firefly />
            </section>
        </header>
    );
}

// The theme defines no error colour. This red clears the 3:1 WCAG AA
// non-text contrast minimum against both the light and dark backgrounds, so
// one value serves both themes. It is used only for the squiggle and the ✕
// glyph; the diagnostic text itself stays on `--color-text`.
const errorRed = '#e5534b';

const styles = stylex.create({
    featureHero: {
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        paddingTop: '4rem',
        // The space below the hero is not this padding alone: the enclosing
        // section adds 1rem, the next section another 1rem, and its title a
        // 2rem top margin. Keeping this at 1rem makes the total below the hero
        // match the 5rem above it.
        paddingBottom: '1rem',
        '@media (max-width: 996px)': {
            paddingTop: '2.5rem',
            paddingBottom: 0,
        },
        background: 'var(--color-background)',
        color: 'var(--color-text)',
        WebkitFontSmoothing: 'antialiased',
        marginLeft: 'auto',
        marginRight: 'auto',
        lineHeight: 1.1,
    },
    columns: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '3rem',
        alignItems: 'center',
        '@media (max-width: 996px)': {
            gridTemplateColumns: '1fr',
            gap: '2rem',
        },
    },
    leftColumn: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'translateY(20px)',
        transition: 'all 1.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    leftColumnVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    logo: {
        height: '130px',
        marginBottom: '1.75rem',
        '@media (max-width: 996px)': {
            height: '100px',
        },
    },
    headline: {
        // Caps the line length so the heading breaks deliberately instead of
        // running the full width of the column.
        maxWidth: '20ch',
        marginBottom: 0,
        lineHeight: 1.2,
        color: 'var(--color-text)',
    },
    rightColumn: {
        display: 'flex',
        flexDirection: 'column',
        opacity: 0,
        filter: 'blur(6px)',
        transform: 'translateY(15px)',
        transition: 'all 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    rightColumnVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    demo: {
        padding: '1.25rem',
        gap: '0.75rem',
        // The pip install block carries its own 10px top margin, so this sets
        // the gap below the card to a full line of separation.
        marginBottom: '1.25rem',
    },
    demoCode: {
        margin: 0,
        padding: 0,
        background: 'transparent',
        fontSize: '0.95rem',
        lineHeight: 1.6,
        color: 'var(--color-text)',
        overflowX: 'auto',
    },
    keyword: {
        color: 'var(--color-primary)',
    },
    squiggle: {
        textDecorationLine: 'underline',
        textDecorationStyle: 'wavy',
        textDecorationColor: errorRed,
        textUnderlineOffset: '3px',
    },
    diagnostic: {
        margin: 0,
        fontSize: '0.85rem',
        lineHeight: 1.5,
        color: 'var(--color-text)',
    },
    errorMark: {
        color: errorRed,
        fontWeight: 'bold',
    },
    buttonRow: {
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: '10px',
        width: '100%',
        marginTop: '10px',
    },
    primaryButton: {
        // Sized to content so both labels stay on one line; they still share
        // the row evenly and wrap to their own lines if the column gets narrow.
        flex: '1 1 auto',
        whiteSpace: 'nowrap',
        padding: '0.6rem 1rem',
        borderRadius: '4px',
        border: '1px solid var(--color-text)',
        backgroundColor: 'var(--color-primary)',
        color: 'var(--color-background)',
        fontWeight: 'bold',
        cursor: 'pointer',
        transition: 'all 0.2s',
        textAlign: 'center',
        ':hover': {
            backgroundColor: 'var(--color-primary-hover)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            transform: 'translateY(-1px)',
        },
    },
    secondaryButton: {
        // Sized to content so both labels stay on one line; they still share
        // the row evenly and wrap to their own lines if the column gets narrow.
        flex: '1 1 auto',
        whiteSpace: 'nowrap',
        padding: '0.6rem 1rem',
        borderRadius: '4px',
        border: '1px solid var(--color-text)',
        backgroundColor: 'transparent',
        color: 'var(--color-text)',
        fontWeight: 'bold',
        cursor: 'pointer',
        transition: 'all 0.2s',
        textAlign: 'center',
        ':hover': {
            backgroundColor:
                'var(--color-landing-page-card-background-hovered)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            transform: 'translateY(-1px)',
        },
    },
});
