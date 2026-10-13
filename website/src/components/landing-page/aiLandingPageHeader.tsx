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
import { log, LoggingEvent } from '../../utils/LoggingUtils';
import DelayedComponent from '../../utils/DelayedComponent';
import { animationDelaySeconds } from '../../utils/componentAnimationDelay';

export default function AiLandingPageHeader(): React.ReactElement {
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
                                Catch type bugs in code your AI agent writes
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
                                styles.steps,
                                isLoaded && styles.stepsVisible
                            )}
                        >
                            <ol {...stylex.props(styles.stepList)}>
                                <li {...stylex.props(styles.step)}>
                                    <h2
                                        {...stylex.props(
                                            styles.stepTitle,
                                            typography.p
                                        )}
                                    >
                                        Install Pyrefly
                                    </h2>
                                    <PipInstallPyrefly />
                                    <div {...stylex.props(styles.buttonRow)}>
                                        <a
                                            href="https://marketplace.visualstudio.com/items?itemName=meta.pyrefly"
                                            target="_blank"
                                            onClick={() =>
                                                log(LoggingEvent.CLICK, {
                                                    button_id:
                                                        'get_vscode_extension',
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
                                                    button_id:
                                                        'get_openvsx_extension',
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
                                </li>

                                <li {...stylex.props(styles.step)}>
                                    <h2
                                        {...stylex.props(
                                            styles.stepTitle,
                                            typography.p
                                        )}
                                    >
                                        Tell your agent to use it
                                    </h2>
                                    <p
                                        {...stylex.props(
                                            styles.stepBody,
                                            typography.p
                                        )}
                                    >
                                        Add a skill file so your agent runs{' '}
                                        <code>pyrefly check</code> before it
                                        calls a task done, or a Stop hook if it
                                        needs more convincing.{' '}
                                        <a
                                            href="/blog/pyrefly-agentic-loop"
                                            onClick={() =>
                                                log(LoggingEvent.CLICK, {
                                                    button_id:
                                                        'ai_agentic_loop_blog',
                                                    transport_type: 'beacon',
                                                })
                                            }
                                            {...stylex.props(styles.link)}
                                        >
                                            Read the setup guide
                                        </a>
                                        .
                                    </p>
                                </li>

                                <li {...stylex.props(styles.step)}>
                                    <h2
                                        {...stylex.props(
                                            styles.stepTitle,
                                            typography.p
                                        )}
                                    >
                                        Let it fix what Pyrefly finds
                                    </h2>
                                    <p
                                        {...stylex.props(
                                            styles.stepBody,
                                            typography.p
                                        )}
                                    >
                                        Errors come back fast and specific
                                        enough to act on, so your agent can
                                        correct itself and re-check before the
                                        code ever reaches you.
                                    </p>
                                </li>
                            </ol>
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

const styles = stylex.create({
    featureHero: {
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        paddingTop: '4rem',
        paddingBottom: '4rem',
        '@media (max-width: 996px)': {
            paddingTop: '2.5rem',
            paddingBottom: '2.5rem',
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
    steps: {
        opacity: 0,
        filter: 'blur(6px)',
        transform: 'translateY(15px)',
        transition: 'all 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    stepsVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    stepList: {
        listStyle: 'none',
        counterReset: 'step',
        margin: 0,
        padding: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '1.75rem',
    },
    step: {
        counterIncrement: 'step',
        position: 'relative',
        paddingLeft: '3rem',
        // The numeral is generated rather than hard-coded so the steps can be
        // reordered or one dropped without renumbering by hand.
        '::before': {
            content: 'counter(step)',
            position: 'absolute',
            left: 0,
            // Optically centres the 2rem numeral against the 1.4rem title line
            // box rather than hanging below it.
            top: '-0.3rem',
            width: '2rem',
            height: '2rem',
            borderRadius: '50%',
            border: '1px solid var(--color-landing-page-card-border)',
            background: 'var(--color-landing-page-card-background)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
        },
    },
    stepTitle: {
        fontWeight: 'bold',
        marginBottom: '0.5rem',
        lineHeight: 1.4,
    },
    stepBody: {
        marginBottom: 0,
        lineHeight: 1.5,
        color: 'var(--color-text)',
    },
    link: {
        color: 'var(--color-primary)',
        textDecoration: 'underline',
        transition:
            'color var(--ifm-transition-fast) var(--ifm-transition-timing-default)',
        ':hover': {
            color: '#BA8E23',
            textDecoration: 'var(--ifm-link-decoration)',
        },
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
