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

export default function LandingPageHeader(): React.ReactElement {
    // Call hooks at the top level of the component
    const lightLogoUrl = useBaseUrl('img/Pyrefly-Brandmark.svg');
    const darkLogoUrl = useBaseUrl('img/Pyrefly-Brandmark-Invert.svg');

    return (
        <header {...stylex.props(styles.featureHero)}>
            <></>
            <DelayedComponent
                delayInSeconds={animationDelaySeconds['LandingPageHeader.Logo']}
            >
                {(isLoaded) => (
                    <section {...stylex.props(styles.logoContainer)}>
                        <ThemedImage
                            alt="Pyrefly Logo"
                            sources={{
                                light: lightLogoUrl,
                                dark: darkLogoUrl,
                            }}
                            {...stylex.props(
                                styles.logo,
                                isLoaded && styles.logoVisible
                            )}
                        />
                    </section>
                )}
            </DelayedComponent>

            <DelayedComponent
                delayInSeconds={
                    animationDelaySeconds['LandingPageHeader.Subtitle']
                }
            >
                {(isLoaded) => (
                    <p
                        {...stylex.props(
                            styles.subtitle,
                            typography.h3,
                            isLoaded && styles.subtitleVisible
                        )}
                    >
                        <span>
                            A fast type checker and language server for Python with powerful IDE features
                        </span>
                    </p>
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
                            styles.buttonGroupVertical,
                            isLoaded && styles.buttonGroupVerticalVisible
                        )}
                    >
                        <PipInstallPyrefly />
                        <a
                            href="https://marketplace.visualstudio.com/items?itemName=meta.pyrefly"
                            target="_blank"
                            onClick={() =>
                                log(LoggingEvent.CLICK, {
                                    button_id: 'get_vscode_extension',
                                })
                            }
                            onMouseEnter={() =>
                                log(LoggingEvent.HOVER, {
                                    button_id: 'get_vscode_extension',
                                })
                            }
                            {...stylex.props(
                                styles.buttonFullWidth,
                                typography.p
                            )}
                        >
                            {' '}
                            Get VSCode Extension{' '}
                        </a>
                    </section>
                )}
            </DelayedComponent>

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
    root: {
        '--title-font-size': '7rem',
        '--subtitle-font-size': '3.5rem',
    },
    logoContainer: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: '0.5rem',
        perspective: '1000px', // Add perspective for 3D effect
    },
    logo: {
        height: '100px',
        opacity: 0,
        filter: 'blur(10px)',
        transform: 'rotateX(30deg) scale(0.9)',
        transformOrigin: 'center center',
        transition: 'all 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    logoVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'rotateX(0deg) scale(1)',
    },

    featureHero: {
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        background: 'var(--color-background)',
        color: 'var(--color-text)',
        WebkitFontSmoothing: 'antialiased',
        marginLeft: 'auto',
        marginRight: 'auto',
        lineHeight: 1.1,
    },
    subtitle: {
        textAlign: 'center',
        marginBottom: '1rem',
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'rotateX(-20deg) translateY(20px)',
        transformOrigin: 'center center',
        transition: 'all 1.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    subtitleVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'rotateX(0deg) translateY(0)',
    },
    buttonGroupVertical: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        marginVertical: '20px',
        width: '100%',
        maxWidth: '430px',
        marginLeft: 'auto',
        marginRight: 'auto',
        opacity: 0,
        filter: 'blur(6px)',
        transform: 'rotateX(15deg) translateY(15px)',
        transformOrigin: 'center center',
        transition: 'all 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    buttonGroupVerticalVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'rotateX(0deg) translateY(0)',
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
        justifyContent: 'space-between',
        width: '100%',
        marginTop: '10px',
    },
    buttonFullWidth: {
        padding: '0.75rem 1.5rem',
        borderRadius: '4px',
        border: '1px solid var(--color-text)',
        backgroundColor: 'var(--color-primary)',
        color: 'var(--color-background)',
        fontWeight: 'bold',
        cursor: 'pointer',
        transition: 'all 0.2s',
        width: '100%',
        textAlign: 'center',
        ':hover': {
            backgroundColor: 'var(--color-primary-hover)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            transform: 'translateY(-1px)',
        },
    },
    button: {
        padding: '0.75rem 1.5rem',
        borderRadius: '4px',
        border: '1px solid var(--color-text)',
        backgroundColor: 'transparent',
        color: 'var(--color-text)',
        cursor: 'pointer',
        transition: 'all 0.2s',
        fontWeight: 'bold',
        width: 'calc(50% - 5px)', // Each button takes up half the space minus a small gap
        textAlign: 'center',
    },
    pip: {
        marginTop: '1rem',
        marginBottom: 0,
    },
});
