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
import DelayedComponent from '../../utils/DelayedComponent';
import { animationDelaySeconds } from '../../utils/componentAnimationDelay';

export default function IDECarousel(): React.ReactElement {
    const vscodeUrl = useBaseUrl('img/vscode.svg');
    const pycharmUrl = useBaseUrl('img/PyCharm.svg');
    const antigravityUrl = useBaseUrl('img/Google_Antigravity_Logo_2025.svg');
    const cursorUrl = useBaseUrl('img/cursor_logo_light.svg');
    const windsurfUrl = useBaseUrl('img/windsurf.svg');
    const neovimUrl = useBaseUrl('img/neovim.svg');

    const IDE_LOGOS = [
        { alt: 'VS Code', svg: vscodeUrl },
        { alt: 'PyCharm', svg: pycharmUrl },
        { alt: 'Google Antigravity', svg: antigravityUrl },
        { alt: 'Cursor', svg: cursorUrl },
        { alt: 'Windsurf', svg: windsurfUrl },
        { alt: 'Neovim', svg: neovimUrl },
    ];

    return (
        <DelayedComponent delayInSeconds={animationDelaySeconds['IDECarousel']}>
            {(isLoaded) => (
                <div
                    {...stylex.props(
                        styles.container,
                        isLoaded && styles.containerVisible
                    )}
                >
                    <div className="container">
                        <h2 {...stylex.props(typography.h2, styles.tagline)}>
                            Works where{' '}
                            <span {...stylex.props(styles.highlight)}>
                                you or your agent
                            </span>{' '}
                            writes Python
                        </h2>
                    </div>
                    <div {...stylex.props(styles.logoStrip)}>
                        <div {...stylex.props(styles.scrollTrack)}>
                            {[...IDE_LOGOS, ...IDE_LOGOS].map((ide, index) => (
                                <div
                                    key={index}
                                    {...stylex.props(styles.logoItem)}
                                >
                                    <img
                                        src={ide.svg}
                                        alt={ide.alt}
                                        {...stylex.props(styles.logoImage)}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </DelayedComponent>
    );
}

const scrollAnimation = stylex.keyframes({
    '0%': { transform: 'translate3d(0, 0, 0)' },
    '100%': { transform: 'translate3d(-50%, 0, 0)' },
});

const styles = stylex.create({
    container: {
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        paddingVertical: '2rem',
        width: '100vw',
        marginLeft: 'calc(-50vw + 50%)',
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'translateY(20px)',
        transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    containerVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    tagline: {
        color: 'var(--color-text)',
        marginBottom: '2rem',
    },
    highlight: {
        color: 'var(--color-primary)',
    },
    logoStrip: {
        overflow: 'hidden',
        width: '100%',
        margin: '4rem 0',
    },
    scrollTrack: {
        display: 'flex',
        alignItems: 'center',
        width: 'max-content',
        animationName: scrollAnimation,
        animationDuration: '25s',
        animationTimingFunction: 'linear',
        animationIterationCount: 'infinite',
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',
    },
    logoItem: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: '1rem',
        marginHorizontal: '2rem',
    },
    logoImage: {
        height: '80px',
        width: 'auto',
    },
});
