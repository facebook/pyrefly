/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import LiteYouTubeEmbed from 'react-lite-youtube-embed';
import 'react-lite-youtube-embed/dist/LiteYouTubeEmbed.css';
import * as stylex from '@stylexjs/stylex';
import { landingPageCardStyles } from './landingPageCardStyles';
import DelayedComponent from '../../utils/DelayedComponent';

export default function PyreflyVideo(): React.ReactElement {
    return (
        <DelayedComponent delayInSeconds={1.4}>
            {(isLoaded) => (
                <div
                    {...stylex.props(
                        landingPageCardStyles.card,
                        styles.body,
                        isLoaded && styles.bodyVisible
                    )}
                >
                    <LiteYouTubeEmbed
                        id="LXaFRKrTJVU"
                        params="autoplay=1&autohide=1&showinfo=0&rel=0"
                        title="Introducing Pyrefly: A new type checker and IDE experience for Python"
                        poster="maxresdefault"
                    />
                </div>
            )}
        </DelayedComponent>
    );
}

const styles = stylex.create({
    header: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    body: {
        maxWidth: '800px',
        margin: '0 auto',
        ':hover': {
            background: 'var(--color-landing-page-card-background)',
        },
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'translateY(20px)',
        transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    bodyVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
});
