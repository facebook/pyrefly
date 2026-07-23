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
import WhyPyreflyGridItem from './whyPyreflyGridItem';
import Firefly from './firefly';
import DelayedComponent from '../../utils/DelayedComponent';
import { animationDelaySeconds } from '../../utils/componentAnimationDelay';
import { log, LoggingEvent } from '../../utils/LoggingUtils';

export default function WhyPyrefly(): React.ReactElement {
    return (
        <DelayedComponent delayInSeconds={animationDelaySeconds['WhyPyrefly']}>
            {(isLoaded) => (
                <div
                    {...stylex.props(
                        styles.whyPyreflyContainer,
                        isLoaded && styles.whyPyreflyContainerVisible
                    )}
                >
                    <div {...stylex.props(styles.whyPyreflyGrid)}>
                        <WhyPyreflyGridItem
                            title="Scale with Confidence"
                            content="Type check over 1.85 million lines of code per second."
                            footnote="Tested using Meta infrastructure (166 cores, 228 GB RAM)"
                            index={0}
                            linkTo="#performance-comparison-section"
                            onClick={() =>
                                log(LoggingEvent.CLICK, {
                                    button_id:
                                        'why_pyrefly_scale_with_confidence',
                                })
                            }
                        />
                        <WhyPyreflyGridItem
                            title="Developer Delight"
                            content="Get lightning fast autocomplete, and catch errors with instant feedback in your favorite editor."
                            index={1}
                            linkTo="#ide-carousel-section"
                            onClick={() =>
                                log(LoggingEvent.CLICK, {
                                    button_id: 'why_pyrefly_developer_delight',
                                })
                            }
                        />
                        <WhyPyreflyGridItem
                            title="Verify AI-written Python"
                            content="Catch type bugs in code your AI agent writes"
                            index={2}
                            href="/blog/pyrefly-agentic-loop/"
                            ctaText="read the blog →"
                            onClick={() =>
                                log(LoggingEvent.CLICK, {
                                    button_id:
                                        'why_pyrefly_verify_ai_written_python',
                                    // Use the beacon transport so the event
                                    // survives the synchronous navigation this
                                    // card triggers.
                                    transport_type: 'beacon',
                                })
                            }
                        />
                        <WhyPyreflyGridItem
                            title="Support at your Fingertips"
                            contentWithLink={{
                                beforeText:
                                    'Have questions or feedback to share? Connect with us!',
                                link: {
                                    text: 'Join our Discord →',
                                    url: 'https://discord.gg/Cf7mFQtW7W',
                                },
                                onClick: () =>
                                    log(LoggingEvent.CLICK, {
                                        button_id: 'join_discord_homepage',
                                    }),
                            }}
                            index={3}
                        />
                    </div>
                    <section {...stylex.props(styles.fireflyContainer)}>
                        <Firefly />
                        <Firefly />
                        <Firefly />
                    </section>
                </div>
            )}
        </DelayedComponent>
    );
}

const styles = stylex.create({
    whyPyreflyContainer: {
        position: 'relative',
        paddingTop: '1.5rem',
        paddingBottom: '2rem',
        width: '100vw',
        marginLeft: 'calc(-50vw + 50%)',
        paddingHorizontal: '2rem',
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'translateY(20px)',
        transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)', // Reduced from 1s to 0.8s
    },
    whyPyreflyContainerVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    whyPyreflyGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '1.5rem',
        position: 'relative',
        zIndex: 2,
        '@media (max-width: 768px)': {
            gridTemplateColumns: 'repeat(1, 1fr)',
        },
    },
    fireflyContainer: {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 1,
        overflow: 'hidden',
    },
});
