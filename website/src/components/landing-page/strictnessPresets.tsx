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
import useBaseUrl from '@docusaurus/useBaseUrl';
import { landingPageCardStyles } from './landingPageCardStyles';
import { log, LoggingEvent } from '../../utils/LoggingUtils';

// Ordered from least to most strict. `off` and `legacy` are deliberately
// omitted: `off` reports nothing, and `legacy` exists for mypy migrations, so
// neither illustrates the progression. All six are covered in the docs.
const PRESETS = [
    {
        name: 'basic',
        summary: 'Best for newcomers to type checking',
        detail: 'Catch only the most obvious errors and ignore everything else.',
    },
    {
        name: 'default',
        summary: 'Best for most projects, most of the time',
        detail: 'You annotate where it helps and want those annotations properly checked.',
    },
    {
        name: 'strict',
        summary: 'Best for libraries and shared code',
        detail: 'Other people depend on your type correctness, so no implicit Any and no missing @override',
    },
];

export default function StrictnessPresets(): React.ReactElement {
    const presetDocsUrl = useBaseUrl('en/docs/configuration/#preset');

    return (
        <div {...stylex.props(styles.root)}>
            <p {...stylex.props(styles.intro, typography.p)}>
                Set a preset in the <code>[tool.pyrefly]</code> section of your{' '}
                <code>pyproject.toml</code> to control how many errors you get.{' '}
                More info in{' '}
                <a
                    href={presetDocsUrl}
                    onClick={() =>
                        log(LoggingEvent.CLICK, {
                            button_id: 'preset_docs',
                            // Use the beacon transport so the event survives
                            // the synchronous navigation that follows.
                            transport_type: 'beacon',
                        })
                    }
                    {...stylex.props(styles.link)}
                >
                    the preset docs
                </a>
                .
            </p>
            <ul {...stylex.props(styles.presets)}>
                {PRESETS.map((preset) => (
                    <li key={preset.name} {...stylex.props(styles.presetItem)}>
                        <div
                            {...stylex.props(
                                landingPageCardStyles.card,
                                styles.card
                            )}
                        >
                            {/* The config is held in one string so the line
                                break survives: JSX collapses whitespace
                                between elements, and the surrounding `pre`
                                only preserves breaks that reach the DOM. */}
                            <pre {...stylex.props(styles.presetConfig)}>
                                <code>{`preset = "${preset.name}"`}</code>
                            </pre>
                            <h3
                                {...stylex.props(
                                    styles.presetSummary,
                                    typography.p
                                )}
                            >
                                {preset.summary}
                            </h3>
                            <p
                                {...stylex.props(
                                    styles.presetDetail,
                                    typography.p
                                )}
                            >
                                {preset.detail}
                            </p>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}

const styles = stylex.create({
    root: {
        // Four cards need more room than the theme's container gives them, so
        // this breaks out of it and centres against the viewport instead. The
        // 92vw cap keeps a margin on either side and, because vw includes the
        // scrollbar, cannot introduce horizontal overflow.
        // Below the breakpoint the cards are already single-column, so
        // breaking out would only misalign them with the section title and
        // leave a narrower margin than the rest of the page.
        width: {
            default: 'min(92vw, 1400px)',
            '@media (max-width: 996px)': '100%',
        },
        marginLeft: {
            default: '50%',
            '@media (max-width: 996px)': 0,
        },
        transform: {
            default: 'translateX(-50%)',
            '@media (max-width: 996px)': 'none',
        },
        // This section and the one below it contribute 1rem of padding each,
        // so 3rem here brings the gap under the cards up to the 5rem above the
        // hero logo (1rem of section padding plus the hero's own 4rem).
        paddingBottom: '3rem',
    },
    intro: {
        textAlign: 'center',
        maxWidth: '58ch',
        marginLeft: 'auto',
        marginRight: 'auto',
        marginBottom: '2rem',
        color: 'var(--color-text)',
    },
    presets: {
        listStyle: 'none',
        margin: 0,
        padding: 0,
        display: 'grid',
        // Four across on a wide screen, reflowing to two and then one rather
        // than squeezing four narrow columns onto a laptop.
        gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
        gap: '1.5rem',
    },
    presetItem: {
        display: 'flex',
    },
    card: {
        alignItems: 'flex-start',
        gap: '0.5rem',
    },
    presetConfig: {
        alignSelf: 'stretch',
        margin: 0,
        padding: '0.5rem 0.75rem',
        borderRadius: '4px',
        border: '1px solid var(--color-landing-page-card-border)',
        background: 'var(--color-background-secondary)',
        color: 'var(--color-primary)',
        fontWeight: 'bold',
        fontSize: '0.85rem',
        lineHeight: 1.6,
        // Every line of the config is short enough to fit, so preserving the
        // breaks verbatim is safe and keeps the two lines aligned.
        whiteSpace: 'pre',
        overflowX: 'auto',
    },
    presetSummary: {
        margin: 0,
        fontWeight: 'bold',
        lineHeight: 1.4,
        color: 'var(--color-text)',
    },
    presetDetail: {
        margin: 0,
        fontSize: '0.9rem',
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
});
