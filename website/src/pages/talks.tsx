/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import { useBaseUrlUtils } from '@docusaurus/useBaseUrl';
import * as stylex from '@stylexjs/stylex';
import typography from '../components/landing-page/typography';
import { landingPageCardStyles } from '../components/landing-page/landingPageCardStyles';
import { talks, type TalkType } from '../data/talks';

const YOUTUBE_ID_REGEX =
    /(?:youtu\.be\/|youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|live\/|shorts\/))([\w-]{11})/;

const FILTERS: { label: string; type: TalkType | null }[] = [
    { label: 'All', type: null },
    { label: 'Talks', type: 'talk' },
    { label: 'Podcasts', type: 'podcast' },
    { label: 'Videos', type: 'video' },
];

const TYPE_BADGES: Record<TalkType, string> = {
    talk: '🎤 Talk',
    podcast: '🎧 Podcast',
    video: '▶️ Video',
};

export default function Talks(): React.ReactElement {
    const { withBaseUrl } = useBaseUrlUtils();
    const [filter, setFilter] = React.useState<TalkType | null>(null);
    // Hide filters for types that have no entries yet.
    const filters = FILTERS.filter(
        (f) => f.type == null || talks.some((t) => t.type === f.type)
    );

    return (
        <Layout
            title="Talks & Podcasts"
            description="Talks, podcast appearances, and videos from the Pyrefly team."
        >
            <main {...stylex.props(styles.page)}>
                <div className="container">
                    <h1 {...stylex.props(typography.h2)}>Talks & Podcasts</h1>
                    <p {...stylex.props(styles.intro)}>
                        Conference talks, podcast appearances, and videos from
                        the Pyrefly team.
                    </p>
                    <div {...stylex.props(styles.filters)}>
                        {filters.map((f) => (
                            <button
                                key={f.label}
                                className={
                                    filter === f.type
                                        ? 'button button--primary'
                                        : 'button button--secondary'
                                }
                                onClick={() => setFilter(f.type)}
                            >
                                {f.label}
                            </button>
                        ))}
                    </div>
                    <div {...stylex.props(styles.grid)}>
                        {talks
                            .filter((t) => filter == null || t.type === filter)
                            .map((talk) => {
                                const youtubeId =
                                    talk.url.match(YOUTUBE_ID_REGEX)?.[1];
                                const image =
                                    talk.image ??
                                    (youtubeId != null
                                        ? `https://i.ytimg.com/vi/${youtubeId}/hqdefault.jpg`
                                        : '/img/Pyrefly-Preview-Symbol.png');
                                return (
                                    <Link
                                        key={talk.url}
                                        to={talk.url}
                                        {...stylex.props(
                                            landingPageCardStyles.card,
                                            styles.card
                                        )}
                                    >
                                        <div
                                            {...stylex.props(
                                                styles.imageWrapper
                                            )}
                                        >
                                            <img
                                                src={withBaseUrl(image)}
                                                alt=""
                                                loading="lazy"
                                                {...stylex.props(styles.image)}
                                            />
                                            <span
                                                {...stylex.props(styles.badge)}
                                            >
                                                {TYPE_BADGES[talk.type]}
                                            </span>
                                        </div>
                                        <h3
                                            {...stylex.props(
                                                typography.h6,
                                                styles.title
                                            )}
                                        >
                                            {talk.title}
                                        </h3>
                                        {talk.subtitle != null && (
                                            <p
                                                {...stylex.props(
                                                    styles.subtitle
                                                )}
                                            >
                                                {talk.subtitle}
                                            </p>
                                        )}
                                    </Link>
                                );
                            })}
                    </div>
                </div>
            </main>
        </Layout>
    );
}

const styles = stylex.create({
    page: {
        paddingTop: '4rem',
        paddingBottom: '6rem',
    },
    intro: {
        fontSize: '1.2rem',
    },
    filters: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.5rem',
        marginBottom: '2rem',
    },
    grid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: '1.5rem',
    },
    card: {
        padding: '1rem',
        color: 'var(--color-text)',
        textDecoration: {
            default: 'none',
            ':hover': 'none',
        },
    },
    imageWrapper: {
        position: 'relative',
        marginBottom: '0.75rem',
    },
    image: {
        display: 'block',
        width: '100%',
        aspectRatio: '16 / 9',
        objectFit: 'cover',
        borderRadius: '6px',
    },
    badge: {
        position: 'absolute',
        top: '0.5rem',
        left: '0.5rem',
        padding: '0.2rem 0.6rem',
        borderRadius: '999px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: '#ffffff',
        fontSize: '0.8rem',
        fontWeight: 500,
    },
    title: {
        margin: 0,
    },
    subtitle: {
        margin: '0.25rem 0 0',
        fontSize: '0.9rem',
        opacity: 0.75,
    },
});
