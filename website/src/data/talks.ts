/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Talks, podcast appearances, and videos shown on the /talks page, in the
 * order they appear. Keep the list newest first by adding new entries at the
 * top.
 *
 * Each card shows `image`, `title`, and `subtitle`, and links to `url`.
 * YouTube links use the video's thumbnail automatically, so `image` is only
 * needed for other links such as podcast episodes. Without one, the card shows
 * the Pyrefly logo.
 */

export type TalkType = 'talk' | 'podcast' | 'video';

export interface Talk {
    title: string;
    type: TalkType;
    url: string;
    /** The conference, podcast, or channel name, shown under the title. */
    subtitle?: string;
    /** A full image URL, such as the og:image of the linked page. */
    image?: string;
}

export const talks: Talk[] = [
    {
        title: 'How to get your agent to use Pyrefly for type checking',
        type: 'video',
        url: 'https://www.youtube.com/watch?v=YHJ6tUPnKhY',
        subtitle: 'Meta Open Source YouTube',
    },
    {
        title: 'Tensor Shapes in the Type System',
        type: 'talk',
        url: 'https://www.youtube.com/watch?v=HE5EyQW_7eY',
        subtitle: 'PyCon US 2026 Typing Summit',
    },
    {
        title: 'Type Checking in Agentic Workflows',
        type: 'talk',
        url: 'https://www.youtube.com/watch?v=xNaKm4fTFtw',
        subtitle: 'PyCon US 2026 Typing Summit',
    },
    {
        title: 'Pyrefly v1.0.0 is here!',
        type: 'video',
        url: 'https://www.youtube.com/watch?v=_o0TZG_xrys',
        subtitle: 'Meta Open Source YouTube',
    },
    {
        title: 'Pydantic support in Pyrefly',
        type: 'video',
        url: 'https://www.youtube.com/watch?v=zXYpSQB57YI',
        subtitle: 'Meta Open Source YouTube',
    },
    {
        title: 'Announcing Pyrefly Beta',
        type: 'video',
        url: 'https://www.youtube.com/watch?v=4o0RLJJ-FAo',
        subtitle: 'Meta Open Source YouTube',
    },
    {
        title: 'Pyrefly: A Scalable Type Checker for a Unified IDE Experience',
        type: 'talk',
        url: 'https://www.youtube.com/watch?v=hlJlzEbSYZg',
        subtitle: 'PyCon UK 2025',
    },
    {
        title: 'Pyrefly: Fast, IDE-friendly typing for Python',
        type: 'podcast',
        url: 'https://www.youtube.com/watch?v=P4RKxl_giH4',
        subtitle: 'Talk Python To Me',
    },
    {
        title: 'More Python Type Checking! Pyrefly with Aaron Pollack & Steven Troxler',
        type: 'podcast',
        url: 'https://www.youtube.com/watch?v=huHF0Rv8L14',
        subtitle: 'Happy Path Programming',
    },
    {
        title: 'Introducing Pyrefly: A new type checker and IDE experience for Python',
        type: 'video',
        url: 'https://www.youtube.com/watch?v=LXaFRKrTJVU',
        subtitle: 'Meta Open Source YouTube',
    },
    {
        title: 'High-Performance Python: Faster Type Checking and Free Threaded Execution',
        type: 'talk',
        url: 'https://www.youtube.com/watch?v=ZTSZ1OCUaeQ',
        subtitle: 'PyCon US 2025',
    },
    {
        title: 'Introducing Pyrefly',
        type: 'talk',
        url: 'https://www.youtube.com/watch?v=7uixlNTOY4s&t=6355s',
        subtitle: 'PyCon US 2025 Typing Summit',
    },
    {
        title: 'Open-sourcing Pyrefly: A faster Python type checker written in Rust',
        type: 'podcast',
        url: 'https://engineering.fb.com/2025/05/15/developer-tools/open-sourcing-pyrefly-a-faster-python-type-checker-written-in-rust/',
        subtitle: 'Meta Tech Podcast',
        image: 'https://engineering.fb.com/wp-content/uploads/2025/05/Meta-Tech-Podcast-episode-75-Pyrefly.png',
    },
];
