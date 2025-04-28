/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/*
 * We should explicitly list all the labels below, so that src/js/docs-categories.js that generates
 * the /en/docs page will automatically work.
 *
 * For categories, the first item must be the index page.
 */

import { fbInternalOnly } from 'docusaurus-plugin-internaldocs-fb/internal';
import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const docsSidebar = [
    {
        type: 'doc' as const,
        id: 'index',
        label: 'Introduction',
    },
    {
        type: 'category' as const,
        label: 'Learn Python Typing',
        description: "Never used a type system before or just new to Pyrefly? Start here!",
        items: ['python-typing-5-minutes', 'python-typing-for-beginners']
    },
    // TODO (T221099224): Release getting started and installation doc pages to public
    {
        type: 'category' as const,
        label: 'Getting Started',
        description: "Never used a type system before or just new to Pyrefly? Start here!",
        collapsed: false,
        items: ['installation', 'configuration'],
    },
    {
        type: 'category' as const,
        label: 'Migrating to Pyrefly',
        link: {
            type: 'doc' as const,
            id: 'migrating-to-pyrefly',
        },
        description: "Never used a type system before or just new to Pyrefly? Start here!",
        items: ['migrating-from-mypy',  'migrating-from-pyright']
    },
    {
        type: 'doc' as const,
        id: 'pyrefly-faq',
        label: 'FAQ',
    },
    {
        type: 'doc' as const,
        id: 'IDE',
        label: 'IDE',
    },
    {
        type: 'doc' as const,
        id: 'error-suppresions',
        label: 'Error Suppressions',
    },
    {
        type: 'doc' as const,
        id: 'error-kinds',
        label: 'Error Kinds',
    },
    {
        type: 'doc' as const,
        id: 'import-resolution',
        label: 'Import Resolution',
    },
];

const sidebars: SidebarsConfig = {
    docsSidebar,
};

export { docsSidebar };
export default sidebars;
