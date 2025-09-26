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
        items: ['typing-for-python-developers', 'python-typing-for-beginners', 'python-features-and-peps']
    },
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
        description: "Never used a type system before or just new to Pyrefly? Start here!",
        collapsed: false,
        items: ['migrating-to-pyrefly', 'migrating-from-mypy',  'migrating-from-pyright']
    },
    {
        type: 'category' as const,
        label: 'Language Server',
        description: "Learn how to use Pyrefly in your IDE",
        collapsed: false,
        items: ['IDE', 'IDE-features']
    },
    {
        type: 'doc' as const,
        id: 'error-suppressions',
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
    {
        type: 'doc' as const,
        id: 'pydantic',
        label: 'Pydantic Support',
    },
    {
        type: 'doc' as const,
        id: 'autotype',
        label: 'Automating Type Annotations',
    },
    {
        type: 'doc' as const,
        id: 'pyrefly-faq',
        label: 'FAQ',
    },
];

const sidebars: SidebarsConfig = {
    docsSidebar,
};

export { docsSidebar };
export default sidebars;
