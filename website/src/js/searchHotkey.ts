/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * Docusaurus client module that adds a Cmd+K / Ctrl+K shortcut to focus the
 * docs search bar.
 *
 * The site uses the `docusaurus-plugin-internaldocs-fb` preset's bundled
 * (lunr-backed) search bar rather than Algolia DocSearch, so it does not get
 * Algolia's built-in Cmd+K handler for free — we wire it up ourselves. The
 * search input is rendered with the stable id `search_input_react`.
 */

import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

if (ExecutionEnvironment.canUseDOM) {
    window.addEventListener('keydown', (e: KeyboardEvent) => {
        if (
            (e.metaKey || e.ctrlKey) &&
            !e.shiftKey &&
            !e.altKey &&
            e.key.toLowerCase() === 'k'
        ) {
            const input = document.getElementById(
                'search_input_react',
            ) as HTMLInputElement | null;
            // Only intercept once the lazy-loaded lunr index has enabled the
            // input; focusing a disabled input is a no-op anyway.
            if (input != null && !input.disabled) {
                e.preventDefault();
                input.focus();
            }
        }
    });
}
