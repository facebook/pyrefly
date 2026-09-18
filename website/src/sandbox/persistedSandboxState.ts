/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

export const SANDBOX_LOCAL_STORAGE_KEY = 'pyrefly-sandbox';

export function resetPersistedSandboxState(): void {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.removeItem(SANDBOX_LOCAL_STORAGE_KEY);
    } catch {
        // localStorage may be unavailable; clearing the URL still resets shared state.
    }

    const params = new URLSearchParams(window.location.search);
    params.delete('project');
    params.delete('code');

    const query = params.toString();
    const newURL = `${window.location.pathname}${query ? `?${query}` : ''}${
        window.location.hash
    }`;
    if (newURL !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
        window.history.replaceState({}, '', newURL);
    }
}
