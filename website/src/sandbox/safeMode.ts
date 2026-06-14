/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

export const SANDBOX_SAFE_MODE_SESSION_STORAGE_KEY =
    'pyrefly-sandbox-safe-mode';

export function isSandboxSafeModeEnabled(): boolean {
    if (typeof window === 'undefined') {
        return false;
    }
    return (
        window.sessionStorage.getItem(SANDBOX_SAFE_MODE_SESSION_STORAGE_KEY) ===
        '1'
    );
}

export function enableSandboxSafeMode(): void {
    window.sessionStorage.setItem(SANDBOX_SAFE_MODE_SESSION_STORAGE_KEY, '1');
}

export function disableSandboxSafeMode(): void {
    window.sessionStorage.removeItem(SANDBOX_SAFE_MODE_SESSION_STORAGE_KEY);
}
