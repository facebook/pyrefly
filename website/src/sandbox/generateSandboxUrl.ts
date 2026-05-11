/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as LZString from 'lz-string';

export interface SandboxProject {
    files: Record<string, string>;
    activeFile: string;
}

/**
 * Generate a pyrefly.org/sandbox URL from a set of files.
 *
 * The sandbox URL encodes the full project state (all files + which file is
 * active) as an lz-string–compressed JSON blob in the `project` query param.
 */
export function generateSandboxUrl(
    files: Record<string, string>,
    activeFile: string = 'sandbox.py',
): string {
    const project: SandboxProject = { files, activeFile };
    const compressed = LZString.compressToEncodedURIComponent(
        JSON.stringify(project),
    );
    return `https://pyrefly.org/sandbox/?project=${compressed}`;
}

/**
 * Decode a pyrefly.org/sandbox URL back into its project state.
 */
export function decodeSandboxUrl(url: string): SandboxProject | null {
    const match = url.match(/[?&]project=([^&]+)/);
    if (!match) {
        return null;
    }
    const decompressed = LZString.decompressFromEncodedURIComponent(match[1]);
    if (!decompressed) {
        return null;
    }
    return JSON.parse(decompressed) as SandboxProject;
}
