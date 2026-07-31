/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as LZString from 'lz-string';
import { deflateSync, inflateSync, strFromU8, strToU8 } from 'fflate';

export interface SandboxProject {
    files: Record<string, string>;
    activeFile: string;
}

const DEFLATE_PREFIX = 'v2.';

// A preset dictionary makes short snippets benefit from patterns that DEFLATE
// cannot learn from the snippet itself. It is serialized like a real project
// so JSON escaping, Python syntax, and common sandbox configuration all match.
// This is part of the v2 wire format: retain it when introducing a new version
// so links generated with this dictionary remain decodable.
const PYTHON_DICTIONARY = strToU8(
    JSON.stringify({
        files: {
            'sandbox.py': [
                'from typing import Any, Callable, ClassVar, Final, Generic, Iterable, Iterator, Literal, Mapping, Never, Optional, Protocol, Self, Sequence, TypeAlias, TypeVar, TypedDict, Union, assert_never, assert_type, cast, overload, reveal_type',
                'from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence',
                'from dataclasses import dataclass, field',
                'import asyncio\nimport collections\nimport functools\nimport os\nimport pathlib\nimport sys',
                '@dataclass\nclass Data:\n    value: str\n    items: list[str]',
                'class Example:\n    def __init__(self, value: object) -> None:\n        self.value = value\n\n    def method(self, arg: object) -> object:\n        return arg',
                'def function(value: object | None = None) -> object:\n    if value is None:\n        return None\n    elif isinstance(value, str):\n        return value\n    else:\n        raise TypeError',
                'async def main() -> None:\n    await asyncio.sleep(0)\n\nif __name__ == "__main__":\n    main()',
                'for item in items:\n    print(item)\nwhile True:\n    break\ntry:\n    pass\nexcept Exception as error:\n    raise\nfinally:\n    pass',
            ].join('\n\n'),
            'pyrefly.toml':
                'python-version = "3.12"\n[errors]\nunimported-directive = "ignore"\nnot-required-key-access = "warn"\n',
            'main.py': '',
            '__init__.py': '',
            'test.py': '',
        },
        activeFile: 'sandbox.py',
    })
);

function bytesToBase64Url(bytes: Uint8Array): string {
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary)
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}

function base64UrlToBytes(encoded: string): Uint8Array {
    const base64 =
        encoded.replace(/-/g, '+').replace(/_/g, '/') +
        '='.repeat((4 - (encoded.length % 4)) % 4);
    return Uint8Array.from(atob(base64), (character) =>
        character.charCodeAt(0)
    );
}

function isSandboxProject(project: unknown): project is SandboxProject {
    if (
        typeof project !== 'object' ||
        project === null ||
        !('files' in project) ||
        !('activeFile' in project)
    ) {
        return false;
    }
    const { files, activeFile } = project;
    return (
        typeof activeFile === 'string' &&
        typeof files === 'object' &&
        files !== null &&
        !Array.isArray(files) &&
        Object.values(files).every((content) => typeof content === 'string')
    );
}

/** Encode a project using raw DEFLATE and URL-safe base64. */
export function encodeSandboxProject(project: SandboxProject): string {
    const compressed = deflateSync(strToU8(JSON.stringify(project)), {
        level: 9,
        dictionary: PYTHON_DICTIONARY,
    });
    return DEFLATE_PREFIX + bytesToBase64Url(compressed);
}

/** Decode current DEFLATE links and legacy LZString links. */
export function decodeSandboxProject(encoded: string): SandboxProject | null {
    try {
        let serialized: string | null;
        if (encoded.startsWith(DEFLATE_PREFIX)) {
            serialized = strFromU8(
                inflateSync(
                    base64UrlToBytes(encoded.slice(DEFLATE_PREFIX.length)),
                    { dictionary: PYTHON_DICTIONARY }
                )
            );
        } else {
            serialized = LZString.decompressFromEncodedURIComponent(encoded);
        }
        if (!serialized) {
            return null;
        }
        const project: unknown = JSON.parse(serialized);
        return isSandboxProject(project) ? project : null;
    } catch {
        return null;
    }
}

/**
 * Generate a pyrefly.org/sandbox URL from a set of files.
 *
 * The sandbox URL encodes the full project state (all files + which file is
 * active) as a compressed JSON blob in the `project` query param.
 */
export function generateSandboxUrl(
    files: Record<string, string>,
    activeFile: string = 'sandbox.py'
): string {
    const project: SandboxProject = { files, activeFile };
    const compressed = encodeSandboxProject(project);
    return `https://pyrefly.org/sandbox/?project=${compressed}`;
}

/**
 * Decode a pyrefly.org/sandbox URL back into its project state.
 */
export function decodeSandboxUrl(url: string): SandboxProject | null {
    let project: string | null;
    try {
        project = new URL(url, 'https://pyrefly.org').searchParams.get(
            'project'
        );
    } catch {
        return null;
    }
    return project ? decodeSandboxProject(project) : null;
}
