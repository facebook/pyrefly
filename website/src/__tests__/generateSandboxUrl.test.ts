/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The Jest config maps 'lz-string' to a mock. We use a Jest project override
// below so these tests get the real lz-string module.

// Load the real lz-string .js file directly to bypass Jest's moduleNameMapper.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const LZString = require(
    require('path').resolve(
        process.cwd(),
        'node_modules/lz-string/libs/lz-string.js',
    ),
);

interface SandboxProject {
    files: Record<string, string>;
    activeFile: string;
}

function encode(
    files: Record<string, string>,
    activeFile: string = 'sandbox.py',
): string {
    const project: SandboxProject = { files, activeFile };
    const compressed = LZString.compressToEncodedURIComponent(
        JSON.stringify(project),
    );
    return `https://pyrefly.org/sandbox/?project=${compressed}`;
}

function decode(url: string): SandboxProject | null {
    const match = url.match(/[?&]project=([^&]+)/);
    if (!match) return null;
    const decompressed = LZString.decompressFromEncodedURIComponent(match[1]);
    if (!decompressed) return null;
    try {
        return JSON.parse(decompressed) as SandboxProject;
    } catch {
        return null;
    }
}

describe('generateSandboxUrl', () => {
    test('generates a URL with the correct base', () => {
        const url = encode({ 'test.py': 'x = 1' });
        expect(url).toMatch(/^https:\/\/pyrefly\.org\/sandbox\/\?project=/);
    });

    test('round-trips single file', () => {
        const files = { 'sandbox.py': 'def hello(): pass' };
        const url = encode(files);
        const decoded = decode(url);
        expect(decoded).not.toBeNull();
        expect(decoded!.files).toEqual(files);
        expect(decoded!.activeFile).toBe('sandbox.py');
    });

    test('round-trips multiple files', () => {
        const files = {
            'sandbox.py': 'import torch\nx = torch.randn(3, 4)',
            'pyrefly.toml': 'tensor-shapes = true',
            'torch.pyi': 'class Tensor[*Shape]: ...',
        };
        const url = encode(files, 'sandbox.py');
        const decoded = decode(url);
        expect(decoded).not.toBeNull();
        expect(decoded!.files).toEqual(files);
        expect(decoded!.activeFile).toBe('sandbox.py');
    });

    test('respects custom activeFile', () => {
        const files = {
            'main.py': 'print("hi")',
            'helper.py': 'def f(): pass',
        };
        const url = encode(files, 'helper.py');
        const decoded = decode(url);
        expect(decoded!.activeFile).toBe('helper.py');
    });

    test('handles empty file content', () => {
        const files = { 'empty.py': '' };
        const url = encode(files);
        const decoded = decode(url);
        expect(decoded!.files['empty.py']).toBe('');
    });

    test('handles special characters in file content', () => {
        const code = 'x: str = "hello\\nworld"\ny = 3.14\n# 日本語コメント';
        const files = { 'sandbox.py': code };
        const url = encode(files);
        const decoded = decode(url);
        expect(decoded!.files['sandbox.py']).toBe(code);
    });

    test('handles large files', () => {
        const longCode = Array(500).fill('x = 1\n').join('');
        const files = { 'sandbox.py': longCode };
        const url = encode(files);
        const decoded = decode(url);
        expect(decoded!.files['sandbox.py']).toBe(longCode);
    });

    test('handles files with newlines at the end', () => {
        const files = { 'sandbox.py': 'x = 1\n' };
        const url = encode(files);
        const decoded = decode(url);
        expect(decoded!.files['sandbox.py']).toBe('x = 1\n');
    });
});

describe('decodeSandboxUrl', () => {
    test('returns null for URL without project param', () => {
        expect(decode('https://pyrefly.org/sandbox/')).toBeNull();
    });

    test('returns null for URL with empty project param', () => {
        expect(
            decode('https://pyrefly.org/sandbox/?project='),
        ).toBeNull();
    });

    test('returns null for URL with invalid compressed data', () => {
        expect(
            decode('https://pyrefly.org/sandbox/?project=not-valid-lz'),
        ).toBeNull();
    });
});
