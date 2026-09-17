/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as LZString from 'lz-string';
import {
    decodeSandboxUrl,
    generateSandboxUrl,
} from '../sandbox/generateSandboxUrl';

describe('generateSandboxUrl', () => {
    test('generates a URL with the correct base', () => {
        const url = generateSandboxUrl({ 'test.py': 'x = 1' });
        expect(url).toMatch(/^https:\/\/pyrefly\.org\/sandbox\/\?project=v2\./);
    });

    test('round-trips single file', () => {
        const files = { 'sandbox.py': 'def hello(): pass' };
        const url = generateSandboxUrl(files);
        const decoded = decodeSandboxUrl(url);
        expect(decoded).not.toBeNull();
        expect(decoded!.files).toEqual(files);
        expect(decoded!.activeFile).toBe('sandbox.py');
    });

    test('round-trips multiple files', () => {
        const files = {
            'sandbox.py': 'import torch\nx = torch.randn(3, 4)',
            'pyrefly.toml': 'python-version = "3.12"',
            'torch.pyi': 'class Tensor[*Shape]: ...',
        };
        const url = generateSandboxUrl(files, 'sandbox.py');
        const decoded = decodeSandboxUrl(url);
        expect(decoded).not.toBeNull();
        expect(decoded!.files).toEqual(files);
        expect(decoded!.activeFile).toBe('sandbox.py');
    });

    test('respects custom activeFile', () => {
        const files = {
            'main.py': 'print("hi")',
            'helper.py': 'def f(): pass',
        };
        const url = generateSandboxUrl(files, 'helper.py');
        const decoded = decodeSandboxUrl(url);
        expect(decoded!.activeFile).toBe('helper.py');
    });

    test('handles empty file content', () => {
        const files = { 'empty.py': '' };
        const url = generateSandboxUrl(files);
        const decoded = decodeSandboxUrl(url);
        expect(decoded!.files['empty.py']).toBe('');
    });

    test('handles special characters in file content', () => {
        const code = 'x: str = "hello\\nworld"\ny = 3.14\n# 日本語コメント';
        const files = { 'sandbox.py': code };
        const url = generateSandboxUrl(files);
        const decoded = decodeSandboxUrl(url);
        expect(decoded!.files['sandbox.py']).toBe(code);
    });

    test('handles large files', () => {
        const longCode = Array(500).fill('x = 1\n').join('');
        const files = { 'sandbox.py': longCode };
        const url = generateSandboxUrl(files);
        const decoded = decodeSandboxUrl(url);
        expect(decoded!.files['sandbox.py']).toBe(longCode);
    });

    test('handles files with newlines at the end', () => {
        const files = { 'sandbox.py': 'x = 1\n' };
        const url = generateSandboxUrl(files);
        const decoded = decodeSandboxUrl(url);
        expect(decoded!.files['sandbox.py']).toBe('x = 1\n');
    });

    test('produces a shorter URL than the legacy encoding', () => {
        const files = {
            'sandbox.py': Array(200)
                .fill('def greet(name: str) -> str:\n    return f"Hi {name}"\n')
                .join(''),
            'pyrefly.toml': 'python-version = "3.12"\n',
        };
        const project = { files, activeFile: 'sandbox.py' };
        const legacy = LZString.compressToEncodedURIComponent(
            JSON.stringify(project)
        );

        expect(generateSandboxUrl(files).length).toBeLessThan(
            `https://pyrefly.org/sandbox/?project=${legacy}`.length
        );
    });
});

describe('decodeSandboxUrl', () => {
    test('returns null for URL without project param', () => {
        expect(decodeSandboxUrl('https://pyrefly.org/sandbox/')).toBeNull();
    });

    test('returns null for URL with empty project param', () => {
        expect(
            decodeSandboxUrl('https://pyrefly.org/sandbox/?project=')
        ).toBeNull();
    });

    test('returns null for URL with invalid compressed data', () => {
        expect(
            decodeSandboxUrl(
                'https://pyrefly.org/sandbox/?project=v2.not-valid-deflate'
            )
        ).toBeNull();
    });

    // Preserve both legacy formats: multi-file projects used `project`, while
    // older single-file links used `code`.

    test('decodes legacy LZString ?project= links', () => {
        const project = {
            files: { 'sandbox.py': 'x = 1\n' },
            activeFile: 'sandbox.py',
        };
        const legacy = LZString.compressToEncodedURIComponent(
            JSON.stringify(project)
        );

        expect(
            decodeSandboxUrl(`https://pyrefly.org/sandbox/?project=${legacy}`)
        ).toEqual(project);
    });

    test('decodes legacy LZString ?code= links', () => {
        const code = 'x = 1\n';
        const legacy = LZString.compressToEncodedURIComponent(code);

        expect(
            decodeSandboxUrl(`https://pyrefly.org/sandbox/?code=${legacy}`)
        ).toEqual({
            files: { 'sandbox.py': code },
            activeFile: 'sandbox.py',
        });
    });
});
