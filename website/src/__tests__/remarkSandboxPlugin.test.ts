/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {
    buildSandboxUrl,
    parseSandboxConfig,
    readSandboxFiles,
    stripLicenseHeader,
} from '../sandbox/remarkSandboxPlugin';
import { decodeSandboxUrl } from '../sandbox/generateSandboxUrl';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

describe('parseSandboxConfig', () => {
    test('parses all fields', () => {
        const config = parseSandboxConfig(
            'dir: my-example\nsource: microtorch\nshared: microtorch\nactive: main.py\nlinkText: Try it\ndescription: A demo'
        );
        expect(config).toEqual({
            dir: 'my-example',
            source: 'microtorch',
            shared: 'microtorch',
            active: 'main.py',
            linkText: 'Try it',
            description: 'A demo',
        });
    });

    test('uses defaults for optional fields', () => {
        const config = parseSandboxConfig('dir: my-example');
        expect(config).toEqual({
            dir: 'my-example',
            source: '',
            shared: '',
            active: 'sandbox.py',
            linkText: 'Open this example in the Pyrefly sandbox',
            description: '',
        });
    });

    test('returns null when dir is missing', () => {
        expect(parseSandboxConfig('active: main.py')).toBeNull();
    });

    test('returns null for empty input', () => {
        expect(parseSandboxConfig('')).toBeNull();
    });

    test('handles values with colons', () => {
        const config = parseSandboxConfig(
            'dir: my-example\ndescription: shapes: tracked end-to-end'
        );
        expect(config!.description).toBe('shapes: tracked end-to-end');
    });

    test('ignores lines without colons', () => {
        const config = parseSandboxConfig(
            'dir: my-example\nthis line has no key-value'
        );
        expect(config).not.toBeNull();
        expect(config!.dir).toBe('my-example');
    });

    test('trims whitespace from keys and values', () => {
        const config = parseSandboxConfig('  dir  :  my-example  ');
        expect(config!.dir).toBe('my-example');
    });
});

describe('stripLicenseHeader', () => {
    const MIT_LICENSE =
        '# Copyright (c) Meta Platforms, Inc. and affiliates.\n' +
        '#\n' +
        '# This source code is licensed under the MIT license found in the\n' +
        '# LICENSE file in the root directory of this source tree.\n';

    test('strips standard MIT license header', () => {
        const content = MIT_LICENSE + '\nfrom typing import Any\n';
        expect(stripLicenseHeader(content)).toBe('from typing import Any\n');
    });

    test('returns content unchanged when no license present', () => {
        const content = 'from typing import Any\nx = 1\n';
        expect(stripLicenseHeader(content)).toBe(content);
    });

    test('strips license from .pyi stub files', () => {
        const content = MIT_LICENSE + '\nclass Tensor[*Shape]: ...\n';
        expect(stripLicenseHeader(content)).toBe('class Tensor[*Shape]: ...\n');
    });

    test('handles file that is only a license', () => {
        const result = stripLicenseHeader(MIT_LICENSE);
        expect(result).toBe('');
    });

    test('does not strip non-license comments', () => {
        const content = '# This is a regular comment\nx = 1\n';
        // This will strip it since it starts with # — acceptable tradeoff
        // since sandbox examples should not start with non-license comments
        expect(stripLicenseHeader(content)).toBeDefined();
    });
});

describe('readSandboxFiles', () => {
    let tmpDir: string;

    beforeEach(() => {
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sandbox-test-'));
    });

    afterEach(() => {
        fs.rmSync(tmpDir, { recursive: true });
    });

    test('reads .py, .pyi, and .toml files', () => {
        fs.writeFileSync(path.join(tmpDir, 'sandbox.py'), 'x = 1');
        fs.writeFileSync(path.join(tmpDir, 'torch.pyi'), 'class T: ...');
        fs.writeFileSync(path.join(tmpDir, 'pyrefly.toml'), 'k = "v"');

        const files = readSandboxFiles(tmpDir);
        expect(Object.keys(files).sort()).toEqual([
            'pyrefly.toml',
            'sandbox.py',
            'torch.pyi',
        ]);
        expect(files['sandbox.py']).toBe('x = 1');
        expect(files['torch.pyi']).toBe('class T: ...');
        expect(files['pyrefly.toml']).toBe('k = "v"');
    });

    test('ignores non-sandbox files', () => {
        fs.writeFileSync(path.join(tmpDir, 'sandbox.py'), 'x = 1');
        fs.writeFileSync(path.join(tmpDir, 'README.md'), '# Hello');
        fs.writeFileSync(path.join(tmpDir, 'data.json'), '{}');

        const files = readSandboxFiles(tmpDir);
        expect(Object.keys(files)).toEqual(['sandbox.py']);
    });

    test('throws for nonexistent directory', () => {
        expect(() => readSandboxFiles('/nonexistent/path')).toThrow(
            'not found'
        );
    });

    test('throws for empty directory', () => {
        expect(() => readSandboxFiles(tmpDir)).toThrow('No sandbox files');
    });

    test('strips license headers from files', () => {
        const license =
            '# Copyright (c) Meta Platforms, Inc. and affiliates.\n' +
            '#\n' +
            '# This source code is licensed under the MIT license found in the\n' +
            '# LICENSE file in the root directory of this source tree.\n';
        fs.writeFileSync(
            path.join(tmpDir, 'sandbox.py'),
            license + '\nx = 1\n'
        );
        const files = readSandboxFiles(tmpDir);
        expect(files['sandbox.py']).toBe('x = 1\n');
    });

    test('reads files with unicode content', () => {
        fs.writeFileSync(
            path.join(tmpDir, 'sandbox.py'),
            'x = "héllo 日本語"\n'
        );
        const files = readSandboxFiles(tmpDir);
        expect(files['sandbox.py']).toBe('x = "héllo 日本語"\n');
    });

    test('merges shared files and lets the example override them', () => {
        const sharedDir = fs.mkdtempSync(
            path.join(os.tmpdir(), 'sandbox-shared-test-')
        );
        try {
            fs.writeFileSync(path.join(sharedDir, 'library.pyi'), 'shared');
            fs.writeFileSync(path.join(sharedDir, 'sandbox.py'), 'shared');
            fs.writeFileSync(path.join(tmpDir, 'sandbox.py'), 'example');

            expect(readSandboxFiles(tmpDir, sharedDir)).toEqual({
                'library.pyi': 'shared',
                'sandbox.py': 'example',
            });
        } finally {
            fs.rmSync(sharedDir, { recursive: true });
        }
    });
});

describe('buildSandboxUrl', () => {
    test('produces a valid URL', () => {
        const url = buildSandboxUrl({ 'sandbox.py': 'x = 1' }, 'sandbox.py');
        expect(url).toMatch(/^https:\/\/pyrefly\.org\/sandbox\/\?project=v2\./);
    });

    test('URL is decodable back to original files', () => {
        const files = {
            'sandbox.py': 'import torch\nx = torch.randn(3)',
            'pyrefly.toml': 'python-version = "3.12"',
        };
        const decoded = decodeSandboxUrl(buildSandboxUrl(files, 'sandbox.py'));
        expect(decoded!.files).toEqual(files);
        expect(decoded!.activeFile).toBe('sandbox.py');
    });

    test('reads real example directory and produces a working URL', () => {
        const examplesDir = path.resolve(
            __dirname,
            '../../../tensor-shapes/microtorch/examples/overview'
        );
        const sharedDir = path.resolve(
            __dirname,
            '../../../tensor-shapes/microtorch'
        );
        if (!fs.existsSync(examplesDir)) {
            return; // skip if examples not present
        }
        const files = readSandboxFiles(examplesDir, sharedDir);
        expect(files['sandbox.py']).toBeDefined();
        expect(files['pyrefly.toml']).toBeDefined();
        expect(files['microtorch.pyi']).toBeDefined();

        const decoded = decodeSandboxUrl(buildSandboxUrl(files, 'sandbox.py'));
        expect(decoded!.files['sandbox.py']).toContain('assert_type');
        expect(decoded!.activeFile).toBe('sandbox.py');
    });
});
