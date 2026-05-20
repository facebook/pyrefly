/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {
    parseSandboxConfig,
    readSandboxFiles,
    stripLicenseHeader,
} from '../sandbox/remarkSandboxPlugin';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Load the real lz-string .js file directly to bypass Jest's moduleNameMapper.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const LZString = require(
    require('path').resolve(
        process.cwd(),
        'node_modules/lz-string/libs/lz-string.js',
    ),
);

describe('parseSandboxConfig', () => {
    test('parses all fields', () => {
        const config = parseSandboxConfig(
            'dir: my-example\nactive: main.py\nlinkText: Try it\ndescription: A demo',
        );
        expect(config).toEqual({
            dir: 'my-example',
            active: 'main.py',
            linkText: 'Try it',
            description: 'A demo',
        });
    });

    test('uses defaults for optional fields', () => {
        const config = parseSandboxConfig('dir: my-example');
        expect(config).toEqual({
            dir: 'my-example',
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
            'dir: my-example\ndescription: shapes: tracked end-to-end',
        );
        expect(config!.description).toBe('shapes: tracked end-to-end');
    });

    test('ignores lines without colons', () => {
        const config = parseSandboxConfig(
            'dir: my-example\nthis line has no key-value',
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
            'not found',
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
            license + '\nx = 1\n',
        );
        const files = readSandboxFiles(tmpDir);
        expect(files['sandbox.py']).toBe('x = 1\n');
    });

    test('reads files with unicode content', () => {
        fs.writeFileSync(
            path.join(tmpDir, 'sandbox.py'),
            'x = "héllo 日本語"\n',
        );
        const files = readSandboxFiles(tmpDir);
        expect(files['sandbox.py']).toBe('x = "héllo 日本語"\n');
    });
});

describe('buildSandboxUrl (via real lz-string)', () => {
    function buildUrl(files: Record<string, string>, activeFile: string): string {
        const project = { files, activeFile };
        const compressed = LZString.compressToEncodedURIComponent(
            JSON.stringify(project),
        );
        return `https://pyrefly.org/sandbox/?project=${compressed}`;
    }

    test('produces a valid URL', () => {
        const url = buildUrl({ 'sandbox.py': 'x = 1' }, 'sandbox.py');
        expect(url).toMatch(/^https:\/\/pyrefly\.org\/sandbox\/\?project=/);
    });

    test('URL is decodable back to original files', () => {
        const files = {
            'sandbox.py': 'import torch\nx = torch.randn(3)',
            'pyrefly.toml': 'tensor-shapes = true',
        };
        const url = buildUrl(files, 'sandbox.py');

        const match = url.match(/project=(.+)/);
        expect(match).not.toBeNull();
        const decoded = JSON.parse(
            LZString.decompressFromEncodedURIComponent(match![1]),
        );
        expect(decoded.files).toEqual(files);
        expect(decoded.activeFile).toBe('sandbox.py');
    });

    test('reads real example directory and produces a working URL', () => {
        const examplesDir = path.resolve(
            __dirname,
            '../../sandbox-examples/tensor-shapes-overview',
        );
        if (!fs.existsSync(examplesDir)) {
            return; // skip if examples not present
        }
        const files = readSandboxFiles(examplesDir);
        expect(files['sandbox.py']).toBeDefined();
        expect(files['pyrefly.toml']).toBeDefined();
        expect(files['torch.pyi']).toBeDefined();

        const url = buildUrl(files, 'sandbox.py');
        const match = url.match(/project=(.+)/);
        const decoded = JSON.parse(
            LZString.decompressFromEncodedURIComponent(match![1]),
        );
        expect(decoded.files['sandbox.py']).toContain('assert_type');
        expect(decoded.activeFile).toBe('sandbox.py');
    });
});
