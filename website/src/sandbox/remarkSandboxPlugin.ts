/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Remark plugin that transforms `sandbox` code blocks into sandbox links.
 *
 * Usage in MDX:
 *
 *     ```sandbox
 *     dir: tensor-shapes-overview
 *     active: sandbox.py
 *     linkText: Open this example in the Pyrefly sandbox
 *     description: See tensor shape tracking in action.
 *     ```
 *
 * The plugin reads all `.py`, `.pyi`, and `.toml` files from the directory
 * `website/sandbox-examples/{dir}/`, compresses them into a sandbox URL,
 * and replaces the code block with a :::tip admonition containing the link.
 *
 * Supported fields (parsed as `key: value` lines):
 *   - dir (required): subdirectory under sandbox-examples/
 *   - active: which file to show initially (default: sandbox.py)
 *   - linkText: the clickable link text
 *   - description: additional text after the link
 */

import * as fs from 'fs';
import * as path from 'path';
import * as LZString from 'lz-string';

const SANDBOX_EXTENSIONS = ['.py', '.pyi', '.toml'];

export interface SandboxConfig {
    dir: string;
    active: string;
    linkText: string;
    description: string;
}

export function parseSandboxConfig(body: string): SandboxConfig | null {
    const lines = body.trim().split('\n');
    const config: Record<string, string> = {};

    for (const line of lines) {
        const colonIdx = line.indexOf(':');
        if (colonIdx === -1) continue;
        const key = line.slice(0, colonIdx).trim();
        const value = line.slice(colonIdx + 1).trim();
        if (key && value) {
            config[key] = value;
        }
    }

    if (!config.dir) {
        return null;
    }

    return {
        dir: config.dir,
        active: config.active ?? 'sandbox.py',
        linkText:
            config.linkText ?? 'Open this example in the Pyrefly sandbox',
        description: config.description ?? '',
    };
}

/**
 * Strip the standard Meta MIT license header from file content so it
 * doesn't appear in the sandbox URL or the user-facing playground.
 */
export function stripLicenseHeader(content: string): string {
    const lines = content.split('\n');
    let i = 0;
    // Skip leading comment lines that are part of the license block
    while (i < lines.length && lines[i].startsWith('#')) {
        i++;
    }
    // Skip the blank line after the license block
    if (i > 0 && i < lines.length && lines[i].trim() === '') {
        i++;
    }
    // Only strip if we actually found comment lines (don't strip code)
    if (i === 0) {
        return content;
    }
    return lines.slice(i).join('\n');
}

export function readSandboxFiles(dirPath: string): Record<string, string> {
    const files: Record<string, string> = {};

    if (!fs.existsSync(dirPath)) {
        throw new Error(`Sandbox examples directory not found: ${dirPath}`);
    }

    const entries = fs.readdirSync(dirPath);
    for (const entry of entries) {
        const ext = path.extname(entry);
        if (SANDBOX_EXTENSIONS.includes(ext)) {
            const filePath = path.join(dirPath, entry);
            const raw = fs.readFileSync(filePath, 'utf-8');
            files[entry] = stripLicenseHeader(raw);
        }
    }

    if (Object.keys(files).length === 0) {
        throw new Error(
            `No sandbox files found in ${dirPath} (expected .py, .pyi, or .toml files)`,
        );
    }

    return files;
}

export function buildSandboxUrl(
    files: Record<string, string>,
    activeFile: string,
): string {
    const project = { files, activeFile };
    const compressed = LZString.compressToEncodedURIComponent(
        JSON.stringify(project),
    );
    return `https://pyrefly.org/sandbox/?project=${compressed}`;
}

// Walks an mdast tree, calling `visitor(node, index, parent)` for every node
// with node.type === targetType.
function visit(
    tree: any,
    targetType: string,
    visitor: (node: any, index: number, parent: any) => void,
): void {
    function walk(node: any, index: number, parent: any): void {
        if (node.type === targetType) {
            visitor(node, index, parent);
        }
        if (Array.isArray(node.children)) {
            for (let i = 0; i < node.children.length; i++) {
                walk(node.children[i], i, node);
            }
        }
    }
    walk(tree, 0, null);
}

export interface RemarkSandboxPluginOptions {
    sandboxExamplesDir?: string;
}

function remarkSandboxPlugin(options?: RemarkSandboxPluginOptions) {
    const sandboxExamplesDir =
        options?.sandboxExamplesDir ??
        path.resolve(__dirname, '../../sandbox-examples');

    return (tree: any) => {
        const replacements: Array<{
            parent: any;
            index: number;
            newNode: any;
        }> = [];

        visit(tree, 'code', (node: any, index: number, parent: any) => {
            if (node.lang !== 'sandbox' || !parent) return;

            const config = parseSandboxConfig(node.value);
            if (!config) {
                throw new Error(
                    `Invalid sandbox code block: missing "dir" field. Content:\n${node.value}`,
                );
            }

            const dirPath = path.join(sandboxExamplesDir, config.dir);
            const files = readSandboxFiles(dirPath);
            const url = buildSandboxUrl(files, config.active);

            // Build a Docusaurus admonition AST node (:::tip)
            const linkNode = {
                type: 'link',
                url,
                children: [{ type: 'text', value: config.linkText }],
            };
            const paragraphChildren: any[] = [linkNode];
            if (config.description) {
                paragraphChildren.push({
                    type: 'text',
                    value: ' — ' + config.description,
                });
            }

            const admonition = {
                type: 'containerDirective',
                name: 'tip',
                attributes: {},
                children: [
                    {
                        type: 'containerDirectiveLabel',
                        children: [
                            { type: 'text', value: 'Try it yourself' },
                        ],
                    },
                    {
                        type: 'paragraph',
                        children: paragraphChildren,
                    },
                ],
                data: {
                    hName: 'admonition',
                    hProperties: { title: 'Try it yourself', type: 'tip' },
                },
            };

            replacements.push({ parent, index, newNode: admonition });
        });

        // Apply in reverse to keep indices valid
        for (let i = replacements.length - 1; i >= 0; i--) {
            const { parent, index, newNode } = replacements[i];
            parent.children.splice(index, 1, newNode);
        }
    };
}

export default remarkSandboxPlugin;
