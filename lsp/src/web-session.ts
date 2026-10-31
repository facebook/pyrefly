/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {Position, Range} from 'vscode-languageserver-protocol';
import {ScalarPosition, ScalarRange, State} from '../wasm/pyrefly_wasm';

export const READ_FILE = 'pyrefly/readWorkspaceFile';
export const SUPPORTED_SCHEMES = ['file', 'vscode-vfs', 'untitled'];
export const CONFIG_FILES = ['pyrefly.toml', 'pyproject.toml'];

export interface ProjectOptions {
  uri: string;
  pythonVersion: string;
  files: Record<string, string>;
}
export interface ReadParams {
  rootUri: string;
  filenames: string[];
}
export interface WorkspaceFile {
  uri: string;
  filename: string;
  content: string;
}

/** Convert an LSP UTF-16 position to a one-based Unicode-scalar position. */
export function toScalar(position: Position, text: string): ScalarPosition {
  const line = text.split('\n')[position.line] ?? '';
  return {
    lineNumber: position.line + 1,
    column: Array.from(line.slice(0, position.character)).length + 1,
  };
}

/** Convert a one-based Unicode-scalar position to an LSP UTF-16 position. */
export function toLsp(position: ScalarPosition, text: string): Position {
  const line = text.split('\n')[position.lineNumber - 1] ?? '';
  return {
    line: position.lineNumber - 1,
    character: Array.from(line).slice(0, position.column - 1).join('').length,
  };
}

export function toLspRange(range: ScalarRange, text: string): Range {
  return {
    start: toLsp({lineNumber: range.startLineNumber, column: range.startColumn}, text),
    end: toLsp({lineNumber: range.endLineNumber, column: range.endColumn}, text),
  };
}

/** Match URI authority and directory boundaries before computing relative names. */
export function relativeFilename(root: string, uri: string): string | undefined {
  const base = new URL(root);
  const resource = new URL(uri);
  const prefix = decodeURIComponent(base.pathname).replace(/\/$/, '') + '/';
  const path = decodeURIComponent(resource.pathname);
  if (base.protocol !== resource.protocol || base.host !== resource.host ||
      !path.startsWith(prefix)) {
    return undefined;
  }
  const name = path.slice(prefix.length);
  return name && !name.split('/').includes('..') ? name : undefined;
}

export function fileUri(root: string, filename: string): string {
  return root.replace(/\/$/, '') + '/' +
    filename.split('/').map(encodeURIComponent).join('/');
}

/** One WASM state and import cache belong to exactly one workspace. */
export class WebSession {
  readonly files = new Map<string, string>();
  readonly uris = new Map<string, string>();
  readonly opened = new Set<string>();
  private readonly reads = new Map<string, Promise<WorkspaceFile | null>>();
  private applied = new Map<string, string>();
  private revision = 0;

  constructor(
    readonly options: ProjectOptions,
    readonly state: State,
    private readonly read: (params: ReadParams) => Promise<WorkspaceFile | null>,
  ) {
    for (const [name, text] of Object.entries(options.files)) {
      this.files.set(name, text);
      this.uris.set(name, fileUri(options.uri, name));
    }
  }

  /** Cache successful, missing, and in-flight reads until a filesystem event. */
  private load(module: string): Promise<WorkspaceFile | null> {
    let request = this.reads.get(module);
    if (!request) {
      const path = module.replace(/\./g, '/');
      request = this.read({
        rootUri: this.options.uri,
        filenames: [path + '/__init__.pyi', path + '/__init__.py', path + '.pyi', path + '.py'],
      });
      this.reads.set(module, request);
    }
    return request;
  }

  /** Recheck until requested workspace modules are loaded or known absent. */
  async rebuild(): Promise<void> {
    this.syncState();
    for (;;) {
      let added = false;
      for (const module of this.state.pendingImports()) {
        const revision = this.revision;
        const response = await this.load(module);
        if (revision !== this.revision) {
          continue;
        }
        if (response && !this.files.has(response.filename)) {
          this.files.set(response.filename, response.content);
          this.uris.set(response.filename, response.uri);
          added = true;
        }
      }
      if (!added) {
        return;
      }
      this.syncState();
    }
  }

  /** Preserve the warm Rust state when only existing source text changes. */
  private syncState(): void {
    let total = 0;
    for (const [name, text] of this.files) {
      const size = new TextEncoder().encode(text).length;
      if (size > 1024 * 1024) {
        throw new Error('Pyrefly web supports files up to 1 MiB: ' + name);
      }
      total += size;
    }
    if (this.files.size > 1000 || total > 10 * 1024 * 1024) {
      throw new Error('Pyrefly web supports at most 1000 files and 10 MiB per workspace.');
    }
    const sameModules = this.applied.size === this.files.size &&
      Array.from(this.files.keys()).every(name => this.applied.has(name)) &&
      CONFIG_FILES.every(name => this.applied.get(name) === this.files.get(name));
    if (sameModules) {
      for (const [name, text] of this.files) {
        if (text !== this.applied.get(name)) {
          this.state.updateSingleFile(name, text);
        }
      }
    } else {
      this.state.updateSandboxFiles(Object.fromEntries(this.files), true);
    }
    this.applied = new Map(this.files);
  }

  /** Remove stale disk content and cached reads after filesystem events. */
  invalidate(filename: string): void {
    this.revision++;
    this.reads.clear();
    if (!this.opened.has(filename)) {
      this.files.delete(filename);
      this.uris.delete(filename);
    }
  }
}
