/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as assert from 'assert';
import {homedir, tmpdir} from 'os';
import {dirname, join} from 'path';
import {resolveLspPath} from '../lspPath';

suite('resolveLspPath', () => {
  const workspace = join(tmpdir(), 'pyrefly-workspace');

  test('don\'t touch default', () => {
    assert.strictEqual(resolveLspPath('', workspace), '');
  });

  test('keep $PATH lookup untouched', () => {
    assert.strictEqual(resolveLspPath('pyrefly', workspace), 'pyrefly');
  });

  test('leave an absolute path alone', () => {
    const absolute = join(workspace, 'pyrefly');
    assert.strictEqual(resolveLspPath(absolute, workspace), absolute);
  });

  test('resolve relative paths against workspace root', () => {
    for (const relative of ['./bin/pyrefly', 'target/debug/pyrefly']) {
      assert.strictEqual(
        resolveLspPath(relative, workspace),
        join(workspace, relative),
      );
    }
  });

  test('resolves parent-relative paths against workspace root', () => {
    assert.strictEqual(
      resolveLspPath('../target/debug/pyrefly', workspace),
      join(dirname(workspace), 'target', 'debug', 'pyrefly'),
    );
  });

  test('handle windows path separators', () => {
    const result = resolveLspPath('.\\bin\\pyrefly.exe', workspace);
    if (process.platform === 'win32') {
      assert.strictEqual(result, join(workspace, 'bin', 'pyrefly.exe'));
    } else {
      assert.strictEqual(result, '.\\bin\\pyrefly.exe');
    }
  });

  test('expand a ~-relative path against homedir', () => {
    assert.strictEqual(
      resolveLspPath('~/bin/pyrefly', workspace),
      join(homedir(), 'bin', 'pyrefly'),
    );
  });

  test('don\'t do anything with an unknown workspace', () => {
    assert.strictEqual(
      resolveLspPath('./bin/pyrefly', undefined),
      './bin/pyrefly',
    );
  });
});
