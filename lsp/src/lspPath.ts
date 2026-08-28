/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import {homedir} from 'os';
import {join, parse, resolve} from 'path';

/**
 * Resolve a `pyrefly.lspPath`, expanding ~ and handling relative paths.
 *
 * We tell a `$PATH` lookup apart from a relative or absolute by checking
 * for a path separator. `~\` is only a home-relative prefix on Windows;
 * on POSIX `\` is an ordinary filename character, so we leave it alone.
 */
export function resolveLspPath(
  lspPath: string,
  cwd: string | undefined,
): string {
  const isHomeRelative =
    lspPath.startsWith('~/') ||
    (process.platform === 'win32' && lspPath.startsWith('~\\'));
  if (isHomeRelative) {
    return join(homedir(), lspPath.slice(1));
  }
  if (cwd == null || parse(lspPath).dir === '') {
    return lspPath;
  }
  return resolve(cwd, lspPath);
}
