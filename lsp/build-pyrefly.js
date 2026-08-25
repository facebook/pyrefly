/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Builds the pyrefly binary and places it in lsp/bin, where the extension looks
// for a bundled language server. Cargo's --artifact-dir would do the placement
// as part of the build, but it is gated behind -Z unstable-options and the
// repository pins the toolchain to stable.
//
// Usage: node build-pyrefly.js [debug|release]

const {spawnSync} = require('child_process');
const fs = require('fs');
const path = require('path');

const profile = process.argv[2] ?? 'release';
if (profile !== 'debug' && profile !== 'release') {
  throw new Error(`Expected a profile of "debug" or "release", got "${profile}"`);
}

// shell: true so that Windows resolves `cargo` to `cargo.exe` via PATHEXT.
const cargo = (args, options) =>
  spawnSync('cargo', args, {
    cwd: path.join(__dirname, '..', 'pyrefly'),
    shell: true,
    ...options,
  });

const build = cargo(['build', profile === 'release' ? '--release' : '--all-features'], {
  stdio: 'inherit',
});
if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

// Ask cargo for the target directory rather than assuming ./target, since it can
// be redirected by CARGO_TARGET_DIR or by build.target-dir in a cargo config.
const metadata = cargo(['metadata', '--format-version', '1', '--no-deps'], {encoding: 'utf8'});
if (metadata.status !== 0) {
  process.stderr.write(metadata.stderr);
  process.exit(metadata.status ?? 1);
}

const name = process.platform === 'win32' ? 'pyrefly.exe' : 'pyrefly';
const binDir = path.join(__dirname, 'bin');
fs.mkdirSync(binDir, {recursive: true});
fs.copyFileSync(
  path.join(JSON.parse(metadata.stdout).target_directory, profile, name),
  path.join(binDir, name),
);
