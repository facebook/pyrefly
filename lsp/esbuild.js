/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

async function main() {
  const nodeEntryPoints = production
    ? ['src/extension.ts']
    : ['src/test/**/*.test.ts', 'src/extension.ts'];
  const webEntryPoints = ['src/extension-web.ts'];
  const webWorkerEntryPoints = ['src/pyrefly-web-worker.ts'];

  const nodeCtx = await esbuild.context({
    entryPoints: nodeEntryPoints,
    bundle: true,
    format: 'cjs',
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: 'node',
    outdir: 'dist',
    external: ['vscode'],
    logLevel: 'warning',
    plugins: [
      /* add to the end of plugins array */
      esbuildProblemMatcherPlugin
    ]
  });

  const webCtx = await esbuild.context({
    entryPoints: webEntryPoints,
    bundle: true,
    format: 'cjs',
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: 'browser',
    outdir: 'dist',
    external: ['vscode'],
    logLevel: 'warning',
    plugins: [
      /* add to the end of plugins array */
      esbuildProblemMatcherPlugin
    ]
  });

  const webWorkerCtx = await esbuild.context({
    entryPoints: webWorkerEntryPoints,
    bundle: true,
    format: 'iife',
    minify: production,
    sourcemap: !production,
    sourcesContent: false,
    platform: 'browser',
    outfile: 'dist/pyrefly-web-worker.js',
    external: ['vscode'],
    logLevel: 'warning',
    define: {
      'import.meta.url': 'self.location.href'
    },
    plugins: [
      /* add to the end of plugins array */
      esbuildProblemMatcherPlugin
    ]
  });

  if (watch) {
    await Promise.all([nodeCtx.watch(), webCtx.watch(), webWorkerCtx.watch()]);
  } else {
    await Promise.all([nodeCtx.rebuild(), webCtx.rebuild(), webWorkerCtx.rebuild()]);
    copyWasmArtifact();
    await Promise.all([nodeCtx.dispose(), webCtx.dispose(), webWorkerCtx.dispose()]);
  }
}

/**
 * @type {import('esbuild').Plugin}
 */
const esbuildProblemMatcherPlugin = {
  name: 'esbuild-problem-matcher',

  setup(build) {
    build.onStart(() => {
      console.log('[watch] build started');
    });
    build.onEnd(result => {
      result.errors.forEach(({ text, location }) => {
        console.error(`✘ [ERROR] ${text}`);
        if (location == null) return;
        console.error(`    ${location.file}:${location.line}:${location.column}:`);
      });
      console.log('[watch] build finished');
    });
  }
};

function copyWasmArtifact() {
  const wasmDir = path.join(__dirname, '..', 'pyrefly_wasm', 'target');
  const outputPath = path.join(__dirname, 'dist', 'pyrefly_wasm_bg.wasm');
  const candidates = [
    path.join(wasmDir, 'pyrefly_wasm_bg.wasm.opt'),
    path.join(wasmDir, 'pyrefly_wasm_bg.wasm'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      fs.copyFileSync(candidate, outputPath);
      console.log('[pyrefly-wasm] copied wasm artifact to dist/');
      return;
    }
  }
  console.warn(
    '[pyrefly-wasm] wasm artifact not found; build pyrefly_wasm to enable web support.',
  );
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
