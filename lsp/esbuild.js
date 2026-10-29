/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');
const {createHash} = require('crypto');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');
const webOnly = process.argv.includes('--web-only');

async function main() {
  const nodeEntryPoints = production
    ? ['src/extension.ts']
    : ['src/test/**/*.test.ts', 'src/extension.ts'];
  const webEntryPoints = ['src/extension-web.ts'];
  const webWorkerEntryPoints = ['src/pyrefly-web-worker.ts'];

  const nodeCtx = webOnly ? undefined : await esbuild.context({
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
      wasmInputsPlugin,
      esbuildProblemMatcherPlugin
    ]
  });

  if (watch) {
    await Promise.all([nodeCtx?.watch(), webCtx.watch(), webWorkerCtx.watch()]);
  } else {
    await Promise.all([nodeCtx?.rebuild(), webCtx.rebuild(), webWorkerCtx.rebuild()]);
    await Promise.all([nodeCtx?.dispose(), webCtx.dispose(), webWorkerCtx.dispose()]);
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

const wasmInputsPlugin = {
  name: 'wasm-inputs',
  setup(build) {
    const inputDir = path.resolve(__dirname, '../pyrefly_wasm/target');
    const js = path.join(inputDir, 'pyrefly_wasm.js');
    const wasm = path.join(inputDir, 'pyrefly_wasm_bg.wasm');
    const manifestPath = path.join(inputDir, 'web-manifest.json');
    let glue;
    build.onStart(() => {
      for (const file of [js, wasm, manifestPath]) {
        if (!fs.existsSync(file)) {
          throw new Error('Missing WASM build input: ' + file +
            '. Run bash pyrefly_wasm/build.sh before building the extension.');
        }
      }
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
      const inputs = new Map([js, wasm].map(file => [file, fs.readFileSync(file)]));
      for (const file of [js, wasm]) {
        const hash = createHash('sha256').update(inputs.get(file)).digest('hex');
        if (hash !== manifest[path.basename(file)]) {
          throw new Error('WASM inputs do not match. Wait for pyrefly_wasm/build.sh to finish.');
        }
      }
      glue = inputs.get(js).toString('utf8');
      fs.mkdirSync(path.resolve(__dirname, 'dist'), {recursive: true});
      fs.writeFileSync(path.resolve(__dirname, 'dist/pyrefly_wasm_bg.wasm'), inputs.get(wasm));
    });
    build.onResolve({filter: /wasm\/pyrefly_wasm$/}, () => ({path: js}));
    build.onLoad({filter: /pyrefly_wasm\.js$/}, () => {
      if (glue === undefined) {
        throw new Error('WASM inputs must pass validation before bundling.');
      }
      return {contents: glue, loader: 'js', watchFiles: [js, wasm, manifestPath]};
    });
    build.onEnd(result => {
      if (result.errors.length) { return; }
      const directory = path.resolve(__dirname, 'dist');
      const manifest = {};
      for (const name of ['pyrefly-web-worker.js', 'pyrefly_wasm_bg.wasm']) {
        manifest[name] = createHash('sha256')
          .update(fs.readFileSync(path.join(directory, name))).digest('hex');
      }
      fs.writeFileSync(path.join(directory, 'web-manifest.json'), JSON.stringify(manifest));
    });
  },
};

main().catch(e => {
  console.error(e);
  process.exit(1);
});
