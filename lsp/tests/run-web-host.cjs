/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const fs = require('node:fs');
const path = require('node:path');
const {open} = require('@vscode/test-web');
const {chromium} = require('playwright');

async function main() {
  const extension = path.resolve(__dirname, '../web-test/extension');
  const testPath = path.join(extension, 'dist/web-test/suite.js');
  fs.mkdirSync(path.dirname(testPath), {recursive: true});
  fs.copyFileSync(path.resolve(__dirname, '../dist/web-test/suite.js'), testPath);
  const server = await open({
    extensionDevelopmentPath: extension,
    extensionTestsPath: testPath,
    browserType: 'none',
    quality: 'stable',
    coi: true,
  });
  const browser = await chromium.launch({
    headless: true, args: process.platform === 'linux' ? ['--no-sandbox'] : [],
  });
  try {
    const page = await browser.newPage();
    await page.exposeFunction('codeAutomationLog', (type, args) => console[type](...args));
    let finish;
    const result = new Promise(resolve => { finish = resolve; });
    // Close the browser outside its exposed callback so callback completion cannot deadlock.
    await page.exposeFunction('codeAutomationExit', code => { setImmediate(() => finish(code)); });
    await page.goto('http://localhost:3000');
    let timeout;
    try {
      const code = await Promise.race([
        result,
        new Promise((_, reject) => {
          timeout = setTimeout(() => reject(new Error('Browser test exceeded 120 seconds.')), 120000);
        }),
      ]);
      if (code !== 0) { throw new Error('Packaged browser-extension test failed: ' + code); }
      console.log('Packaged browser-extension test passed.');
    } finally {
      clearTimeout(timeout);
    }
  } finally {
    await browser.close();
    server.dispose();
  }
}

main().catch(error => { console.error(error); process.exit(1); });
