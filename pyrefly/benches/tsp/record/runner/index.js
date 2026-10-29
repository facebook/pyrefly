/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// VS Code extension test entry point that drives Pylance through the features
// that `benches/tsp/pytorch.rs` replays. Before each feature it writes a
// `=== PHASE <name>` marker into the files that `record.sh` captures, so that
// `extract.py` can split the recorded TSP requests by feature.

const fs = require('fs');
const vscode = require('vscode');

const WORK = process.env.RECORD_WORK;
const IR = `${process.env.RECORD_PYTORCH}/torch/_inductor/ir.py`;
// Keep in sync with `LINE` and `METHOD` in `benches/tsp/pytorch.rs`.
const POSITION = new vscode.Position(489, 28);

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

// Copy Pylance's log to `${WORK}/pylance.log` as it grows. VS Code rotates the
// log every 5 MB, which a single cold start or re-check outgrows, by renaming
// `Python Language Server.log` to `Python Language Server.1.log`; when the log
// shrinks, the rest of the old file is in `.1.log`.
function tailPylanceLog() {
  let offset = 0;
  const copy = (path, from) => {
    const data = fs.readFileSync(path);
    fs.appendFileSync(`${WORK}/pylance.log`, data.subarray(from));
    return data.length;
  };
  return setInterval(() => {
    const sessions = fs.readdirSync(`${WORK}/user/logs`).sort();
    const dir = `${WORK}/user/logs/${sessions[sessions.length - 1]}/window1/exthost/ms-python.python`;
    const log = `${dir}/Python Language Server.log`;
    if (!fs.existsSync(log)) return;
    if (fs.statSync(log).size < offset) {
      copy(`${dir}/Python Language Server.1.log`, offset);
      offset = 0;
    }
    offset = copy(log, offset);
  }, 100);
}

function mark(phase) {
  const line = `\n=== PHASE ${phase} ${Date.now()}\n`;
  fs.appendFileSync(`${WORK}/pyrefly.log`, line);
  fs.appendFileSync(`${WORK}/main_in.log`, line);
}

// Wait until pyrefly has logged nothing for `ms`, i.e. Pylance has stopped
// sending requests.
async function quiet(ms) {
  const size = () => fs.statSync(`${WORK}/pyrefly.log`).size;
  let last = size();
  let since = Date.now();
  while (Date.now() - since < ms) {
    await sleep(500);
    if (size() !== last) {
      last = size();
      since = Date.now();
    }
  }
}

exports.run = async function () {
  const tail = tailPylanceLog();
  await vscode.extensions.getExtension('ms-python.vscode-pylance').activate();
  const uri = vscode.Uri.file(IR);
  const doc = await vscode.workspace.openTextDocument(uri);
  const editor = await vscode.window.showTextDocument(doc);

  // Opening the file starts Pylance's cold start: it resolves imports and
  // its background checker sweeps the file. Pyrefly logs nothing before it
  // serves a request, so wait for the first one before waiting for quiet.
  const deadline = Date.now() + 600000;
  while (!fs.readFileSync(`${WORK}/pyrefly.log`, 'utf8').includes('processed event')) {
    if (Date.now() > deadline) throw new Error('Pylance did not use pyrefly; see the Pylance log');
    await sleep(500);
  }
  await quiet(5000);
  mark('startup_done');

  // The first answered hover means that both servers are up and the file is
  // checked. Pylance answers on its own when it does not use pyrefly, so also
  // require that pyrefly has served a request.
  for (;;) {
    const hovers = await vscode.commands.executeCommand('vscode.executeHoverProvider', uri, POSITION);
    const served = fs.readFileSync(`${WORK}/pyrefly.log`, 'utf8').includes('processed event');
    if (hovers.length > 0 && served) break;
    if (Date.now() > deadline) throw new Error('Pylance did not use pyrefly; see the Pylance log');
    await sleep(2000);
  }
  await quiet(5000);

  let edits = 0;
  for (const [phase, serve] of [
    ['hover', () => vscode.commands.executeCommand('vscode.executeHoverProvider', uri, POSITION)],
    ['completion', () => vscode.commands.executeCommand('vscode.executeCompletionItemProvider', uri, POSITION)],
    ['recheck', async () => {}],
  ]) {
    mark(`${phase}_edit`);
    await editor.edit(edit => edit.insert(new vscode.Position(doc.lineCount, 0), `\n# recorder edit ${++edits}\n`));
    mark(phase);
    await serve();
    await quiet(5000);
  }
  mark('done');
  await sleep(500);
  clearInterval(tail);
};
