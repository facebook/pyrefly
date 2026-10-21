/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sits between VS Code and the language server and can hold back the client's
// responses to the server's `workspace/configuration` requests. Every other
// message passes through in order, so the requests that VS Code sends while a
// response is held reach the server before the settings do.
//
// Usage: node lsp-shim.js <server> <instant|delay|drop> <log> [server args...]
//
// `instant` forwards every configuration response as soon as the client sends
// it. `delay` copies a slow Python extension, which the client waits on before it
// answers any configuration request: the shim holds every configuration
// response from the first one until HOLD_MS later, then forwards them in
// order. `drop` copies a client that never answers the first configuration
// request: the shim discards that response and forwards the rest.
//
// The log is one JSON object per line, naming each client message in the order
// that the shim forwarded it to the server, so a test can check that the race
// it relies on really happened.

const {spawn} = require('child_process');
const fs = require('fs');

const HOLD_MS = 3000;
const [server, mode, logPath, ...serverArgs] = process.argv.slice(2);
if (!['instant', 'delay', 'drop'].includes(mode)) {
  throw new Error(`unknown mode: ${mode}`);
}
fs.writeFileSync(logPath, '');

function log(entry) {
  fs.appendFileSync(logPath, JSON.stringify(entry) + '\n');
}

const child = spawn(server, serverArgs, {stdio: ['pipe', 'pipe', 'inherit']});
child.on('exit', code => process.exit(code ?? 1));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}

/** Call `onMessage` with each complete LSP frame read from `stream`. */
function readFrames(stream, onMessage) {
  let buffer = Buffer.alloc(0);
  stream.on('data', chunk => {
    buffer = Buffer.concat([buffer, chunk]);
    for (;;) {
      const headerEnd = buffer.indexOf('\r\n\r\n');
      if (headerEnd < 0) {
        return;
      }
      const length = Number(
        /Content-Length: *(\d+)/i.exec(buffer.subarray(0, headerEnd).toString())[1],
      );
      const bodyStart = headerEnd + 4;
      if (buffer.length < bodyStart + length) {
        return;
      }
      const frame = buffer.subarray(0, bodyStart + length);
      const message = JSON.parse(buffer.subarray(bodyStart, bodyStart + length));
      buffer = buffer.subarray(bodyStart + length);
      onMessage(frame, message);
    }
  });
}

/** The ids of the server's configuration requests, in the order sent. */
const configurationIds = [];

readFrames(child.stdout, (frame, message) => {
  if (message.method === 'workspace/configuration') {
    configurationIds.push(message.id);
  }
  process.stdout.write(frame);
});

function forwardConfigurationResponse(frame) {
  log({method: 'workspace/configuration response'});
  child.stdin.write(frame);
}

/** In `delay` mode, the responses held until the hold ends; `undefined` after. */
let held;
let holdStarted = false;

readFrames(process.stdin, (frame, message) => {
  const isConfigurationResponse =
    message.method === undefined && configurationIds.includes(message.id);
  if (!isConfigurationResponse) {
    log({method: message.method ?? 'response'});
    child.stdin.write(frame);
  } else if (mode === 'instant') {
    forwardConfigurationResponse(frame);
  } else if (mode === 'drop') {
    if (message.id === configurationIds[0]) {
      log({dropped: 'workspace/configuration response'});
    } else {
      forwardConfigurationResponse(frame);
    }
  } else if (!holdStarted) {
    holdStarted = true;
    log({held: 'workspace/configuration response'});
    held = [frame];
    setTimeout(() => {
      held.forEach(forwardConfigurationResponse);
      held = undefined;
    }, HOLD_MS);
  } else if (held !== undefined) {
    held.push(frame);
  } else {
    forwardConfigurationResponse(frame);
  }
});
// The editor closing the pipe ends the session, even when the server still
// holds a request that never completes.
process.stdin.on('end', () => {
  child.stdin.end();
  setTimeout(() => child.kill(), 1000).unref();
});
