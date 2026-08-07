/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {LanguageClient} from 'vscode-languageclient/node';

let statusBarItem: vscode.StatusBarItem;

/**
 * The `pyrefly/textDocument/typeErrorDisplayStatus` wire-shape version
 * this client supports. Declared to the server in
 * `initializationOptions.pyrefly.typeErrorDisplayStatusVersion` (see
 * `extension.ts`). Bump this in lockstep with adding a new V<N>
 * renderer below, so the version literal, the response type, and the
 * dispatch all change together.
 */
export const TYPE_ERROR_DISPLAY_STATUS_VERSION = 'v2' as const;

/**
 * V2 wire shape for `pyrefly/textDocument/typeErrorDisplayStatus`. The
 * client opts into this richer shape via the version handshake above.
 * An older binary that doesn't know V2 returns a bare V1 string when
 * it sees the unrecognized version, so when running against a pre-V2
 * binary we receive a string and fall back to the V1 renderer below.
 */
type TypeErrorDisplayStatusV2 = {
  version: typeof TYPE_ERROR_DISPLAY_STATUS_VERSION;
  // null → status bar shows just "Pyrefly" (no parenthetical).
  // string → status bar shows `Pyrefly (label)`.
  label: string | null;
  // Markdown — fed straight into MarkdownString.
  tooltip: string;
  docsUrl: string;
};

/// Update the status bar based on current configuration
export async function updateStatusBar(client: LanguageClient) {
  const document = vscode.window.activeTextEditor?.document;
  if (
    document == null ||
    (document.uri.scheme !== 'file' &&
      document.uri.scheme !== 'vscode-notebook-cell' &&
      document.uri.scheme !== 'untitled') ||
    document.languageId !== 'python'
  ) {
    statusBarItem?.hide();
    return;
  }
  let status: unknown;
  try {
    // The server only reads `uri` from the payload (deserializes as
    // `TextDocumentIdentifier`), so send just that — no need to ship
    // the file text on every status-bar refresh.
    status = await client.sendRequest(
      'pyrefly/textDocument/typeErrorDisplayStatus',
      client.code2ProtocolConverter.asTextDocumentIdentifier(document),
    );
  } catch {
    statusBarItem?.hide();
    return;
  }

  if (!statusBarItem) {
    statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
    );
    statusBarItem.name = 'Pyrefly';
  }

  // Dispatch on response shape. Old servers (or new servers handling
  // an old client that didn't declare a version) return a bare string;
  // new servers handling a new client return `{ version: "v2", ... }`.
  // Renderers return `true` when they wrote a recognized shape into
  // `statusBarItem`; `false` means they couldn't render and the item
  // should stay hidden — important so an unrecognized V1 string doesn't
  // re-show a stale status item below.
  let rendered = false;
  if (typeof status === 'string') {
    rendered = renderV1(status);
  } else if (status != null && typeof status === 'object') {
    const v2 = status as {version?: string};
    if (v2.version === TYPE_ERROR_DISPLAY_STATUS_VERSION) {
      renderV2(status as TypeErrorDisplayStatusV2);
      rendered = true;
    }
    // Unknown future version: server clamping should prevent this in
    // practice; leave `rendered` false so the item is hidden below.
  }
  if (rendered) {
    statusBarItem.show();
  } else {
    statusBarItem.hide();
  }
}

/**
 * V1 renderer: legacy bare-string responses from older binaries. Kept
 * verbatim from the pre-versioning implementation so users on older
 * servers see exactly what they did before.
 *
 * Returns `true` when the status string was recognized and the renderer
 * wrote into `statusBarItem`; `false` for unrecognized strings — the
 * caller hides the item rather than surfacing stale state.
 */
function renderV1(status: string): boolean {
  switch (status) {
    case 'no-config-file':
    case 'disabled-due-to-missing-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is disabled by default.
Create a [\`pyrefly.toml\`](https://pyrefly.org/getting-started) file or set displayTypeErrors to \`force-on\` in settings to show type errors.`);
      return true;
    case 'disabled-in-ide-config':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip =
        new vscode.MarkdownString(`Pyrefly type checking is explicitly disabled.
No errors will be shown even if there is a [\`pyrefly.toml\`](https://pyrefly.org/en/docs/configuration/) file.`);
      return true;
    case 'disabled-in-config-file':
      statusBarItem.text = 'Pyrefly (error-off)';
      statusBarItem.tooltip = new vscode.MarkdownString(
        `Pyrefly type checking is disabled through a config file.`,
      );
      return true;
    case 'enabled-in-ide-config':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is explicitly enabled.\nType errors will always be shown.',
      );
      return true;
    case 'enabled-in-config-file':
      statusBarItem.text = 'Pyrefly';
      statusBarItem.tooltip = new vscode.MarkdownString(
        'Pyrefly type checking is enabled through a config file.',
      );
      return true;
    default:
      return false;
  }
}

/**
 * V2 renderer: server controls the wording. The status bar text is
 * `Pyrefly` plus an optional preset parenthetical; the tooltip is
 * markdown straight from the server.
 */
function renderV2(status: TypeErrorDisplayStatusV2) {
  statusBarItem.text =
    status.label == null ? 'Pyrefly' : `Pyrefly (${status.label})`;
  if (status.tooltip) {
    const md = new vscode.MarkdownString(status.tooltip);
    // The kill-switch and IDE-override tooltips embed
    // `command:workbench.action.openSettings?["<setting-id>"]` links so
    // clicking the setting name jumps the user straight into the
    // Settings UI. `MarkdownString` rejects `command:` URIs unless
    // `isTrusted` allow-lists them — narrow the allow-list to just the
    // one command we use rather than blanket-trusting everything.
    md.isTrusted = {enabledCommands: ['workbench.action.openSettings']};
    if (status.docsUrl) {
      // Render as `Docs: <url>` where <url> is the visible text and
      // also the link target — keeps the URL readable in the hover
      // (and copyable) while still clickable.
      md.appendMarkdown(`\n\nDocs: [${status.docsUrl}](${status.docsUrl})`);
    }
    statusBarItem.tooltip = md;
  } else {
    statusBarItem.tooltip = undefined;
  }
}

export function getStatusBarItem(): vscode.StatusBarItem {
  return statusBarItem;
}
