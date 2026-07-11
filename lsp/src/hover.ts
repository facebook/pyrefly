/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as vscode from 'vscode';
import {
  HoverParams,
  HoverRequest,
  LanguageClient,
} from 'vscode-languageclient/node';

type HoverWithVerbosity = HoverParams & {
  verbosityLevel: number;
};

/** Register the VS Code-specific hover provider that supports +/- controls. */
export function registerHoverProvider(
  context: vscode.ExtensionContext,
  getClient: () => LanguageClient,
): void {
  let lastHoverAndLevel: [vscode.Hover, number] | undefined;

  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      [
        {scheme: 'file', language: 'python'},
        {scheme: 'untitled', language: 'python'},
        {scheme: 'vscode-notebook-cell', language: 'python'},
        {scheme: 'inmemory', language: 'python'},
      ],
      ({
        async provideHover(
          document: vscode.TextDocument,
          position: vscode.Position,
          token: vscode.CancellationToken,
          hoverContext?: vscode.HoverContext,
        ) {
          const client = getClient();
          const previousLevel =
            hoverContext?.previousHover &&
            lastHoverAndLevel?.[0] === hoverContext.previousHover
              ? lastHoverAndLevel![1]
              : 0;
          const verbosityLevel = Math.max(
            0,
            previousLevel + (hoverContext?.verbosityDelta ?? 0),
          );
          const params = {
            ...client.code2ProtocolConverter.asTextDocumentPositionParams(
              document,
              position,
            ),
            verbosityLevel,
          } as HoverWithVerbosity;
          const result = await client.sendRequest(
            HoverRequest.type,
            params,
            token,
          );
          if (!result || token.isCancellationRequested) {
            return undefined;
          }

          const canIncreaseVerbosity =
            (result as typeof result & {canIncreaseVerbosity?: boolean})
              .canIncreaseVerbosity ?? false;
          const hover = client.protocol2CodeConverter.asHover(result);
          if (!hover) {
            return undefined;
          }
          const verboseHover = new vscode.VerboseHover(
            hover.contents,
            hover.range,
            canIncreaseVerbosity,
            verbosityLevel > 0,
          );
          lastHoverAndLevel = [verboseHover, verbosityLevel];
          return verboseHover;
        },
      } as unknown as vscode.HoverProvider),
    ),
  );
}
