/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import type * as monaco from 'monaco-editor';

type InlayHintDecorationOptions = monaco.editor.IModelDecorationOptions & {
    readonly description?: string;
};

type InlayHintDecoration = Pick<monaco.editor.IModelDecoration, 'id'> & {
    readonly options: InlayHintDecorationOptions;
};

type InlayHintModel = monaco.editor.ITextModel & {
    changeDecorations(
        callback: (accessor: {
            changeDecorationOptions(
                id: string,
                options: monaco.editor.IModelDecorationOptions
            ): void;
        }) => void
    ): void;
};

const stabilizedModels = new WeakSet<monaco.editor.ITextModel>();

/** Keep Monaco's inlay hint anchors from growing when text is inserted at an edge. */
export default function stabilizeInlayHintDecorations(
    model: monaco.editor.ITextModel,
    stableStickiness: number
): void {
    if (stabilizedModels.has(model)) return;
    stabilizedModels.add(model);
    // Monaco uses this model method internally but does not expose it in ITextModel.
    const inlayHintModel = model as InlayHintModel;

    const stabilize = () => {
        const decorations = (
            inlayHintModel.getAllDecorations() as InlayHintDecoration[]
        ).filter(
            (decoration) =>
                decoration.options.description === 'InlayHint' &&
                decoration.options.stickiness !== stableStickiness
        );
        if (decorations.length === 0) return;

        inlayHintModel.changeDecorations((accessor) => {
            decorations.forEach((decoration) =>
                accessor.changeDecorationOptions(decoration.id, {
                    ...decoration.options,
                    stickiness: stableStickiness,
                })
            );
        });
    };

    inlayHintModel.onDidChangeDecorations(stabilize);
    stabilize();
}
