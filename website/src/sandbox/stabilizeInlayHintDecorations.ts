/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

interface DecorationOptions {
    description?: string;
    stickiness?: number;
    [key: string]: unknown;
}

interface Decoration {
    id: string;
    options: DecorationOptions;
}

interface DecorationChangeAccessor {
    changeDecorationOptions(id: string, options: DecorationOptions): void;
}

interface InlayHintModel {
    changeDecorations(
        callback: (accessor: DecorationChangeAccessor) => void
    ): void;
    getAllDecorations(): Decoration[];
    onDidChangeDecorations(listener: () => void): unknown;
}

const stabilizedModels = new WeakSet<object>();

/** Keep Monaco's inlay hint anchors from growing when text is inserted at an edge. */
export default function stabilizeInlayHintDecorations(
    model: object,
    stableStickiness: number
): void {
    if (stabilizedModels.has(model)) return;
    stabilizedModels.add(model);
    // Monaco uses this model method internally but does not expose it in ITextModel.
    const inlayHintModel = model as InlayHintModel;

    const stabilize = () => {
        const decorations = inlayHintModel
            .getAllDecorations()
            .filter(
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
