/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import stabilizeInlayHintDecorations from '../sandbox/stabilizeInlayHintDecorations';

test('prevents Monaco inlay hint decorations from growing at their edges', () => {
    const changeDecorationOptions = jest.fn();
    let decorations: Array<{
        id: string;
        options: { description: string; stickiness: number; after?: object };
    }> = [];
    let decorationsChanged = () => {};
    const inlayHintOptions = {
        description: 'InlayHint',
        stickiness: 0,
        after: { content: ' -> int' },
    };
    const model = {
        changeDecorations: (
            callback: (accessor: {
                changeDecorationOptions: typeof changeDecorationOptions;
            }) => void
        ) =>
            callback({ changeDecorationOptions }),
        getAllDecorations: () => decorations,
        onDidChangeDecorations: jest.fn((listener: () => void) => {
            decorationsChanged = listener;
        }),
    };

    stabilizeInlayHintDecorations(model, 1);
    decorations = [
        { id: 'inlay-hint', options: inlayHintOptions },
        { id: 'other', options: { description: 'other', stickiness: 0 } },
    ];
    decorationsChanged();

    expect(changeDecorationOptions).toHaveBeenCalledWith('inlay-hint', {
        ...inlayHintOptions,
        stickiness: 1,
    });
    expect(changeDecorationOptions).toHaveBeenCalledTimes(1);
});
