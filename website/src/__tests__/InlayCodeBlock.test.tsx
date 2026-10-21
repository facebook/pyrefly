/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import { render, screen } from '@testing-library/react';
import * as React from 'react';

import InlayCodeBlock from '../components/InlayCodeBlock';

jest.mock('@docusaurus/theme-common', () => ({
    usePrismTheme: () => ({ plain: {}, styles: [] }),
}));

describe('InlayCodeBlock', () => {
    it('renders hints at their source anchors', () => {
        render(
            <InlayCodeBlock
                code={'x = make_tensor()\ny = x'}
                hints={[
                    { line: 1, after: 'x', label: ': Tensor[[B, T]]' },
                    { line: 2, after: 'y', label: ': Tensor[[B, T]]' },
                ]}
            />
        );

        const renderedHints = screen.getAllByLabelText(
            'Inferred type Tensor[[B, T]]'
        );
        expect(renderedHints).toHaveLength(2);
        expect(renderedHints[0].previousSibling?.textContent).toBe('x');
        expect(renderedHints[1].previousSibling?.textContent).toBe('y');
    });
});
