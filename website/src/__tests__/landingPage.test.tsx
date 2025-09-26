/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import '@testing-library/jest-dom';
import { render } from '@testing-library/react';
import Home from '../pages/index';

describe('Home page', () => {
    test('renders internal site correctly', () => {
        process.env.INTERNAL_STATIC_DOCS = '1';
        const { container } = render(<Home />);

        expectHeaderAndTitleIsCorrect(container);

        // Verify that the new sections are present
        const whyPyreFlySection = document.getElementById(
            'why-pyrefly-section'
        );
        expect(whyPyreFlySection).toBeInTheDocument();

        const performanceComparisonSection = document.getElementById(
            'performance-comparison-section'
        );
        expect(performanceComparisonSection).toBeInTheDocument();

        // Run test with --update-snapshot to update the snapshot if the test is failing after
        // you made a intentional change to the home page
        expect(container).toMatchSnapshot();
    });

    test('renders external site correctly', () => {
        process.env.INTERNAL_STATIC_DOCS = undefined;
        const { container } = render(<Home />);

        expectHeaderAndTitleIsCorrect(container);

        // Verify that the new sections are not present
        const whyPyreFlySection = document.getElementById(
            'why-pyrefly-section'
        );
        expect(whyPyreFlySection).toBeInTheDocument();

        const performanceComparisonSection = document.getElementById(
            'performance-comparison-section'
        );
        expect(performanceComparisonSection).toBeInTheDocument();

        const quotesSection = document.getElementById('quotes-section');
        expect(quotesSection).not.toBeInTheDocument();

        // Run test with --update-snapshot to update the snapshot if the test is failing after
        // you made a intentional change to the home page
        expect(container).toMatchSnapshot();
    });

    function expectHeaderAndTitleIsCorrect(container: HTMLElement) {
        // Verify that title is set correctly
        expect(document.title).toBe(
            'Pyrefly: A Fast Python Type Checker and Language Server'
        );

        // Verify that fireflies are present
        const fireflyElements = document.querySelectorAll('#firefly');
        expect(fireflyElements.length).toBeGreaterThan(1);

        // Verify that header is correct
        const header = container.querySelector('header');
        expect(header).toBeInTheDocument();

        // Verify that the logo image is present
        const logoImg = header?.querySelector(
            'img[src="/img/Pyrefly-Brandmark.svg"]'
        );
        expect(logoImg).toBeInTheDocument();
        expect(logoImg?.getAttribute('alt')).toBe('Pyrefly Logo');

        // verify subtitle is correct
        const subtitle = container.querySelector('header p');
        expect(subtitle).toBeInTheDocument();
        const subtitleText = subtitle?.textContent?.trim();
        expect(subtitleText).toBe(
            'A fast type checker and language server for Python with powerful IDE features'
        );
    }
});
