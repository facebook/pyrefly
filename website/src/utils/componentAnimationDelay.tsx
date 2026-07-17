/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

/**
 * This file creates a map of components that use DelayedComponent,
 * along with their respective time delays. This is calculated using ANIMATION_DELAY_BETWEEN_COMPONENTS_SECONDS to offset
 * the delay of each component.
 *
 * This should be used to manage and coordinate animations across the website.
 */

// Process components in the order they appear in the orderedComponents array
const COMPONENTS_TO_ANIMATE = [
    // Landing Page Header
    'LandingPageHeader.Logo',
    'LandingPageHeader.Subtitle',
    'LandingPageHeader.ButtonGroup',
    // Why Pyrefly sections
    'WhyPyrefly',
    'WhyPyreflyGridItem',
    // IDE Carousel section
    'IDECarousel',
    // AI Confidence section
    'AIConfidence',
    // Performance comparison section
    'PerformanceComparisonChartSectionTitle',
    'PerformanceComparisonChartSection',
    // Video section
    'PyreflyVideoTitle',
    'PyreflyVideo',

    // Other components
    'QuoteCard', // this isn't currently used, but we foresee adding this section in the future
];

type DelayedComponentName = (typeof COMPONENTS_TO_ANIMATE)[number];

/**
 * Record of components that have more than one child and need to be handled differently
 */
const componentsWithMoreThanOneChild: Record<DelayedComponentName, number> = {
    'WhyPyreflyGridItem': 4,
};

const ANIMATION_DELAY_BETWEEN_COMPONENTS_SECONDS = 0.15;

/**
 * Calculate delays automatically based on component configurations and order
 */
function calculateDelays(): Record<DelayedComponentName, number> {
    // Initialize with empty values to satisfy TypeScript
    const delays = {} as Record<DelayedComponentName, number>;
    let currentDelay = 0;


    // First component always starts at 0
    COMPONENTS_TO_ANIMATE.forEach((componentName, index) => {
        const numElements = componentName in componentsWithMoreThanOneChild ? componentsWithMoreThanOneChild[componentName] : 1;

        // First component always starts at 0
        if (index === 0) {
            delays[componentName] = 0;
            currentDelay = 0;
        } else {
            delays[componentName] = currentDelay;
        }

        // Calculate next delay based on delayed children
        currentDelay += ANIMATION_DELAY_BETWEEN_COMPONENTS_SECONDS * numElements;
    });

    return delays;
}

/**
 * A map of component names to their delay times in seconds.
 * Access delays using animationDelaySeconds['componentName'] syntax.
 */
export const animationDelaySeconds: Record<DelayedComponentName, number> = calculateDelays();

/**
 * Helper function to get the delay for a WhyPyreflyGridItem based on its index
 * @param index The index of the grid item
 * @returns The delay in seconds
 */
export function getWhyPyreflyGridItemDelay(index: number): number {
    return animationDelaySeconds['WhyPyreflyGridItem'] + index * 0.1;
}
