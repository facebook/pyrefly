/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import Layout from '@theme/Layout';
import PerformanceComparisonChartSection from './PerformanceComparisonChartSection';
import PyreflyVideo from './PyreflyVideo';
import LandingPageSection from './landingPageSection';
import PresetLandingPageHeader from './presetLandingPageHeader';
import StrictnessPresets from './strictnessPresets';
import IDECarousel from './IDECarousel';

// Shared layout behind the preset landing page URLs registered in
// docusaurus.config.ts.
export default function PresetLandingPage(): React.ReactElement {
    return (
        <Layout
            id="preset-landing-page"
            title="Catch Python bugs before you run the code with Pyrefly"
            description="Pyrefly reads your Python and tells you what's broken — a misspelled name, a wrong argument, a missing import. Start with basic mode for minimal noise, then dial up the strictness."
            // These URLs exist only as campaign destinations. They duplicate
            // the home page below the fold, so keep them out of search results
            // rather than competing with pyrefly.org itself.
            noIndex={true}
        >
            <LandingPageSection
                id="header-section"
                child={<PresetLandingPageHeader />}
            />
            <LandingPageSection
                id="strictness-presets-section"
                title="Improve code quality at your own pace"
                child={<StrictnessPresets />}
                isTitleCentered={true}
            />
            <LandingPageSection
                id="ide-carousel-section"
                child={<IDECarousel />}
            />
            <LandingPageSection
                id="performance-comparison-section"
                title="Performance Comparison"
                child={<PerformanceComparisonChartSection />}
            />
            <LandingPageSection
                id="pyrefly-video"
                title="See Pyrefly in Action"
                child={<PyreflyVideo />}
                isLastSection={true}
                isTitleCentered={true}
            />
        </Layout>
    );
}
