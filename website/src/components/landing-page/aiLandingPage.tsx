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
import AiLandingPageHeader from './aiLandingPageHeader';
import IDECarousel from './IDECarousel';

// Shared layout behind the AI landing page URLs registered in
// docusaurus.config.ts.
export default function AiLandingPage(): React.ReactElement {
    return (
        <Layout
            id="ai-landing-page"
            title="Verify AI-written Python with Pyrefly"
            description="Catch type bugs in code your AI agent writes. Pyrefly is an open-source Python type checker built in Rust, fast enough to keep up with your agent's inference loop."
            // These URLs exist only as campaign destinations. They duplicate
            // the home page below the fold, so keep them out of search results
            // rather than competing with pyrefly.org itself.
            noIndex={true}
        >
            <LandingPageSection
                id="header-section"
                child={<AiLandingPageHeader />}
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
