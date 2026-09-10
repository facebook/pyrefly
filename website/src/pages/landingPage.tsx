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
import PerformanceComparisonChartSection from '../components/landing-page/PerformanceComparisonChartSection';
import WhyPyrefly from '../components/landing-page/whyPyrefly';
import PyreflyVideo from '../components/landing-page/PyreflyVideo';
import LandingPageSection from '../components/landing-page/landingPageSection';
import LandingPageHeader from '../components/landing-page/landingPageHeader';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function LandingPage(): React.ReactElement {
    const { siteConfig } = useDocusaurusContext();

    return (
        <Layout
            id="new-landing-page"
            title="Pyrefly: A Fast Python Type Checker and Language Server"
            description={siteConfig.description}
        >
            <LandingPageSection
                id="header-section"
                isFirstSection={true}
                child={<LandingPageHeader />}
            />
            <LandingPageSection
                id="why-pyrefly-section"
                child={<WhyPyrefly />}
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
