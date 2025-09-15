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
import Banner from '../components/Banner';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { log, LoggingEvent } from '../utils/LoggingUtils';

export default function LandingPage(): React.ReactElement {
    const { siteConfig } = useDocusaurusContext();

    return (
        <Layout
            id="new-landing-page"
            title="Pyrefly: A Static Type Checker for Python"
            description={siteConfig.description}
        >
            <Banner
                text="📝 The 2025 Python Typing and Tooling Survey is now open! Help shape the future of typed Python by taking the survey"
                variant="tertiary"
                cta={{
                    text: "Take the Survey",
                    href: "https://forms.gle/6d9WxYEF4zNGLvUq5",
                    external: true,
                    onClick: () =>log(LoggingEvent.CLICK, {
                        button_id: 'take_typing_survey_2025',
                        })
                }}
                dismissible={true}
            />
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
