/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import * as stylex from '@stylexjs/stylex';
import PerformanceComparisonChart from './PerformanceComparisonChart';
import PerformanceComparisonDescription from './PerformanceComparisonDescription';
import PerformanceComparisonButton from './PerformanceComparisonButton';
import { Project, ProjectValue } from './PerformanceComparisonTypes';
import { useState, useEffect } from 'react';
import { landingPageCardStyles } from './landingPageCardStyles';

export default function PerformanceComparisonChartSection(): React.ReactElement {
    const [selectedProject, setSelectedProject] = useState<ProjectValue>(
        Project.PYTORCH
    );
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        // Delay the animation to start after the Why Pyrefly section animations
        const timer = setTimeout(() => {
            setIsLoaded(true);
        }, 1200); // slightly later than Performance Comparison Header (1100ms)

        return () => clearTimeout(timer);
    }, []);

    return (
        <div
            {...stylex.props(
                landingPageCardStyles.card,
                styles.body,
                isLoaded && styles.bodyVisible
            )}
        >
            <div
                {...stylex.props(
                    styles.buttonRow,
                    isLoaded && styles.contentVisible
                )}
            >
                <PerformanceComparisonButton
                    project={Project.PYTORCH}
                    selectedProject={selectedProject}
                    setSelectedProject={setSelectedProject}
                />
                <PerformanceComparisonButton
                    project={Project.INSTAGRAM}
                    selectedProject={selectedProject}
                    setSelectedProject={setSelectedProject}
                />
            </div>
            <div
                {...stylex.props(
                    styles.chartContainer,
                    isLoaded && styles.contentVisible
                )}
            >
                <PerformanceComparisonDescription project={selectedProject} />
                <PerformanceComparisonChart
                    project={selectedProject}
                    isLoaded={isLoaded}
                />
            </div>
        </div>
    );
}

const styles = stylex.create({
    body: {
        padding: '1.75rem',
        ':hover': {
            background: 'var(--color-landing-page-card-background)',
        },
        opacity: 0,
        filter: 'blur(8px)',
        transform: 'translateY(20px)',
        transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    bodyVisible: {
        opacity: 1,
        filter: 'blur(0px)',
        transform: 'translateY(0)',
    },
    buttonRow: {
        display: 'flex',
        justifyContent: 'center',
        gap: '1rem',
        marginBottom: '2rem',
        opacity: 0,
        transform: 'translateY(15px)',
        transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    chartContainer: {
        paddingHorizontal: '1rem',
        opacity: 0,
        transform: 'translateY(15px)',
        transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
    contentVisible: {
        opacity: 1,
        transform: 'translateY(0)',
    },
});
