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
import ProgressBar from './ProgressBar';
import {
    Project,
    TypeChecker,
    type ProjectValue,
    type TypeCheckerValue,
} from './PerformanceComparisonTypes';

import PerformanceComparisonChartTimer from './PerformanceComparisonChartTimer';
import TypecheckerTooltip from './TypecheckerTooltip';

interface TypeCheckerData {
    typechecker: TypeCheckerValue;
    durationInSeconds: number;
}

interface ProjectData {
    project: ProjectValue;
    data: TypeCheckerData[];
}

interface PerformanceComparisonChartProps {
    project: ProjectValue;
    isLoaded: boolean;
}

export default function PerformanceComparisonChart({
    project,
    isLoaded,
}: PerformanceComparisonChartProps): React.ReactElement {
    const data = getData(project);

    // Calculate the maximum duration for scaling
    const maxDuration = Math.max(...data.map((item) => item.durationInSeconds));

    return (
        <div key={project}>
            {data.map((typechecker, index) => (
                <div
                    {...stylex.props(
                        styles.barContainer,
                        index !== data.length - 1 ? { marginBottom: 20 } : null
                    )}
                    key={index}
                >
                    <div>
                        <div {...stylex.props(styles.typecheckerNameContainer)}>
                            <span {...stylex.props(styles.typecheckerName)}>
                                <strong>{typechecker.typechecker}</strong>
                            </span>
                            <TypecheckerTooltip
                                typechecker={typechecker.typechecker}
                                project={project}
                            />
                        </div>
                    </div>
                    <div {...stylex.props(styles.BarTimerContainer)}>
                        <div {...stylex.props(styles.progressBarContainer)}>
                            <ProgressBar
                                durationInSeconds={
                                    typechecker.durationInSeconds
                                }
                                maxDurationInSeconds={maxDuration}
                                highlight={
                                    typechecker.typechecker ===
                                    TypeChecker.PYREFLY
                                }
                                isLoaded={isLoaded}
                            />
                        </div>
                        <span {...stylex.props(styles.duration)}>
                            <PerformanceComparisonChartTimer
                                targetSeconds={typechecker.durationInSeconds}
                                isLoaded={isLoaded}
                            />
                        </span>
                    </div>
                </div>
            ))}
        </div>
    );
}

const styles = stylex.create({
    barContainer: {
        flex: 1,
        display: 'flex',
        '@media(max-width: 760px)': {
            flexDirection: 'column',
        },
    },
    BarTimerContainer: {
        flex: 2,
        display: 'flex',
    },
    typecheckerNameContainer: {
        display: 'flex',
        alignItems: 'left',
        width: 150,
    },
    typecheckerName: {
        display: 'inline-block',
        fontSize: 20,
    },
    progressBarContainer: {
        flexGrow: 1,
        marginRight: 20,
        paddingTop: '8px',
    },
    duration: {
        marginLeft: 'auto',
        minWidth: '70px', // Ensure enough space for 3 digits + decimal + 's'
        textAlign: 'right',
    },
});

function getData(project: ProjectValue): TypeCheckerData[] {
    const filteredData = performanceComparisonChartData
        .filter((data) => data.project === project)
        .map((data) => data.data);

    if (filteredData.length === 0) {
        throw new Error(`No data found for project ${project}`);
    }

    return filteredData[0];
}

// TODO (T222936871): Add data for pandas and pytorch after we publish Pyrefly and get it to a stable state
const performanceComparisonChartData: ProjectData[] = [
    {
        project: Project.INSTAGRAM,
        data: [
            { typechecker: TypeChecker.PYREFLY, durationInSeconds: 13.36 },
            { typechecker: TypeChecker.PYRE, durationInSeconds: 475.77 },
        ],
    },
    {
        project: Project.PYTORCH,
        data: [
            { typechecker: TypeChecker.PYREFLY, durationInSeconds: 2.32 },
            { typechecker: TypeChecker.PYRIGHT, durationInSeconds: 35.16 },
            { typechecker: TypeChecker.MYPY, durationInSeconds: 48.06 },
        ],
    },
];
