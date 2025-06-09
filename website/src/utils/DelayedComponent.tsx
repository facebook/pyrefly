/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';

interface DelayedComponentProps {
    /**
     * The delay in seconds before rendering the children
     */
    delayInSeconds: number;

    /**
     * The content to render after the delay
     */
    children: React.ReactNode | ((isLoaded: boolean) => React.ReactNode);
}

/**
 * A wrapper component that delays the rendering of its children.
 * This is useful for creating staggered animations or delayed rendering effects.
 *
 * @example
 * // Render WhyPyrefly component after a 0.5 second delay
 * <DelayedComponent delayInSeconds={0.5}>
 *   <WhyPyrefly />
 * </DelayedComponent>
 *
 * @example
 * // Render PerformanceComparisonChartSection after a 1.2 second delay
 * // and pass the isLoaded state to it
 * <DelayedComponent delayInSeconds={1.2}>
 *   {(isLoaded) => <PerformanceComparisonChartSection isLoaded={isLoaded} />}
 * </DelayedComponent>
 */
export class DelayedComponent extends React.Component<
    DelayedComponentProps,
    { isLoaded: boolean }
> {
    private timer: NodeJS.Timeout | null = null;

    constructor(props: DelayedComponentProps) {
        super(props);
        this.state = {
            isLoaded: false,
        };
    }

    componentDidMount() {
        const { delayInSeconds } = this.props;
        const delayInMs = delayInSeconds * 1000;

        this.timer = setTimeout(() => {
            this.setState({ isLoaded: true });
        }, delayInMs);
    }

    componentWillUnmount() {
        if (this.timer) {
            clearTimeout(this.timer);
        }
    }

    render() {
        const { children } = this.props;
        const { isLoaded } = this.state;

        // If children is a function, call it with the isLoaded state
        if (typeof children === 'function') {
            return (
                <div style={{ opacity: isLoaded ? 1 : 0 }}>
                    {children(isLoaded)}
                </div>
            );
        }

        // Otherwise, just render the children directly with opacity based on isLoaded
        return <div style={{ opacity: isLoaded ? 1 : 0 }}>{children}</div>;
    }
}

export default DelayedComponent;
