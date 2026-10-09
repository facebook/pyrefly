/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import NavbarItem from '@theme-original/NavbarItem';
import { log, LoggingEvent } from '../../utils/LoggingUtils';

type Props = React.ComponentProps<'a'> & { label?: string };

/**
 * Wraps the default navbar item to log a click event for each top nav tab.
 */
export default function NavbarItemWrapper(props: Props): React.ReactElement {
    // Icon-only items (GitHub, Discord) have an aria-label instead of a label.
    const name = props.label ?? props['aria-label'];
    if (name == null) {
        return <NavbarItem {...props} />;
    }
    return (
        <NavbarItem
            {...props}
            onClick={(e) => {
                log(LoggingEvent.CLICK, {
                    button_id: `tab: ${name.toLowerCase()}`,
                });
                props.onClick?.(e);
            }}
        />
    );
}
