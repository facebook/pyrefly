/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import { useEffect } from 'react';

// Reusable redirect component. The target URL is passed via the route's
// `customProps.url` field in docusaurus.config.ts
export default function ExternalRedirect({ url }: { url: string }): null {
    useEffect(() => {
        window.location.replace(url);
    }, [url]);
    return null;
}
