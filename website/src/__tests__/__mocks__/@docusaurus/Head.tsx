/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import * as React from 'react';

// Mock for @docusaurus/Head. The real component portals its children into the
// document <head>; rendering them inline as a fragment is enough for tests.
export default function Head({
    children,
}: {
    children: React.ReactNode;
}): React.JSX.Element {
    return <>{children}</>;
}
