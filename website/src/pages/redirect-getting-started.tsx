/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import {useHistory} from '@docusaurus/router';

// Linked from the LSP status-bar tooltip and the CLI upsell — short,
// memorable URL that lands users on the install/onboarding docs.
//
// Implemented as a `useEffect` + `useHistory().replace` rather than the
// more obvious `<Redirect to=…>` because react-router-dom@5's `Redirect`
// (re-exported by `@docusaurus/router`) ships class-component types
// that fail JSX type-checking against modern React's element types
// (`Property 'refs' is missing in Component<RedirectProps, …>`).
// `useHistory` returns a plain object, sidestepping the upstream typing
// mismatch.
export default function RedirectGettingStarted(): null {
    const history = useHistory();
    React.useEffect(() => {
        history.replace('/en/docs/installation/');
    }, [history]);
    return null;
}
