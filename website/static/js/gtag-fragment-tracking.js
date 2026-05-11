/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// GA4 strips URL fragments by default. Override page_location on initial load
// and hash changes so we can track which error-kind anchor users navigate to.
(function () {
  function sendHashPageView() {
    if (typeof gtag === 'function' && window.location.hash) {
      gtag('event', 'page_view', {
        page_location: window.location.href,
      });
    }
  }

  // Track hash changes (anchor clicks, SPA navigation).
  window.addEventListener('hashchange', sendHashPageView);

  // Track the initial fragment if the page loads with one.
  if (window.location.hash) {
    // Defer so the gtag snippet has time to initialize.
    window.addEventListener('load', sendHashPageView);
  }
})();
