/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const marker = process.env.PYREFLY_EXTENSION_TEST_STARTED;
if (marker) {
	require('node:fs').writeFileSync(marker, '');
}
