/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anstream::ColorChoice;
pub use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use pyrefly_util::trace::init_tracing;

pub fn init_test() {
    ColorChoice::write_global(ColorChoice::Always);
    init_tracing(true, true);
}
