/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::num::NonZeroUsize;

use anstream::ColorChoice;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::init_thread_pool;
use pyrefly_util::trace::init_tracing;

pub fn init_test() {
    ColorChoice::write_global(ColorChoice::Always);
    init_tracing(true, true);
    // Enough threads to see parallelism bugs, but not too many to debug through.
    init_thread_pool(ThreadCount::NumThreads(NonZeroUsize::new(3).unwrap()));
}
