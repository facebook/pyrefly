/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::backtrace::Backtrace;
use std::sync::Once;

use tracing::error;

pub fn exit_on_panic() {
    // Sometimes we get two simultaneous panics, and there output gets co-mingled.
    // Make sure we only report one panic.
    static PANIC_LOCK: Once = Once::new();

    std::panic::set_hook(Box::new(move |info| {
        PANIC_LOCK.call_once(|| {
            error!(
                "Thread panicked, shutting down: {info}\nBacktrace:\n{}",
                Backtrace::force_capture()
            );
            eprintln!("Sorry, Pyrefly crashed, this is always a bug in Pyrefly itself.");
            if cfg!(fbcode_build) {
                eprintln!("Please report the bug at https://fb.workplace.com/groups/pyreqa");
            } else {
                eprintln!("Please report the bug at https://github.com/facebook/pyrefly/issues/new")
            }
        });
        std::process::exit(1);
    }));
}
