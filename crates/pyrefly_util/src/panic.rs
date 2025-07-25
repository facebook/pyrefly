/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::backtrace::Backtrace;
use std::panic::PanicHookInfo;
use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use tracing::error;
use yansi::Paint;

static HAS_PANICKED: AtomicBool = AtomicBool::new(false);

/// Returns true if Pyrefly has panicked up to this point.
pub fn has_panicked() -> bool {
    HAS_PANICKED.load(Ordering::Relaxed)
}

/// The code that Rust uses for panics.
pub const PANIC_EXIT_CODE: u8 = 101;

pub fn print_panic(info: &PanicHookInfo<'_>) {
    HAS_PANICKED.store(true, Ordering::Relaxed);

    // Sometimes we get two simultaneous panics, and there output gets co-mingled.
    // Make sure we only report one panic.
    static PANIC_LOCK: Once = Once::new();

    PANIC_LOCK.call_once(|| {
        error!(
            "Thread panicked, shutting down: {info}\nBacktrace:\n{}",
            Backtrace::force_capture()
        );

        let out = |x: &str| anstream::eprintln!("{} {x}", Paint::magenta("PANIC"));

        out("Sorry, Pyrefly crashed, this is always a bug in Pyrefly itself.");
        if cfg!(fbcode_build) {
            out("Please report the bug at https://fb.workplace.com/groups/pyreqa");
        } else {
            out("Please report the bug at https://github.com/facebook/pyrefly/issues/new")
        }
    });
}

/// When Pyrefly panics, we want to print a backtrace and exit.
///
/// We want to exit immediately because otherwise we'd have to ensure all our thread/lock code
/// properly bubbled up the panic, without producing a deadlock, or another panic, which is
/// a) a lot of work, and b) almost impossible to test.
pub fn exit_on_panic() {
    std::panic::set_hook(Box::new(move |info| {
        print_panic(info);
        std::process::exit(PANIC_EXIT_CODE as i32);
    }));
}
