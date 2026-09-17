/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A wall-clock timer for internal profiling counters that can be globally
//! disabled.
//!
//! `Instant::now()` is normally free (read via the vDSO), but under
//! instrumentation that bypasses the vDSO each reading is a real `clock_gettime`
//! syscall, and a single type-check makes dozens. Callers that don't need the
//! counters disable timing so every [`Timer`] becomes a no-op; it is enabled by
//! default so telemetry stays populated.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use web_time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable all [`Timer`]s process-wide. Enabled by default; a disabled
/// timer never calls `Instant::now()` and reports zero elapsed time.
pub fn set_timing_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// A profiling timer that captures a start instant only when timing is enabled.
/// When disabled, `elapsed*` return zero and no `clock_gettime` syscall is made.
#[derive(Debug, Clone, Copy)]
pub struct Timer(Option<Instant>);

impl Timer {
    /// Start a timer, reading the clock only if timing is enabled.
    pub fn start() -> Self {
        Self(ENABLED.load(Ordering::Relaxed).then(Instant::now))
    }

    pub fn elapsed(&self) -> Duration {
        self.0.map_or(Duration::ZERO, |start| start.elapsed())
    }

    pub fn elapsed_nanos(&self) -> u64 {
        self.0.map_or(0, |start| start.elapsed().as_nanos() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_toggle_controls_the_clock() {
        // Enabled by default: the timer captures a start instant.
        assert!(
            Timer::start().0.is_some(),
            "timing should be enabled by default"
        );

        // Disabled: no clock is read and elapsed time is zero.
        set_timing_enabled(false);
        let disabled = Timer::start();
        assert!(
            disabled.0.is_none(),
            "a disabled timer must not read the clock"
        );
        assert_eq!(disabled.elapsed(), Duration::ZERO);
        assert_eq!(disabled.elapsed_nanos(), 0);

        // Restore the global so other tests observe the default.
        set_timing_enabled(true);
        assert!(Timer::start().0.is_some());
    }
}
