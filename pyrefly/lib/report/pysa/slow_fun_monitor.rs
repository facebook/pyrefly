/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use dashmap::DashMap;
use pyrefly_util::lock::Condvar;
use pyrefly_util::lock::Mutex;
use tracing::info;

// Represents a request to monitor a function.
struct MonitorEntry {
    message: String,
    start: Instant,
    max_time_in_seconds: u64,
}

// Handle to start monitoring functions using `::monitor_function()`
pub struct SlowFunctionMonitor {
    next_id: Arc<AtomicU64>,
    entries: Arc<DashMap<u64, MonitorEntry>>,
}

impl SlowFunctionMonitor {
    /// Execute the given function `f` and monitor its execution time.
    /// Warn when the function takes more than `max_time_in_seconds`,
    /// then `2 * max_time_in_seconds`, etc.
    pub fn monitor_function<F, R>(&self, f: F, message: String, max_time_in_seconds: u64) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let monitor_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.entries.insert(
            monitor_id,
            MonitorEntry {
                message,
                start,
                max_time_in_seconds,
            },
        );

        let result = f();

        self.entries.remove(&monitor_id);

        result
    }
}

/// Calls the given function and provides it a handle to a SlowFunctionMonitor,
/// which can be used to call `monitor_function()` to monitor slow functions.
/// Internally, this uses a single thread with low overhead.
pub fn slow_fun_monitor_scope<F, R>(f: F) -> R
where
    F: FnOnce(&SlowFunctionMonitor) -> R,
{
    thread::scope(|s| {
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let done_clone = Arc::clone(&done);
        let entries = Arc::new(DashMap::<u64, MonitorEntry>::new());
        let entries_clone = Arc::clone(&entries);

        // Monitoring thread, which wakes up every 500ms and checks if a monitored
        // function reached the threshold.
        let monitoring_thread = s.spawn(move || {
            loop {
                // Using a condvar, so that we can wake up immediately when the
                // monitoring scope ends. Using a traditional atomic boolean would
                // lead us to wait an extra 500ms at the end.
                let (done_mutex, done_cvar) = &*done;
                let done_lock = done_mutex.lock();
                let wait_result = done_cvar.wait_timeout(done_lock, Duration::from_millis(500));

                if *wait_result.0 {
                    break;
                }

                for mut entry in entries.iter_mut() {
                    let entry = entry.value_mut();
                    if entry.start.elapsed().as_secs() >= entry.max_time_in_seconds {
                        info!(
                            "{} is taking more than {} seconds!",
                            entry.message, entry.max_time_in_seconds
                        );
                        entry.max_time_in_seconds *= 2;
                    }
                }
            }
        });

        let next_id = Arc::new(AtomicU64::new(0));
        let monitor = SlowFunctionMonitor {
            next_id,
            entries: entries_clone,
        };

        let result = f(&monitor);

        {
            let (done_mutex, done_cvar) = &*done_clone;
            let mut done_lock = done_mutex.lock();
            *done_lock = true;
            done_cvar.notify_one(); // Wake up the monitoring thread.
        }
        monitoring_thread.join().unwrap();

        result
    })
}
