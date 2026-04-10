/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Instant;

pub struct StepLogger {
    start: Instant,
    step_done_message: String,
}

impl StepLogger {
    pub fn start(step_start_message: &str, step_done_message: &str) -> Self {
        tracing::debug!("{}...", step_start_message);
        Self {
            start: Instant::now(),
            step_done_message: step_done_message.to_owned(),
        }
    }

    pub fn finish(&self) {
        tracing::debug!(
            "{} in {:.3}s",
            self.step_done_message,
            self.start.elapsed().as_secs_f32()
        );
    }
}
