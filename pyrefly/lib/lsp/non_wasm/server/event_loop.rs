/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::time::Instant;

use pyrefly_util::telemetry::QueueName;
use pyrefly_util::telemetry::Telemetry;
use pyrefly_util::telemetry::TelemetryEvent;
use pyrefly_util::telemetry::TelemetryEventKind;
use tracing::error;
use tracing::info;

use super::ProcessEvent;
use super::Server;
use crate::lsp::non_wasm::transaction_manager::TransactionManager;

pub(super) fn run(server: &Server, telemetry: &dyn Telemetry, lsp_start_time: Instant) {
    let mut ide_transaction_manager = TransactionManager::default();
    let mut canceled_requests = HashSet::new();
    // Start at 1 because task_id 0 is used by the startup event below.
    let mut next_task_id = 1_usize;
    TelemetryEvent::new_task(
        TelemetryEventKind::LspStartup,
        server.telemetry_state(),
        QueueName::LspQueue,
        0,
        lsp_start_time,
    )
    .finish_and_record(telemetry, None);
    while let Ok(event) = server.lsp_queue.recv() {
        let subsequent_mutation = server.lsp_queue.has_subsequent_mutation(&event);
        let task_id = next_task_id;
        next_task_id += 1;
        let (mut event_telemetry, queue_duration) = TelemetryEvent::new_dequeued(
            TelemetryEventKind::LspEvent(event.describe()),
            event.enqueued_at(),
            server.telemetry_state(),
            QueueName::LspQueue,
            task_id,
        );
        let event_description = event.describe();
        let result = server.process_event(
            &mut ide_transaction_manager,
            &mut canceled_requests,
            telemetry,
            &mut event_telemetry,
            subsequent_mutation,
            event,
        );
        let process_duration = event_telemetry.finish_and_record(telemetry, result.as_ref().err());
        match result {
            Ok(ProcessEvent::Continue) => {
                info!(
                    "Language server processed event `{}` in {:.2}s ({:.2}s waiting)",
                    event_description,
                    process_duration.as_secs_f32(),
                    queue_duration.as_secs_f32()
                );
            }
            Ok(ProcessEvent::Exit) => break,
            Err(e) => {
                // Log the error and continue processing the next event
                error!("Error processing event `{}`: {:?}", event_description, e);
            }
        }
    }
    info!("waiting for connection to close");
    server.recheck_queue.stop();
    server.find_reference_queue.stop();
    server.sourcedb_queue.stop();
}
