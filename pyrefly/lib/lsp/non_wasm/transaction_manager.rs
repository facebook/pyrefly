/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_util::telemetry::TelemetryEvent;

use crate::state::require::Require;
use crate::state::state::CommittingTransaction;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::state::state::TransactionData;

/// `TransactionManager` aims to always produce a transaction that contains the up-to-date
/// in-memory contents.
#[derive(Default)]
pub struct TransactionManager<'a> {
    /// Invariant:
    /// If it's None, then the main `State` already contains up-to-date checked content
    /// of all in-memory files.
    /// Otherwise, it will contain up-to-date checked content of all in-memory files.
    saved_state: Option<TransactionData<'a>>,
}

impl<'a> TransactionManager<'a> {
    #[expect(clippy::result_large_err)] // Both results are basically the same size
    /// Produce a possibly committable transaction in order to recheck in-memory files.
    pub fn get_possibly_committable_transaction(
        &mut self,
        state: &'a State,
    ) -> Result<CommittingTransaction<'a>, Transaction<'a>> {
        // If there is no ongoing recheck due to on-disk changes, we should prefer to commit
        // the in-memory changes into the main state.
        if let Some(transaction) = state.try_new_committable_transaction(Require::Exports, None) {
            // If we can commit in-memory changes, then there is no point of holding the
            // non-committable transaction with a possibly outdated view of the `ReadableState`
            // so we can destroy the saved state.
            self.saved_state = None;
            Ok(transaction)
        } else {
            // If there is an ongoing recheck, trying to get a committable transaction will block
            // until the recheck is finished. This is bad for perceived perf. Therefore, we will
            // temporarily use a non-committable transaction to hold the information that's necessary
            // to power IDE services.
            Err(self.non_committable_transaction(state))
        }
    }

    /// Produce a `Transaction` to power readonly IDE services.
    /// This transaction will never be able to be committed.
    /// After using it, the state should be saved by calling the `save` method.
    ///
    /// The `Transaction` will always contain the handles of all open files with the latest content.
    /// It might be created fresh from state, or reused from previously saved state.
    ///
    /// If we were unable to restore a transaction from saved state, we create a fresh transaction.
    /// Callers may need to re-validate open files in this case.
    pub fn non_committable_transaction(&mut self, state: &'a State) -> Transaction<'a> {
        let previous_blocking = match self.saved_state.take() {
            Some(saved_state) => match saved_state.restore() {
                Ok(mut tx) => {
                    // The saved cancellation belonged to the previous consumer and has already
                    // taken effect; clearing it upholds the invariant that a transaction
                    // handed out here can perform work, instead of silently doing none.
                    tx.reset_cancellation();
                    return tx;
                }
                Err(blocked) => Some(blocked),
            },
            None => None,
        };
        let mut tx = state.transaction();
        tx.set_fresh();
        if let Some(d) = previous_blocking {
            tx.add_locked_blocking_duration(d);
        }
        tx
    }

    /// This function should be called once we finished using transaction for an LSP request.
    pub fn save(&mut self, transaction: Transaction<'a>, telemetry: &mut TelemetryEvent) {
        self.saved_state = Some(transaction.save(telemetry))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_util::telemetry::QueueName;
    use pyrefly_util::telemetry::TelemetryEventKind;
    use pyrefly_util::telemetry::TelemetryServerState;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
    use uuid::Uuid;

    use super::*;
    use crate::module::finder::DirEntryCache;
    use crate::module::finder::find_import;
    use crate::test::util::TestEnv;

    /// A recheck cancels the in-flight reads that block its commit. That cancellation belongs
    /// to the request being aborted, so it should *not* survive into the saved transaction.
    #[test]
    fn test_restored_transaction_is_not_still_cancelled() {
        let mut test_env = TestEnv::new();
        test_env.add("first", "x: int = 1\n");
        test_env.add("second", "y: int = \"not an int\"\n");
        let config_file = test_env.config();
        let sys_info = test_env.sys_info();
        let state = State::new(test_env.config_finder(), TEST_THREAD_COUNT);
        let handle = |name: &str| {
            let name = ModuleName::from_str(name);
            let path = find_import(&config_file, name, None, None, &DirEntryCache::new(), None)
                .finding()
                .unwrap();
            Handle::new(name, path, sys_info.dupe())
        };

        let mut manager = TransactionManager::default();
        let mut transaction = manager.non_committable_transaction(&state);
        transaction.set_memory(test_env.get_memory());
        transaction.run(&[handle("first")], Require::Everything, None);
        transaction.get_cancellation_handle().cancel();
        let mut telemetry = TelemetryEvent::new_task(
            TelemetryEventKind::InvalidateConfig,
            TelemetryServerState {
                has_sourcedb: false,
                id: Uuid::new_v4(),
                surface: None,
                server_start_time: Instant::now(),
                agent_session_id: None,
                agent_invocation_id: None,
                active_experiments: Vec::new(),
            },
            QueueName::RecheckQueue,
            0,
            Instant::now(),
        );
        manager.save(transaction, &mut telemetry);

        let second = handle("second");
        let mut transaction = manager.non_committable_transaction(&state);
        transaction.set_memory(test_env.get_memory());
        transaction.run(&[second.dupe()], Require::Everything, None);
        assert_eq!(
            transaction
                .get_errors([&second])
                .collect_errors()
                .ordinary
                .len(),
            1
        );
    }
}
