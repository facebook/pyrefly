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
        if let Some(transaction) = state.try_new_committable_transaction(Require::indexing(), None)
        {
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
        self.saved_state
            .take()
            .and_then(|saved_state| saved_state.restore())
            .unwrap_or_else(|| state.transaction())
    }

    /// This function should be called once we finished using transaction for an LSP request.
    pub fn save(&mut self, transaction: Transaction<'a>, telemetry: &mut TelemetryEvent) {
        self.saved_state = Some(transaction.save(telemetry))
    }
}
