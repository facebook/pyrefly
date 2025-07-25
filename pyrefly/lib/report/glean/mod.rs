/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::report::glean::facts::Glean;
use crate::state::handle::Handle;
use crate::state::state::Transaction;

pub mod convert;
pub mod facts;
pub mod schema;

pub fn glean(transaction: &Transaction, handle: &Handle) -> String {
    fn f(transaction: &Transaction, handle: &Handle) -> Option<Glean> {
        Some(Glean::new(transaction, handle))
    }

    let data = f(transaction, handle).expect("Glean data be ready");
    serde_json::to_string_pretty(&data).unwrap()
}
