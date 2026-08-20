/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the `typeServer/getExpectedType` TSP request.

use lsp_server::ResponseError;
use tsp_types::GetTypeParams;
use tsp_types::Type;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspConnection;
use crate::tsp::validation::parse_uri;

impl<T: TspInterface> TspConnection<T> {
    /// Return the expected type at the given position.
    ///
    /// The expected type is the type that a surrounding context demands.
    /// For example, in `foo(4)` where `def foo(a: int | str)`, the expected
    /// type of the argument `4` is `int | str`. Where no expected-type context
    /// applies, this falls back to the computed type at the position.
    pub fn handle_get_expected_type(
        &self,
        params: GetTypeParams,
    ) -> Result<Option<Type>, ResponseError> {
        self.validate_snapshot(params.snapshot)?;
        // Validate the URI is parseable (rejects malformed strings).
        // Any valid scheme is accepted — notebook cell URIs are resolved
        // to notebook paths inside expected_type_at_position.
        parse_uri(params.uri())?;
        let position = params.position();
        Ok(self
            .inner()
            .expected_type_at_position(params.uri(), position.line, position.character))
    }
}
