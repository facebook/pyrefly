/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the `typeServer/getDeclaredType` TSP request.

use lsp_server::ResponseError;
use tsp_types::GetTypeParams;
use tsp_types::Type;

use crate::lsp::non_wasm::server::TspInterface;
use crate::tsp::server::TspConnection;
use crate::tsp::validation::parse_uri;

impl<T: TspInterface> TspConnection<T> {
    /// Return the declared type at the given position.
    ///
    /// The declared type is the annotation explicitly written by the user.
    /// For example, `a: int | str` has declared type `int | str` even if
    /// type narrowing later restricts the computed type to `int`.
    ///
    /// Currently piggy-backs on `get_type_at_position`, which returns the
    /// computed type. A future improvement can separate the annotation type
    /// from the inferred type in the binding infrastructure.
    pub fn handle_get_declared_type(
        &self,
        params: GetTypeParams,
    ) -> Result<Option<Type>, ResponseError> {
        self.validate_snapshot(params.snapshot)?;
        // Validate the URI is parseable (rejects malformed strings).
        // Any valid scheme is accepted — notebook cell URIs are resolved
        // to notebook paths inside get_type_at_position.
        parse_uri(params.uri())?;
        let position = params.position();
        let ty = self
            .inner()
            .get_type_at_position(params.uri(), position.line, position.character);
        Ok(ty.map(|t| self.convert_type(&t, Some(params.uri()))))
    }
}
