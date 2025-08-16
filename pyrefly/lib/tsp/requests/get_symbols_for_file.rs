/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Implementation of the getSymbolsForFile TSP request

use lsp_server::ResponseError;
use pyrefly_python::module::Module;
use tsp_types as tsp;

use crate::state::state::Transaction;
use crate::tsp::server::TspServer;

impl TspServer {
    pub fn get_symbols_for_file(
        &self,
        transaction: &Transaction<'_>,
        params: tsp::GetSymbolsForFileParams,
    ) -> Result<Option<tsp::FileSymbolInfo>, ResponseError> {
        // Use common helper to validate, get handle, module info and maybe a fresh transaction
        let file_url = lsp_types::Url::parse(&params.uri).map_err(|_| ResponseError {
            code: lsp_server::ErrorCode::InvalidParams as i32,
            message: "Invalid uri".to_owned(),
            data: None,
        })?;
        let (handle, module_info, transaction_to_use) = self.with_active_transaction(
            transaction,
            &file_url,
            params.snapshot,
            crate::state::require::Require::Everything,
        )?;

        // Use the appropriate transaction for the rest of the function
        let active_transaction = transaction_to_use.as_ref().unwrap_or(transaction);

        // Get all symbols from the file using the existing symbols() method
        let document_symbols = active_transaction.symbols(&handle).unwrap_or_default();

        // Convert from LSP DocumentSymbol to TSP Symbol
        let mut tsp_symbols = Vec::new();
        Self::convert_document_symbols_to_tsp(
            &document_symbols,
            &mut tsp_symbols,
            &file_url,
            &module_info,
            module_info.name().as_str(),
        );

        Ok(Some(tsp::FileSymbolInfo {
            uri: params.uri,
            symbols: tsp_symbols,
        }))
    }

    /// Convert LSP DocumentSymbols to TSP Symbols recursively
    fn convert_document_symbols_to_tsp(
        document_symbols: &[lsp_types::DocumentSymbol],
        tsp_symbols: &mut Vec<tsp::Symbol>,
        uri: &lsp_types::Url,
        _module_info: &Module,
        module_name_str: &str,
    ) {
        for doc_symbol in document_symbols {
            // Create the TSP Symbol from DocumentSymbol
            let tsp_symbol = Self::document_symbol_to_tsp_symbol(doc_symbol, uri, module_name_str);
            tsp_symbols.push(tsp_symbol);

            // Recursively convert children if they exist
            if let Some(ref children) = doc_symbol.children {
                Self::convert_document_symbols_to_tsp(
                    children,
                    tsp_symbols,
                    uri,
                    _module_info,
                    module_name_str,
                );
            }
        }
    }

    /// Convert a single DocumentSymbol to a TSP Symbol
    fn document_symbol_to_tsp_symbol(
        doc_symbol: &lsp_types::DocumentSymbol,
        uri: &lsp_types::Url,
        module_name_str: &str,
    ) -> tsp::Symbol {
        // Create a Node from the symbol's selection range
        let node = tsp::Node {
            uri: uri.to_string(),
            range: tsp_types::from_lsp_range(doc_symbol.selection_range),
        };

        // Convert LSP SymbolKind to TSP DeclarationCategory using common function
        let (category, flags) = Self::lsp_symbol_kind_to_tsp_category(doc_symbol.kind);

        // Parse module name into ModuleName struct using common function
        let module_name = Self::create_tsp_module_name(module_name_str);

        // Create a declaration for this symbol via builder
        let declaration = crate::tsp::requests::common::DeclarationBuilder::new(
            doc_symbol.name.clone(),
            module_name,
            uri.clone(),
        )
        .handle_str(format!(
            "symbol_{}_{}_{}",
            doc_symbol.name,
            doc_symbol.selection_range.start.line,
            doc_symbol.selection_range.start.character
        ))
        .category(category)
        .flags(flags)
        .node(node.clone())
        .build();

        tsp::Symbol {
            node,
            name: doc_symbol.name.clone(),
            decls: vec![declaration],
            synthesized_types: Vec::new(), // TODO: Could be enhanced to include type information
        }
    }
}
