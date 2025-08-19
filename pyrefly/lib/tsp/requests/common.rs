/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Common utilities shared between TSP request implementations

use lsp_server::ErrorCode;
use lsp_server::ResponseError;
use ruff_text_size::TextSize;
use tsp_types as tsp;
use tsp_types::snapshot_outdated_error;

use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::require::Require;
use crate::state::state::Transaction;
use crate::tsp::server::TspServer;
/// Common validation for snapshot
impl TspServer {
    pub fn validate_snapshot(&self, snapshot: i32) -> Result<(), ResponseError> {
        if snapshot != self.current_snapshot() {
            return Err(snapshot_outdated_error());
        }
        Ok(())
    }

    /// Common validation for language services enablement
    pub fn validate_language_services(&self, uri: &lsp_types::Url) -> Result<(), ResponseError> {
        if self.inner.make_handle_if_enabled(uri).is_none() {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Language services disabled".to_owned(),
                data: None,
            });
        }
        Ok(())
    }

    /// Convert from pyrefly symbol kind to TSP declaration category and flags
    pub fn symbol_kind_to_tsp_category(
        symbol_kind: pyrefly_python::symbol_kind::SymbolKind,
    ) -> (tsp::DeclarationCategory, tsp::DeclarationFlags) {
        match symbol_kind {
            pyrefly_python::symbol_kind::SymbolKind::Function => (
                tsp::DeclarationCategory::Function,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::Class => (
                tsp::DeclarationCategory::Class,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::Variable => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::Constant => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new().with_constant(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::Parameter => (
                tsp::DeclarationCategory::Param,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::TypeParameter => (
                tsp::DeclarationCategory::TypeParam,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::TypeAlias => (
                tsp::DeclarationCategory::TypeAlias,
                tsp::DeclarationFlags::new(),
            ),
            pyrefly_python::symbol_kind::SymbolKind::Attribute => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new().with_class_member(),
            ),
            _ => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new(),
            ),
        }
    }

    /// Convert from pyrefly symbol kind to TSP declaration category and flags with class member flag
    pub fn symbol_kind_to_tsp_category_with_class_member(
        symbol_kind: pyrefly_python::symbol_kind::SymbolKind,
    ) -> (tsp::DeclarationCategory, tsp::DeclarationFlags) {
        let (category, mut flags) = Self::symbol_kind_to_tsp_category(symbol_kind);

        // Add class member flag for appropriate categories
        match category {
            tsp::DeclarationCategory::Function | tsp::DeclarationCategory::Variable => {
                flags = flags.with_class_member();
            }
            _ => {}
        }

        (category, flags)
    }

    /// Convert LSP SymbolKind to TSP DeclarationCategory and flags
    pub fn lsp_symbol_kind_to_tsp_category(
        kind: lsp_types::SymbolKind,
    ) -> (tsp::DeclarationCategory, tsp::DeclarationFlags) {
        match kind {
            lsp_types::SymbolKind::FUNCTION | lsp_types::SymbolKind::METHOD => (
                tsp::DeclarationCategory::Function,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::CLASS => (
                tsp::DeclarationCategory::Class,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::VARIABLE | lsp_types::SymbolKind::FIELD => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::CONSTANT => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new().with_constant(),
            ),
            lsp_types::SymbolKind::MODULE | lsp_types::SymbolKind::NAMESPACE => (
                tsp::DeclarationCategory::Import,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::CONSTRUCTOR => (
                tsp::DeclarationCategory::Function,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::PROPERTY => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new().with_class_member(),
            ),
            lsp_types::SymbolKind::ENUM | lsp_types::SymbolKind::ENUM_MEMBER => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new().with_constant(),
            ),
            lsp_types::SymbolKind::INTERFACE => (
                tsp::DeclarationCategory::Class,
                tsp::DeclarationFlags::new(),
            ),
            lsp_types::SymbolKind::TYPE_PARAMETER => (
                tsp::DeclarationCategory::TypeParam,
                tsp::DeclarationFlags::new(),
            ),
            // Default to variable for other kinds
            _ => (
                tsp::DeclarationCategory::Variable,
                tsp::DeclarationFlags::new(),
            ),
        }
    }

    /// Create a TSP ModuleName from a module name string
    pub fn create_tsp_module_name(module_name_str: &str) -> tsp::ModuleName {
        let module_parts: Vec<String> = module_name_str.split('.').map(|s| s.to_owned()).collect();
        tsp::ModuleName {
            leading_dots: 0,
            name_parts: module_parts,
        }
    }

    /// Check if a module is from builtins and update category accordingly
    pub fn apply_builtins_category(
        category: tsp::DeclarationCategory,
        flags: tsp::DeclarationFlags,
        module_name: &tsp::ModuleName,
    ) -> (tsp::DeclarationCategory, tsp::DeclarationFlags) {
        if module_name
            .name_parts
            .first()
            .is_some_and(|first| first == "builtins")
        {
            match category {
                tsp::DeclarationCategory::Function
                | tsp::DeclarationCategory::Class
                | tsp::DeclarationCategory::Variable => {
                    (tsp::DeclarationCategory::Intrinsic, flags)
                }
                _ => (category, flags),
            }
        } else {
            (category, flags)
        }
    }

    /// Get or load module info from the current transaction or load it at the required level.
    /// Returns the ModuleInfo and, if a fresh transaction was needed, that Transaction.
    pub fn get_or_load_module_info(
        &self,
        transaction: &Transaction<'_>,
        handle: &Handle,
        required_level: Require,
    ) -> Result<(ModuleInfo, Option<Transaction>), ResponseError> {
        if let Some(info) = transaction.get_module_info(handle) {
            return Ok((info, None));
        }

        let Some(fresh_transaction) =
            self.load_module_if_needed(transaction, handle, required_level)
        else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Failed to load module".to_owned(),
                data: None,
            });
        };

        let Some(info) = fresh_transaction.get_module_info(handle) else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Failed to get module info after loading".to_owned(),
                data: None,
            });
        };

        Ok((info, Some(fresh_transaction)))
    }

    /// Validate snapshot and language services, obtain Handle and ModuleInfo, and an optional
    /// fresh Transaction if the module needed loading. Encapsulates common handler boilerplate.
    pub fn with_active_transaction(
        &self,
        transaction: &Transaction<'_>,
        uri: &lsp_types::Url,
        snapshot: i32,
        required_level: Require,
    ) -> Result<(Handle, ModuleInfo, Option<Transaction>), ResponseError> {
        // Snapshot validation
        self.validate_snapshot(snapshot)?;

        // Language services validation
        self.validate_language_services(uri)?;

        // Create handle (after validation we expect it to be enabled)
        let Some(handle) = self.inner.make_handle_if_enabled(uri) else {
            return Err(ResponseError {
                code: ErrorCode::RequestFailed as i32,
                message: "Language services disabled".to_owned(),
                data: None,
            });
        };

        // Get or load module info
        let (module_info, maybe_fresh_tx) =
            self.get_or_load_module_info(transaction, &handle, required_level)?;

        Ok((handle, module_info, maybe_fresh_tx))
    }

    /// Load a module in a fresh transaction if it's not available in the current transaction
    /// or if it doesn't meet the required level
    pub fn load_module_if_needed(
        &self,
        transaction: &Transaction<'_>,
        handle: &crate::state::handle::Handle,
        required_level: crate::state::require::Require,
    ) -> Option<Transaction> {
        // Check if module is already loaded in the current transaction
        // For now, we'll be conservative and always reload if Everything is required
        // but the module exists (since we can't easily check the current requirement level)
        let should_reload = match required_level {
            crate::state::require::Require::Everything => {
                // For Everything requirement, reload even if module exists to ensure full analysis
                transaction.get_module_info(handle).is_some()
            }
            _ => {
                // For lighter requirements, don't reload if module already exists
                transaction.get_module_info(handle).is_none()
            }
        };

        if !should_reload && transaction.get_module_info(handle).is_some() {
            return None; // Module already loaded and sufficient
        }

        // Module not loaded or needs upgrade, create a fresh transaction and load it
        let mut fresh_transaction = self.inner.state.transaction();
        fresh_transaction.run(&[(handle.clone(), required_level)]);

        // Verify the module was loaded successfully
        if fresh_transaction.get_module_info(handle).is_some() {
            Some(fresh_transaction)
        } else {
            None // Module couldn't be loaded even with fresh transaction
        }
    }

    pub fn current_snapshot(&self) -> i32 {
        self.inner.state.current_snapshot()
    }

    /// Converts a pyrefly type to TSP type and registers it in the lookup table
    pub fn convert_and_register_type(&self, py_type: crate::types::types::Type) -> tsp::Type {
        // First register the type to get a stable handle
        let handle_str = self.inner.state.register_type_handle(py_type.clone());

        // Create TSP type with the stable handle
        crate::tsp::common::convert_to_tsp_type_with_handle(py_type, handle_str)
    }

    /// Looks up a pyrefly type from an integer TSP type handle
    fn lookup_type_by_int_handle(&self, id: i32) -> Option<crate::types::types::Type> {
        // For now, convert integer handle to string and use the existing lookup
        // In a more sophisticated implementation, we might have separate integer and string lookups
        let handle_str = format!("{id}");
        self.inner.state.lookup_type_from_handle(&handle_str)
    }

    /// Looks up a pyrefly type from a TSP Type
    pub fn lookup_type_from_tsp_type(
        &self,
        tsp_type: &tsp::Type,
    ) -> Option<crate::types::types::Type> {
        match &tsp_type.handle {
            tsp::TypeHandle::String(handle_str) => {
                self.inner.state.lookup_type_from_handle(handle_str)
            }
            tsp::TypeHandle::Int(id) => self.lookup_type_by_int_handle(*id),
        }
    }
}

/// A builder to create TSP declarations consistently across handlers
pub struct DeclarationBuilder {
    handle: Option<tsp::DeclarationHandle>,
    category: tsp::DeclarationCategory,
    flags: tsp::DeclarationFlags,
    node: Option<tsp::Node>,
    module_name: tsp::ModuleName,
    name: String,
    uri: lsp_types::Url,
}

impl DeclarationBuilder {
    /// Create a builder for a symbol in the given module and URI
    pub fn new(name: impl Into<String>, module_name: tsp::ModuleName, uri: lsp_types::Url) -> Self {
        Self {
            handle: None,
            category: tsp::DeclarationCategory::Variable,
            flags: tsp::DeclarationFlags::new(),
            node: None,
            module_name,
            name: name.into(),
            uri,
        }
    }

    /// Set a string handle
    pub fn handle_str(mut self, handle: impl Into<String>) -> Self {
        self.handle = Some(tsp::DeclarationHandle::String(handle.into()));
        self
    }

    /// Set the declaration category
    pub fn category(mut self, category: tsp::DeclarationCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the flags
    pub fn flags(mut self, flags: tsp::DeclarationFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set the declaration node directly
    pub fn node(mut self, node: tsp::Node) -> Self {
        self.node = Some(node);
        self
    }

    /// Build the declaration
    pub fn build(self) -> tsp::Declaration {
        tsp::Declaration {
            handle: self.handle.unwrap_or_else(|| {
                // default unique-ish handle from name and uri
                tsp::DeclarationHandle::String(format!(
                    "decl_{}_{}_{}",
                    self.name,
                    self.uri,
                    self.node
                        .as_ref()
                        .map(|n| format!("{}:{}", n.range.start.line, n.range.start.character))
                        .unwrap_or_else(|| "no_node".to_owned())
                ))
            }),
            category: self.category,
            flags: self.flags,
            node: self.node,
            module_name: self.module_name,
            name: self.name,
            uri: self.uri.to_string(),
        }
    }
}

/// Helper to compute the start position of a TSP node in the given module.
pub fn node_start_position(module_info: &ModuleInfo, node: &tsp::Node) -> TextSize {
    let pos = &node.range.start;
    module_info
        .lined_buffer()
        .from_lsp_position(lsp_types::Position {
            line: pos.line,
            character: pos.character,
        })
}
