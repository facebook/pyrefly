/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::symbol_kind::SymbolKind;
use pyrefly_types::types::Type;
use pyrefly_util::telemetry::EmptyResponseReason;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged as _;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;
use vec1::vec1;

use super::DefinitionMetadata;
use super::FindDefinitionItemWithDocstring;
use super::FindPreference;
use crate::alt::answers::Answers;
use crate::export::exports::Export;
use crate::state::lsp::ImportBehavior;
use crate::state::lsp::resolve_relative_module_name;
use crate::state::state::Transaction;

/// Search file content for a symbol name and return the `TextRange` of its
/// first occurrence as a whole word. Used by go-to-definition to locate
/// symbols in non-Python source files (e.g. thrift struct/enum names).
pub(super) fn find_symbol_range_in_text(content: &str, symbol: &str) -> Option<TextRange> {
    fn is_word_char(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }
    let bytes = content.as_bytes();
    let mut start = 0;
    while let Some(pos) = content[start..].find(symbol) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0 || !is_word_char(bytes[abs_pos - 1]);
        let after_pos = abs_pos + symbol.len();
        let after_ok = after_pos >= bytes.len() || !is_word_char(bytes[after_pos]);
        if before_ok && after_ok {
            return Some(TextRange::new(
                TextSize::new(abs_pos as u32),
                TextSize::new(after_pos as u32),
            ));
        }
        start = abs_pos + 1;
    }
    None
}

impl Transaction<'_> {
    pub(crate) fn config_has_extra_extensions(&self, handle: &Handle) -> bool {
        !self
            .config_finder()
            .python_file(handle.module_kind(), handle.path())
            .extra_file_extensions
            .is_empty()
    }

    /// Search a non-Python source file for a symbol definition and return a
    /// `FindDefinitionItemWithDocstring` pointing at the symbol's location.
    /// Falls back to the module start if the symbol is not found.
    ///
    /// This handles go-to-definition for imports from non-Python files (e.g.
    /// `.thrift`) by locating the symbol name in the source text, so the user
    /// lands on the actual definition rather than just the top of the file.
    pub(crate) fn find_symbol_in_non_python_module(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        symbol_name: &str,
        preference: FindPreference,
    ) -> Result<Option<FindDefinitionItemWithDocstring>, EmptyResponseReason> {
        let Some(module_handle) =
            self.import_handle_with_preference(handle, module_name, preference)
        else {
            return Err(EmptyResponseReason::ModuleNotFound);
        };
        let _ = self.get_exports(&module_handle);
        let module_info = self
            .get_module_info(&module_handle)
            .ok_or(EmptyResponseReason::ModuleInfoNotFound)?;

        let definition_range =
            find_symbol_range_in_text(module_info.contents(), symbol_name).unwrap_or_default();

        Ok(Some(FindDefinitionItemWithDocstring {
            metadata: DefinitionMetadata::Module,
            definition_range,
            module: module_info,
            docstring_range: None,
            display_name: Some(symbol_name.to_owned()),
        }))
    }

    /// For __files__/__recursefiles__ directory imports in non-Python modules, the
    /// virtual module doesn't exist on disk. Navigate to the parent module.
    pub(crate) fn find_definition_directory_import(
        &self,
        handle: &Handle,
        resolved_module_name: &str,
        preference: FindPreference,
    ) -> Result<Option<Vec1<FindDefinitionItemWithDocstring>>, EmptyResponseReason> {
        let module_str = resolved_module_name;
        if let Some(parent) = module_str
            .strip_suffix(".__files__")
            .or_else(|| module_str.strip_suffix(".__recursefiles__"))
            && let Some(item) = self.find_definition_for_imported_module(
                handle,
                ModuleName::from_str(parent),
                preference,
            )?
        {
            return Ok(Some(vec1![item]));
        }
        Ok(None)
    }

    pub(crate) fn resolve_intermediate_non_python_module_definition(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        name: &str,
        preference: FindPreference,
    ) -> Option<(Handle, Export)> {
        // The import target is unresolvable through any
        // chase path (export, submodule, `__getattr__`).
        // For non-Python modules (e.g. .thrift files imported
        // via extra_file_extensions), try to locate the symbol
        // by text search in the source file.
        if self.config_has_extra_extensions(handle)
            && let Some(module_handle) =
                self.import_handle_with_preference(handle, module_name, preference)
            && let Some(module_info) = self.get_module_info(&module_handle)
            && !module_info
                .path()
                .as_path()
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| PYTHON_EXTENSIONS.contains(&ext))
        {
            // The first found occurrence of the symbol, or the top of the
            // file if it couldn't be resolved.
            let definition_range =
                find_symbol_range_in_text(module_info.contents(), name).unwrap_or_default();
            return Some((
                module_handle,
                Export {
                    location: definition_range,
                    symbol_kind: Some(SymbolKind::Variable),
                    docstring_range: None,
                    deprecation: None,
                    is_final: false,
                    special_export: None,
                },
            ));
        }
        None
    }

    pub(crate) fn find_definition_for_attribute_in_non_python_module(
        &self,
        handle: &Handle,
        name: &str,
        preference: FindPreference,
        answers: &Answers,
        base_type: Type,
        base_range: TextRange,
    ) -> Option<Vec1<FindDefinitionItemWithDocstring>> {
        // Fallback for non-Python modules (e.g. .thrift files that can't be
        // parsed as Python): navigate to the module file itself. Only applies
        // when extra file extensions are configured.
        let config_has_extra_extensions = self.config_has_extra_extensions(handle);
        if !config_has_extra_extensions {
            return None;
        }
        if let Type::Module(ref module) = base_type {
            let module_name = ModuleName::from_parts(module.parts());
            if let Ok(Some(item)) =
                self.find_symbol_in_non_python_module(handle, module_name, name, preference)
                && !item.is_python_module()
            {
                return Some(vec1![item]);
            }
        }
        // Fallback for base expressions that resolve to Any/Unknown.
        // Two cases:
        // 1. Nested attribute: `module.Container.member` — the base
        //    `module.Container` is an ExprAttribute whose inner value
        //    has Type::Module.
        // 2. From-import: `from module import Name; Name.member` — the
        //    base `Name` is an ExprName imported from a non-Python module.
        if base_type.is_any()
            && let Some(mod_module) = self.get_ast(handle)
        {
            let covering_nodes = Ast::locate_node(&mod_module, base_range.start());
            for node in &covering_nodes {
                if let AnyNodeRef::ExprAttribute(attr) = node
                    && attr.range() == base_range
                    && let Some(Type::Module(ref module)) =
                        answers.get_type_trace(attr.value.range())
                {
                    let module_name = ModuleName::from_parts(module.parts());
                    if let Ok(Some(item)) =
                        self.find_symbol_in_non_python_module(handle, module_name, name, preference)
                        && !item.is_python_module()
                    {
                        return Some(vec1![item]);
                    }
                    break;
                }
                if let AnyNodeRef::ExprName(expr_name) = node
                    && expr_name.range() == base_range
                {
                    let id = Ast::expr_name_identifier((*expr_name).clone());
                    if let Ok(Some(def_item)) =
                        self.find_definition_for_name_use(handle, &id, preference)
                        && !def_item.is_python_module()
                    {
                        let definition_range =
                            find_symbol_range_in_text(def_item.module.contents(), name)
                                .unwrap_or_default();
                        return Some(vec1![FindDefinitionItemWithDocstring {
                            metadata: DefinitionMetadata::Module,
                            definition_range,
                            module: def_item.module,
                            docstring_range: None,
                            display_name: Some(name.to_owned()),
                        }]);
                    }
                    break;
                }
            }
        }
        None
    }

    /// When extra file extensions are configured and jumping through
    /// everything, if the definition resolved back to the import
    /// statement itself, the import couldn't be followed (e.g. the
    /// source module is a non-Python file like .thrift). Fall through
    /// to navigate to the source module file instead.
    pub(crate) fn remap_find_definition_non_python_import_file(
        &self,
        item: FindDefinitionItemWithDocstring,
        handle: &Handle,
        preference: FindPreference,
        module_name: ModuleName,
        dots: u32,
        name_after_import: Identifier,
    ) -> Result<Vec1<FindDefinitionItemWithDocstring>, EmptyResponseReason> {
        let config_has_extra_extensions = self.config_has_extra_extensions(handle);
        // Do we want to get as close to the definition as possible?
        // If not, we don't want to keep searching for the module
        // on a non-Python file
        let jump_through_everything = matches!(
            preference.import_behavior,
            ImportBehavior::JumpThroughEverything,
        );
        // Is the thing we found the import statement?
        let found_item_is_import = item.definition_range == name_after_import.range;

        let is_eligible_for_fallback =
            config_has_extra_extensions && jump_through_everything && found_item_is_import;
        if !is_eligible_for_fallback {
            return Ok(vec1![item]);
        }
        let resolved_module_name = resolve_relative_module_name(handle, module_name, dots);
        if let Ok(Some(module_item)) = self.find_symbol_in_non_python_module(
            handle,
            resolved_module_name,
            name_after_import.id.as_str(),
            preference,
        ) {
            if module_item.is_python_module() {
                Ok(vec1![item])
            } else {
                Ok(vec1![module_item])
            }
        } else {
            Ok(vec1![item])
        }
    }

    /// For extra-extension modules (e.g. .thrift, .cinc), the file
    /// extension is part of the module name. Clicking on a filename
    /// component like `TranslationCheckConfig` in
    /// `from pkg.TranslationCheckConfig.thrift import XYZ` truncates
    /// to `pkg.TranslationCheckConfig`, which doesn't resolve. Try
    /// extending with subsequent components until a module is found.
    pub(crate) fn fallback_find_definition_module_name_with_suffix(
        &self,
        handle: &Handle,
        preference: FindPreference,
        components: &[Name],
        target_idx: usize,
    ) -> Option<FindDefinitionItemWithDocstring> {
        if self.config_has_extra_extensions(handle) {
            // Start at target_idx + 2: components[..=target_idx] was
            // already tried above, so the first new slice is [..target_idx+2].
            for end in (target_idx + 2)..=components.len() {
                let extended = ModuleName::from_parts(&components[..end]);
                if let Ok(Some(item)) =
                    self.find_definition_for_imported_module(handle, extended, preference)
                {
                    return Some(item);
                }
            }
        }
        None
    }
}

impl FindDefinitionItemWithDocstring {
    pub(crate) fn is_python_module(&self) -> bool {
        self.module
            .path()
            .as_path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| PYTHON_EXTENSIONS.contains(&ext))
    }
}
