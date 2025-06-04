/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use itertools::Itertools;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::DocumentSymbol;
use pyrefly_util::gas::Gas;
use pyrefly_util::prelude::VecExt;
use pyrefly_util::task_heap::Cancelled;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::ordered_set::OrderedSet;

use crate::alt::attr::AttrDefinition;
use crate::alt::attr::AttrInfo;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::export::definitions::DocString;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::module_path::ModulePath;
use crate::module::module_path::ModulePathDetails;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::state::handle::Handle;
use crate::state::ide::IntermediateDefinition;
use crate::state::ide::binding_to_intermediate_definition;
use crate::state::ide::key_to_intermediate_definition;
use crate::state::require::Require;
use crate::state::state::CancellableTransaction;
use crate::state::state::Transaction;
use crate::sys_info::SysInfo;
use crate::types::lsp::source_range_to_range;
use crate::types::module::Module;
use crate::types::types::Type;

const INITIAL_GAS: Gas = Gas::new(20);

#[derive(Clone)]
pub enum DefinitionMetadata {
    Attribute(Name),
    Module,
    Variable,
    VariableOrAttribute(Name),
}

enum ImportIdentifier {
    // The name of a module. ex: `x` in `import x` or `from x import name`
    Module(ModuleName),
    // A name from a module's exports. ex: `name` in `from x import name`
    // Note: these are also definitions
    Name(ModuleName),
}

impl ImportIdentifier {
    fn module_name(&self) -> ModuleName {
        match self {
            ImportIdentifier::Module(module_name) => *module_name,
            ImportIdentifier::Name(module_name) => *module_name,
        }
    }
}

impl<'a> Transaction<'a> {
    fn get_type(&self, handle: &Handle, key: &Key) -> Option<Type> {
        let idx = self.get_bindings(handle)?.key_to_idx(key);
        let ans = self.get_answers(handle)?;
        Some(ans.for_display(ans.get_idx(idx)?.arc_clone_ty()))
    }

    fn get_type_trace(&self, handle: &Handle, range: TextRange) -> Option<Type> {
        let ans = self.get_answers(handle)?;
        Some(ans.for_display(ans.get_type_trace(range)?.arc_clone()))
    }

    fn identifier_at(&self, handle: &Handle, position: TextSize) -> Option<Identifier> {
        let mod_module = self.get_ast(handle)?;
        fn f(x: &Expr, find: TextSize, res: &mut Option<Identifier>) {
            if let Expr::Name(x) = x
                && x.range.contains_inclusive(find)
            {
                *res = Some(Ast::expr_name_identifier(x.clone()));
            } else {
                x.recurse(&mut |x| f(x, find, res));
            }
        }
        let mut res = None;
        mod_module.visit(&mut |x| f(x, position, &mut res));
        res
    }

    fn import_at(&self, handle: &Handle, position: TextSize) -> Option<ImportIdentifier> {
        fn visit_stmt(x: &Stmt, find: TextSize, res: &mut Option<ImportIdentifier>) {
            match x {
                Stmt::Import(stmt_import) => {
                    let mut parts = Vec::new();
                    for name in stmt_import.names.iter() {
                        parts.push(name.name.clone());
                        if name.range.contains_inclusive(find) {
                            *res = Some(ImportIdentifier::Module(ModuleName::from_parts(
                                parts.clone(),
                            )));
                        }
                    }
                }
                Stmt::ImportFrom(stmt_import_from) => {
                    if let Some(id) = &stmt_import_from.module {
                        if id.range.contains_inclusive(find) {
                            *res = Some(ImportIdentifier::Module(ModuleName::from_name(&id.id)));
                        } else {
                            for name in stmt_import_from.names.iter() {
                                if name.range.contains_inclusive(find) {
                                    *res =
                                        Some(ImportIdentifier::Name(ModuleName::from_name(&id.id)));
                                }
                            }
                        }
                    }
                }
                _ => x.recurse(&mut |x| visit_stmt(x, find, res)),
            }
        }

        let mut res = None;
        self.get_ast(handle)?
            .body
            .visit(&mut |x| visit_stmt(x, position, &mut res));
        res
    }

    fn definition_at(&self, handle: &Handle, position: TextSize) -> Option<Key> {
        self.get_bindings(handle)?
            .definition_at_position(position)
            .cloned()
    }

    fn attribute_at(&self, handle: &Handle, position: TextSize) -> Option<ExprAttribute> {
        let mod_module = self.get_ast(handle)?;
        fn f(x: &Expr, find: TextSize, res: &mut Option<ExprAttribute>) {
            if let Expr::Attribute(x) = x
                && x.attr.range.contains_inclusive(find)
            {
                *res = Some(x.clone());
            } else {
                x.recurse(&mut |x| f(x, find, res));
            }
        }
        let mut res = None;
        mod_module.visit(&mut |x| f(x, position, &mut res));
        res
    }

    pub fn get_type_at(&self, handle: &Handle, position: TextSize) -> Option<Type> {
        if let Some(key) = self.definition_at(handle, position) {
            return self.get_type(handle, &key);
        }
        if let Some(id) = self.identifier_at(handle, position) {
            if self.get_bindings(handle)?.is_valid_usage(&id) {
                return self.get_type(handle, &Key::BoundName(ShortIdentifier::new(&id)));
            } else {
                return None;
            }
        }
        if let Some(m) = self.import_at(handle, position) {
            let module_name = m.module_name();
            return Some(Type::Module(Module::new(
                module_name.components().first().unwrap().clone(),
                OrderedSet::from_iter([module_name]),
            )));
        }
        let attribute = self.attribute_at(handle, position)?;
        self.get_type_trace(handle, attribute.range)
    }

    fn resolve_named_import(
        &self,
        handle: &Handle,
        module_name: ModuleName,
        name: Name,
        gas: &mut Gas,
    ) -> Option<(Handle, Export)> {
        let mut m = module_name;
        while !gas.stop() {
            let handle = self.import_handle(handle, m, None).ok()?;
            match self.get_exports(&handle).get(&name) {
                Some(ExportLocation::ThisModule(export)) => {
                    return Some((handle.clone(), export.clone()));
                }
                Some(ExportLocation::OtherModule(module)) => m = *module,
                None => return None,
            }
        }
        None
    }

    fn resolve_intermediate_definition(
        &self,
        handle: &Handle,
        intermediate_definition: IntermediateDefinition,
        mut gas: Gas,
    ) -> Option<(Handle, Export)> {
        match intermediate_definition {
            IntermediateDefinition::Local(export) => Some((handle.dupe(), export)),
            IntermediateDefinition::NamedImport(module_name, name) => {
                self.resolve_named_import(handle, module_name, name, &mut gas)
            }
            IntermediateDefinition::Module(name) => {
                let handle = self.import_handle(handle, name, None).ok()?;
                let docstring = self.get_module_docstring(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        docstring,
                    },
                ))
            }
        }
    }

    fn resolve_attribute_definition(
        &self,
        handle: &Handle,
        attr_name: &Name,
        definition: AttrDefinition,
    ) -> Option<(TextRangeWithModuleInfo, Option<DocString>)> {
        match definition {
            AttrDefinition::FullyResolved(text_range_with_module_info) => {
                // TODO(kylei): attribute docstrings
                Some((text_range_with_module_info, None))
            }
            AttrDefinition::PartiallyResolvedImportedModuleAttribute { module_name } => {
                let mut gas = INITIAL_GAS;
                let (handle, export) =
                    self.resolve_named_import(handle, module_name, attr_name.clone(), &mut gas)?;
                let module_info = self.get_module_info(&handle)?;
                Some((
                    TextRangeWithModuleInfo::new(module_info, export.location),
                    export.docstring,
                ))
            }
        }
    }

    fn key_to_export(&self, handle: &Handle, key: &Key, mut gas: Gas) -> Option<(Handle, Export)> {
        let bindings = self.get_bindings(handle)?;
        let intermediate_definition = key_to_intermediate_definition(&bindings, key, &mut gas)?;
        self.resolve_intermediate_definition(handle, intermediate_definition, gas)
    }

    fn binding_to_export(
        &self,
        handle: &Handle,
        binding: &Binding,
        mut gas: Gas,
    ) -> Option<(Handle, Export)> {
        let bindings = self.get_bindings(handle)?;
        let intermediate_definition =
            binding_to_intermediate_definition(&bindings, binding, &mut gas)?;
        self.resolve_intermediate_definition(handle, intermediate_definition, gas)
    }

    /// Find the definition, metadata and optionally the docstring for the given position.
    pub fn find_definition(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<(
        DefinitionMetadata,
        TextRangeWithModuleInfo,
        Option<DocString>,
    )> {
        if let Some(key) = self.definition_at(handle, position) {
            let (
                handle,
                Export {
                    location,
                    docstring,
                },
            ) = self.key_to_export(handle, &key, INITIAL_GAS)?;
            let name = Name::new(self.get_module_info(&handle)?.code_at(location));
            return Some((
                DefinitionMetadata::VariableOrAttribute(name),
                TextRangeWithModuleInfo::new(self.get_module_info(&handle)?, location),
                docstring,
            ));
        }
        if let Some(id) = self.identifier_at(handle, position) {
            if !self.get_bindings(handle)?.is_valid_usage(&id) {
                return None;
            }
            let (
                handle,
                Export {
                    location,
                    docstring,
                },
            ) = self.key_to_export(
                handle,
                &Key::BoundName(ShortIdentifier::new(&id)),
                INITIAL_GAS,
            )?;
            return Some((
                DefinitionMetadata::Variable,
                TextRangeWithModuleInfo::new(self.get_module_info(&handle)?, location),
                docstring,
            ));
        }
        if let Some(m) = self.import_at(handle, position) {
            let module_name = m.module_name();
            let handle = self.import_handle(handle, module_name, None).ok()?;
            return Some((
                DefinitionMetadata::Module,
                TextRangeWithModuleInfo::new(self.get_module_info(&handle)?, TextRange::default()),
                self.get_module_docstring(&handle),
            ));
        }
        let attribute = self.attribute_at(handle, position)?;
        let base_type = self
            .get_answers(handle)?
            .get_type_trace(attribute.value.range())?;
        self.ad_hoc_solve(handle, |solver| {
            let items = solver.completions(base_type.arc_clone(), Some(&attribute.attr.id), false);
            items.into_iter().find_map(|x| {
                if x.name == attribute.attr.id {
                    let (definition, docstring) =
                        self.resolve_attribute_definition(handle, &x.name, x.definition?)?;
                    Some((DefinitionMetadata::Attribute(x.name), definition, docstring))
                } else {
                    None
                }
            })
        })
        .flatten()
    }

    pub fn goto_definition(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<TextRangeWithModuleInfo> {
        self.find_definition(handle, position).map(|x| x.1)
    }

    pub fn docstring(&self, handle: &Handle, position: TextSize) -> Option<DocString> {
        self.find_definition(handle, position).map(|x| x.2)?
    }

    pub fn find_local_references(&self, handle: &Handle, position: TextSize) -> Vec<TextRange> {
        if let Some((definition_kind, definition, _docstring)) =
            self.find_definition(handle, position)
        {
            self.local_references_from_definition(handle, definition_kind, definition)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    fn local_references_from_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition: TextRangeWithModuleInfo,
    ) -> Option<Vec<TextRange>> {
        if handle.path() != definition.module_info.path() {
            let index = self.get_solutions(handle)?.get_index()?;
            let index = index.lock();
            let mut references = Vec::new();
            for ((imported_module_name, imported_name), ranges) in
                &index.externally_defined_variable_references
            {
                let mut gas = INITIAL_GAS;
                if let Some((imported_handle, export)) = self.resolve_named_import(
                    handle,
                    *imported_module_name,
                    imported_name.clone(),
                    &mut gas,
                ) && imported_handle.path().as_path() == definition.module_info.path().as_path()
                    && export.location == definition.range
                {
                    references.extend(ranges.iter().copied());
                }
            }
            for (attribute_module_path, def_and_ref_ranges) in
                &index.externally_defined_attribute_references
            {
                if attribute_module_path == definition.module_info.path() {
                    for (def_range, ref_range) in def_and_ref_ranges {
                        if def_range == &definition.range {
                            references.push(*ref_range);
                        }
                    }
                }
            }
            references.sort_by_key(|range| range.start());
            references.dedup();
            return Some(references);
        }
        let mut references = match definition_metadata {
            DefinitionMetadata::Attribute(expected_name) => {
                self.local_attribute_references_from_definition(handle, &definition, &expected_name)
            }
            DefinitionMetadata::Module => Vec::new(),
            DefinitionMetadata::Variable => self
                .local_variable_references_from_definition(handle, &definition)
                .unwrap_or_default(),
            DefinitionMetadata::VariableOrAttribute(expected_name) => [
                self.local_attribute_references_from_definition(
                    handle,
                    &definition,
                    &expected_name,
                ),
                self.local_variable_references_from_definition(handle, &definition)
                    .unwrap_or_default(),
            ]
            .concat(),
        };
        if definition.module_info.path() == handle.path() {
            references.push(definition.range);
        }
        references.sort_by_key(|range| range.start());
        references.dedup();
        Some(references)
    }

    fn local_attribute_references_from_definition(
        &self,
        handle: &Handle,
        definition: &TextRangeWithModuleInfo,
        expected_name: &Name,
    ) -> Vec<TextRange> {
        // We first find all the attributes of the form `<expr>.<expected_name>`.
        // These are candidates for the references of `definition`.
        let relevant_attributes = if let Some(mod_module) = self.get_ast(handle) {
            fn f(x: &Expr, expected_name: &Name, res: &mut Vec<ExprAttribute>) {
                if let Expr::Attribute(x) = x
                    && x.attr.id == *expected_name
                {
                    res.push(x.clone());
                }
                x.recurse(&mut |x| f(x, expected_name, res));
            }
            let mut res = Vec::new();
            mod_module.visit(&mut |x| f(x, expected_name, &mut res));
            res
        } else {
            Vec::new()
        };
        // For each attribute we found above, we will test whether it actually will jump to the
        // given `definition`.
        self.ad_hoc_solve(handle, |solver| {
            let mut references = Vec::new();
            for attribute in relevant_attributes {
                if let Some(answers) = self.get_answers(handle)
                    && let Some(base_type) = answers.get_type_trace(attribute.value.range())
                {
                    for AttrInfo {
                        name,
                        ty: _,
                        definition: attribute_definition,
                    } in solver.completions(base_type.arc_clone(), Some(expected_name), false)
                    {
                        if let Some((
                            TextRangeWithModuleInfo {
                                module_info: module,
                                range,
                            },
                            _,
                        )) = attribute_definition.and_then(|definition| {
                            self.resolve_attribute_definition(handle, &name, definition)
                        }) && module.path() == definition.module_info.path()
                            && range == definition.range
                        {
                            references.push(attribute.attr.range());
                        }
                    }
                }
            }
            references
        })
        .unwrap_or_default()
    }

    fn local_variable_references_from_definition(
        &self,
        handle: &Handle,
        definition: &TextRangeWithModuleInfo,
    ) -> Option<Vec<TextRange>> {
        let bindings = self.get_bindings(handle)?;
        let reference_module_info = self.get_module_info(handle)?;
        let definition_code = definition.module_info.code_at(definition.range);
        let mut references = Vec::new();
        for idx in bindings.keys::<Key>() {
            let binding = bindings.get(idx);
            if let Some((
                definition_handle,
                Export {
                    location,
                    docstring: _,
                },
            )) = self.binding_to_export(handle, binding, INITIAL_GAS)
                && definition_handle.path() == definition.module_info.path()
                && definition.range == location
            {
                let reference_range = bindings.idx_to_key(idx).range();
                let reference_code = reference_module_info.code_at(reference_range);
                // Sanity check: the reference should have the same text as the definition.
                // This check helps to filter out from synthetic bindings.
                if reference_code == definition_code {
                    references.push(reference_range);
                }
            }
        }
        Some(references)
    }

    pub fn completion(&self, handle: &Handle, position: TextSize) -> Vec<CompletionItem> {
        let mut results = self
            .completion_unsorted_opt(handle, position)
            .unwrap_or_default();
        for item in &mut results {
            let sort_text = if item.label.starts_with("__") {
                "2"
            } else if item.label.as_str().starts_with("_") {
                "1"
            } else {
                "0"
            }
            .to_owned();
            item.sort_text = Some(sort_text);
        }
        results.sort_by(|item1, item2| item1.sort_text.cmp(&item2.sort_text));
        results
    }

    fn completion_unsorted_opt(
        &self,
        handle: &Handle,
        position: TextSize,
    ) -> Option<Vec<CompletionItem>> {
        if let Some(import) = self.import_at(handle, position) {
            return match import {
                ImportIdentifier::Name(module_name) => {
                    let handle = self.import_handle(handle, module_name, None).ok()?;
                    let exports = self.get_exports(&handle);
                    let completions = exports
                        .keys()
                        .map(|name| CompletionItem {
                            label: name.to_string(),
                            // todo(kylei): completion kind for exports
                            kind: Some(CompletionItemKind::VARIABLE),
                            ..Default::default()
                        })
                        .collect();
                    Some(completions)
                }
                ImportIdentifier::Module(_module_name) => {
                    // TODO(kylei): completion for module names
                    None
                }
            };
        } else if self.identifier_at(handle, position).is_some() {
            let bindings = self.get_bindings(handle)?;
            let module_info = self.get_module_info(handle)?;
            let names = bindings
                .available_definitions(position)
                .into_iter()
                .filter_map(|idx| {
                    let key = bindings.idx_to_key(idx);
                    if let Key::Definition(id) = key {
                        let detail = self.get_type(handle, key).map(|t| t.to_string());
                        Some(CompletionItem {
                            label: module_info.code_at(id.range()).to_owned(),
                            detail,
                            kind: Some(CompletionItemKind::VARIABLE),
                            ..Default::default()
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            return Some(names);
        }
        let attribute = self.attribute_at(handle, position)?;
        let base_type = self
            .get_answers(handle)?
            .get_type_trace(attribute.value.range())?;
        self.ad_hoc_solve(handle, |solver| {
            solver
                .completions(base_type.arc_clone(), None, true)
                .into_map(|x| CompletionItem {
                    label: x.name.as_str().to_owned(),
                    detail: x.ty.map(|t| t.to_string()),
                    kind: Some(CompletionItemKind::FIELD),
                    ..Default::default()
                })
        })
    }

    pub fn inlay_hints(&self, handle: &Handle) -> Option<Vec<(TextSize, String)>> {
        let is_interesting_type = |x: &Type| !x.is_error();
        let is_interesting_expr = |x: &Expr| !Ast::is_literal(x);

        let bindings = self.get_bindings(handle)?;
        let mut res = Vec::new();
        for idx in bindings.keys::<Key>() {
            match bindings.idx_to_key(idx) {
                key @ Key::ReturnType(id) => {
                    match bindings.get(bindings.key_to_idx(&Key::Definition(id.clone()))) {
                        Binding::Function(x, _pred, _class_meta) => {
                            if matches!(&bindings.get(idx), Binding::ReturnType(ret) if ret.annot.is_none())
                                && let Some(ty) = self.get_type(handle, key)
                                && is_interesting_type(&ty)
                            {
                                let fun = bindings.get(*x);
                                res.push((fun.def.parameters.range.end(), format!(" -> {ty}")));
                            }
                        }
                        _ => {}
                    }
                }
                key @ Key::Definition(_) if let Some(ty) = self.get_type(handle, key) => {
                    let e = match bindings.get(idx) {
                        Binding::NameAssign(_, None, e) => Some(&**e),
                        Binding::Expr(None, e) => Some(e),
                        _ => None,
                    };
                    if let Some(e) = e
                        && is_interesting_expr(e)
                        && is_interesting_type(&ty)
                    {
                        let ty = format!(": {}", ty);
                        res.push((key.range().end(), ty));
                    }
                }
                _ => {}
            }
        }
        Some(res)
    }

    #[allow(deprecated)] // The `deprecated` field
    pub fn symbols(&self, handle: &Handle) -> Option<Vec<DocumentSymbol>> {
        let ast = self.get_ast(handle)?;
        let module_info = self.get_module_info(handle)?;
        fn recurse_stmt_adding_symbols<'a>(
            stmt: &'a Stmt,
            symbols: &'a mut Vec<DocumentSymbol>,
            module_info: &ModuleInfo,
        ) {
            let mut recursed_symbols = Vec::new();
            stmt.recurse(&mut |stmt| {
                recurse_stmt_adding_symbols(stmt, &mut recursed_symbols, module_info)
            });

            match stmt {
                Stmt::FunctionDef(stmt_function_def) => {
                    let mut children = Vec::new();
                    children.append(&mut recursed_symbols);
                    // todo(kylei): better approach to filtering out "" for all symbols
                    let name = match stmt_function_def.name.as_str() {
                        "" => "unknown".to_owned(),
                        name => name.to_owned(),
                    };
                    symbols.push(DocumentSymbol {
                        name,
                        detail: None,
                        kind: lsp_types::SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range: source_range_to_range(
                            &module_info.source_range(stmt_function_def.range),
                        ),
                        selection_range: source_range_to_range(
                            &module_info.source_range(stmt_function_def.name.range),
                        ),
                        children: Some(children),
                    });
                }
                Stmt::ClassDef(stmt_class_def) => {
                    let mut children = Vec::new();
                    children.append(&mut recursed_symbols);
                    let name = match stmt_class_def.name.as_str() {
                        "" => "unknown".to_owned(),
                        name => name.to_owned(),
                    };
                    symbols.push(DocumentSymbol {
                        name,
                        detail: None,
                        kind: lsp_types::SymbolKind::CLASS,
                        tags: None,
                        deprecated: None,
                        range: source_range_to_range(
                            &module_info.source_range(stmt_class_def.range),
                        ),
                        selection_range: source_range_to_range(
                            &module_info.source_range(stmt_class_def.name.range),
                        ),
                        children: Some(children),
                    });
                }
                _ => {}
            };
            symbols.append(&mut recursed_symbols);
        }
        let mut result = Vec::new();
        ast.body
            .visit(&mut |stmt| recurse_stmt_adding_symbols(stmt, &mut result, &module_info));
        Some(result)
    }
}

impl<'a> CancellableTransaction<'a> {
    /// Returns Err if the request is canceled in the middle of a run.
    pub fn find_global_references_from_definition(
        &mut self,
        sys_info: &SysInfo,
        definition_kind: DefinitionMetadata,
        definition: TextRangeWithModuleInfo,
    ) -> Result<Vec<(ModuleInfo, Vec<TextRange>)>, Cancelled> {
        // General strategy:
        // 1: Compute the set of transitive rdeps.
        // 2. Find references in each one of them using the index computed during earlier checking
        let mut transitive_rdeps = match definition.module_info.path().details() {
            ModulePathDetails::Memory(path_buf) => {
                let handle_of_filesystem_counterpart = Handle::new(
                    definition.module_info.name(),
                    ModulePath::filesystem(path_buf.clone()),
                    sys_info.dupe(),
                );
                // In-memory files can never be found through import resolution (no rdeps),
                // so we must compute the transitive rdeps of its filesystem counterpart instead.
                let mut rdeps = self
                    .as_ref()
                    .get_transitive_rdeps(handle_of_filesystem_counterpart.dupe());
                // We still add itself to the rdeps set, so that we will still find local references
                // within the file.
                rdeps.insert(Handle::new(
                    definition.module_info.name(),
                    definition.module_info.path().dupe(),
                    sys_info.dupe(),
                ));
                rdeps
            }
            _ => {
                let definition_handle = Handle::new(
                    definition.module_info.name(),
                    definition.module_info.path().dupe(),
                    sys_info.dupe(),
                );
                let rdeps = self.as_ref().get_transitive_rdeps(definition_handle.dupe());
                // We still need to know everything about the definition file, because the index
                // only contains non-local references.
                self.run(&[(definition_handle, Require::Everything)])?;
                rdeps
            }
        };
        // Remove the filesystem counterpart from candidate list,
        // otherwise we will have results from both filesystem and in-memory version of the file.
        for fs_counterpart_of_in_memory_handles in transitive_rdeps
            .iter()
            .filter_map(|handle| match handle.path().details() {
                ModulePathDetails::Memory(path_buf) => Some(Handle::new(
                    handle.module(),
                    ModulePath::filesystem(path_buf.clone()),
                    handle.sys_info().dupe(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
        {
            transitive_rdeps.remove(&fs_counterpart_of_in_memory_handles);
        }
        let candidate_handles_for_references = transitive_rdeps
            .into_iter()
            .sorted_by_key(|h| h.path().dupe())
            .collect::<Vec<_>>();
        let mut global_references = Vec::new();
        for handle in candidate_handles_for_references {
            let definition = match definition.module_info.path().details() {
                // Special-case for definition inside in-memory file
                // Calling `local_references_from_definition` naively
                // will find no references outside of the in-memory file because
                // file systems don't contain in-memory files.
                ModulePathDetails::Memory(path_buf)
                    // Why do exclude the case of finding references within the same in-memory file?
                    // If we are finding references within the same in-memory file, 
                    // then there is no problem for us to use the in-memory definition location.
                    if handle.path() != definition.module_info.path() =>
                {
                    // Below, we try to patch the definition location to be at the same offset, but
                    // making the path to be filesystem path instead. In this way, in the happy case
                    // where the in-memory content is exactly the same as the filesystem content,
                    // we can successfully find all the references. However, if the content diverge, 
                    // then we will miss definitions from other files.
                    // 
                    // In general, other than checking the reverse dependency against the in-memory
                    // content, there is not much we can do: the in-memory content can diverge from
                    // the filesystem content in arbitrary ways.
                    let TextRangeWithModuleInfo { module_info, range } = &definition;
                    let module_info = if let Some(info) = self.as_ref().get_module_info(&Handle::new(
                        module_info.name(),
                        ModulePath::filesystem(path_buf.clone()),
                        handle.sys_info().dupe(),
                    )) {
                        info
                    } else {
                        module_info.dupe()
                    };
                    TextRangeWithModuleInfo {
                        module_info,
                        range: *range,
                    }
                }
                _ => definition.clone(),
            };
            let references = self
                .as_ref()
                .local_references_from_definition(&handle, definition_kind.clone(), definition)
                .unwrap_or_default();
            if !references.is_empty()
                && let Some(module_info) = self.as_ref().get_module_info(&handle)
            {
                global_references.push((module_info, references));
            }
        }
        Ok(global_references)
    }
}
