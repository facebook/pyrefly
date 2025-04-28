/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use lsp_types::DocumentSymbol;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::ordered_set::OrderedSet;

use crate::alt::attr::AttrInfo;
use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::export::definitions::DocString;
use crate::export::exports::Export;
use crate::export::exports::ExportLocation;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::TextRangeWithModuleInfo;
use crate::module::module_name::ModuleName;
use crate::module::short_identifier::ShortIdentifier;
use crate::ruff::ast::Ast;
use crate::state::handle::Handle;
use crate::state::state::Transaction;
use crate::types::lsp::source_range_to_range;
use crate::types::module::Module;
use crate::types::types::Type;
use crate::util::gas::Gas;
use crate::util::prelude::VecExt;
use crate::util::visit::Visit;

const INITIAL_GAS: Gas = Gas::new(20);

pub enum DefinitionMetadata {
    Attribute(Name),
    Module,
    Variable,
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

    fn import_at(&self, handle: &Handle, position: TextSize) -> Option<ModuleName> {
        let module = self.get_ast(handle)?;
        for (module, text_range) in Ast::imports(&module, handle.module(), handle.path().is_init())
        {
            if text_range.contains_inclusive(position) {
                return Some(module);
            }
        }
        None
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
                return self.get_type(handle, &Key::Usage(ShortIdentifier::new(&id)));
            } else {
                return None;
            }
        }
        if let Some(m) = self.import_at(handle, position) {
            return Some(Type::Module(Module::new(
                m.components().first().unwrap().clone(),
                OrderedSet::from_iter([(m)]),
            )));
        }
        let attribute = self.attribute_at(handle, position)?;
        self.get_type_trace(handle, attribute.range)
    }

    fn key_to_export(&self, handle: &Handle, key: &Key, gas: Gas) -> Option<(Handle, Export)> {
        let bindings = self.get_bindings(handle)?;
        let idx = bindings.key_to_idx(key);
        let res = self.binding_to_export(handle, bindings.get(idx), gas);
        if res.is_none()
            && let Key::Definition(x) = key
        {
            return Some((
                handle.dupe(),
                Export {
                    location: x.range(),
                    docstring: None,
                },
            ));
        }
        if res.is_none()
            && let Key::Anywhere(_, range) = key
        {
            return Some((
                handle.dupe(),
                Export {
                    location: *range,
                    docstring: None,
                },
            ));
        }
        res
    }

    fn binding_to_export(
        &self,
        handle: &Handle,
        binding: &Binding,
        mut gas: Gas,
    ) -> Option<(Handle, Export)> {
        if gas.stop() {
            return None;
        }
        let bindings = self.get_bindings(handle)?;
        match binding {
            Binding::Forward(k) => self.key_to_export(handle, bindings.idx_to_key(*k), gas),
            Binding::Default(_, m) => self.binding_to_export(handle, m, gas),
            Binding::Phi(ks) if !ks.is_empty() => {
                self.key_to_export(handle, bindings.idx_to_key(*ks.iter().next().unwrap()), gas)
            }
            Binding::Import(mut m, name) => {
                while !gas.stop() {
                    let handle = self.import_handle(handle, m, None).ok()?;
                    match self.get_exports(&handle).get(name) {
                        Some(ExportLocation::ThisModule(export)) => {
                            return Some((handle.clone(), export.clone()));
                        }
                        Some(ExportLocation::OtherModule(module)) => m = *module,
                        None => return None,
                    }
                }
                None
            }
            Binding::Module(name, _, _) => {
                let handle = self.import_handle(handle, *name, None).ok()?;
                let docstring = self.get_module_docstring(&handle);
                Some((
                    handle,
                    Export {
                        location: TextRange::default(),
                        docstring,
                    },
                ))
            }
            Binding::CheckLegacyTypeParam(k, _) => {
                let binding = bindings.get(*k);
                self.key_to_export(handle, bindings.idx_to_key(binding.0), gas)
            }
            _ => None,
        }
    }

    /// Find the definition, metadata and optionally the docstring for the given position.
    fn find_definition(
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
            return Some((
                DefinitionMetadata::Variable,
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
            ) = self.key_to_export(handle, &Key::Usage(ShortIdentifier::new(&id)), INITIAL_GAS)?;
            return Some((
                DefinitionMetadata::Variable,
                TextRangeWithModuleInfo::new(self.get_module_info(&handle)?, location),
                docstring,
            ));
        }
        if let Some(m) = self.import_at(handle, position) {
            let handle = self.import_handle(handle, m, None).ok()?;
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
            let items = solver.completions(base_type.arc_clone(), false);
            items.into_iter().find_map(|x| {
                if x.name == attribute.attr.id {
                    // TODO(kylei): attribute docstrings
                    Some((
                        DefinitionMetadata::Attribute(x.name),
                        TextRangeWithModuleInfo::new(x.module?, x.range?),
                        None,
                    ))
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
        } else {
            Vec::new()
        }
    }

    fn local_references_from_definition(
        &self,
        handle: &Handle,
        definition_metadata: DefinitionMetadata,
        definition: TextRangeWithModuleInfo,
    ) -> Vec<TextRange> {
        let mut references = match definition_metadata {
            DefinitionMetadata::Attribute(expected_name) => {
                self.local_attribute_references_from_definition(handle, &definition, &expected_name)
            }
            DefinitionMetadata::Module => Vec::new(),
            DefinitionMetadata::Variable => {
                self.local_variable_references_from_definition(handle, &definition)
            }
        };
        if definition.module_info.path() == handle.path() {
            references.push(definition.range);
        }
        references.sort_by_key(|range| range.start());
        references.dedup();
        references
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
                        module,
                        range,
                    } in solver.completions(base_type.arc_clone(), false)
                    {
                        if let Some(module) = module
                            && let Some(range) = range
                            && &name == expected_name
                            && module.path() == definition.module_info.path()
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
    ) -> Vec<TextRange> {
        let Some(bindings) = self.get_bindings(handle) else {
            return Vec::new();
        };
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
                references.push(bindings.idx_to_key(idx).range());
            }
        }
        references
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
        if self.identifier_at(handle, position).is_some() {
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
                .completions(base_type.arc_clone(), true)
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

    #[allow(dead_code)]
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
                    symbols.push(DocumentSymbol {
                        name: stmt_function_def.name.to_string(),
                        detail: None,
                        kind: lsp_types::SymbolKind::FUNCTION,
                        tags: None,
                        #[expect(deprecated)]
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
                    symbols.push(DocumentSymbol {
                        name: stmt_class_def.name.to_string(),
                        detail: None,
                        kind: lsp_types::SymbolKind::CLASS,
                        tags: None,
                        #[expect(deprecated)]
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
