/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use ruff_python_ast::Decorator;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::decorator_matches_name;
use super::extract_shared::line_end_position;
use super::extract_shared::line_indent_and_start;
use super::extract_shared::member_name_from_stmt;
use super::extract_shared::prepare_insertion_text;
use super::extract_shared::reindent_block;
use super::extract_shared::selection_anchor;
use super::types::LocalRefactorCodeAction;
use crate::state::ide::insert_import_edit;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::Transaction;

fn move_kind() -> CodeActionKind {
    CodeActionKind::new("refactor.move")
}

#[derive(Clone, Copy, Debug)]
enum ParentKind<'a> {
    Module,
    Function,
    Class(&'a StmtClassDef),
}

#[derive(Clone, Copy, Debug)]
struct LocalFunctionContext<'a> {
    function_def: &'a StmtFunctionDef,
    parent_body: &'a [Stmt],
    parent: ParentKind<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MethodWrapper {
    None,
    Staticmethod,
    Classmethod,
}

/// Builds move-module-member code actions for top-level symbols.
pub(crate) fn move_module_member_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
    import_format: ImportFormat,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let member_stmt = find_module_member(ast.as_ref(), selection_point)?;
    let member_name = member_name_from_stmt(member_stmt)?;
    let (from_indent, _) = line_indent_and_start(source, member_stmt.range().start())?;
    let member_text = reindent_statement(source, member_stmt.range(), &from_indent, "");

    let removal_range = statement_removal_range(source, member_stmt)?;
    let mut actions = Vec::new();
    for (target_handle, target_info, target_ast) in
        sibling_module_targets(transaction, handle, &module_info)?
    {
        if target_info.path() == module_info.path() {
            continue;
        }
        let insert_edit =
            build_module_insertion_edit(&target_info, target_ast.as_ref(), &member_text, None)?;
        let (removal_edit, import_edit) = build_removal_and_import_edits(
            transaction,
            handle,
            &module_info,
            ast.as_ref(),
            &member_name,
            &target_handle,
            import_format,
            removal_range,
        )?;
        let mut edits = vec![insert_edit, removal_edit];
        if let Some(import_edit) = import_edit {
            edits.push(import_edit);
        }
        actions.push(LocalRefactorCodeAction {
            title: format!(
                "Move `{member_name}` to `{}`",
                target_handle.module().as_str()
            ),
            edits,
            kind: move_kind(),
        });
    }
    if actions.is_empty() {
        None
    } else {
        actions.sort_by(|a, b| a.title.cmp(&b.title));
        Some(actions)
    }
}

/// Builds make-local-function/method-top-level code actions.
pub(crate) fn make_local_function_top_level_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
    import_format: ImportFormat,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let context = find_local_function_context(ast.as_ref(), selection_point, ParentKind::Module)?;
    if matches!(context.parent, ParentKind::Module) {
        return None;
    }
    if let ParentKind::Class(class_def) = context.parent {
        let is_top_level_class = ast
            .body
            .iter()
            .any(|stmt| matches!(stmt, Stmt::ClassDef(def) if def.range() == class_def.range()));
        if !is_top_level_class {
            return None;
        }
    }
    if matches!(context.parent, ParentKind::Function)
        && contains_nonlocal_or_global(context.function_def)
    {
        return None;
    }

    let wrapper_kind = if let ParentKind::Class(_) = context.parent {
        method_wrapper_kind(context.function_def)?
    } else {
        MethodWrapper::None
    };
    let (function_text, from_indent) =
        function_text_for_top_level(source, context.function_def, wrapper_kind)?;
    let removal_range = statement_removal_range_from_range(source, context.function_def.range())?;

    let mut actions = Vec::new();
    // Current module action.
    if let Some(action) = build_local_function_move_action(
        transaction,
        handle,
        &module_info,
        ast.as_ref(),
        source,
        &function_text,
        &from_indent,
        removal_range,
        &context,
        None,
        import_format,
        wrapper_kind,
    ) {
        actions.push(action);
    }

    if let Some(targets) = sibling_module_targets(transaction, handle, &module_info) {
        for (target_handle, target_info, target_ast) in targets {
            if target_info.path() == module_info.path() {
                continue;
            }
            if let Some(action) = build_local_function_move_action(
                transaction,
                handle,
                &module_info,
                ast.as_ref(),
                source,
                &function_text,
                &from_indent,
                removal_range,
                &context,
                Some((target_handle, target_info, target_ast)),
                import_format,
                wrapper_kind,
            ) {
                actions.push(action);
            }
        }
    }

    if actions.is_empty() {
        None
    } else {
        actions.sort_by(|a, b| a.title.cmp(&b.title));
        Some(actions)
    }
}

fn build_local_function_move_action(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    ast: &ModModule,
    source: &str,
    function_text: &str,
    from_indent: &str,
    removal_range: TextRange,
    context: &LocalFunctionContext<'_>,
    target: Option<(Handle, Module, std::sync::Arc<ModModule>)>,
    import_format: ImportFormat,
    wrapper_kind: MethodWrapper,
) -> Option<LocalRefactorCodeAction> {
    let (title, insert_edit, import_target) = match target {
        Some((target_handle, target_info, target_ast)) => {
            let insert_edit = build_module_insertion_edit(
                &target_info,
                target_ast.as_ref(),
                function_text,
                None,
            )?;
            (
                format!(
                    "Move `{}` to `{}`",
                    context.function_def.name.id,
                    target_handle.module().as_str()
                ),
                insert_edit,
                Some(target_handle),
            )
        }
        None => {
            let insert_position = match context.parent {
                ParentKind::Class(class_def) => {
                    let (_, line_start) = line_indent_and_start(source, class_def.range().start())?;
                    line_start
                }
                ParentKind::Function => module_insertion_point(ast, source)?,
                ParentKind::Module => return None,
            };
            let insert_text = prepare_insertion_text(source, insert_position, function_text);
            (
                format!("Make `{}` top-level", context.function_def.name.id),
                (
                    module_info.dupe(),
                    TextRange::at(insert_position, TextSize::new(0)),
                    insert_text,
                ),
                None,
            )
        }
    };

    let removal_edit = build_local_function_removal_edit(
        module_info,
        context,
        removal_range,
        from_indent,
        wrapper_kind,
    )?;

    let import_edit = if let Some(target_handle) = import_target {
        build_import_edit(
            transaction,
            handle,
            module_info,
            ast,
            &context.function_def.name.id,
            &target_handle,
            import_format,
        )
    } else {
        None
    };

    let mut edits = vec![insert_edit, removal_edit];
    if let Some(import_edit) = import_edit {
        edits.push(import_edit);
    }
    Some(LocalRefactorCodeAction {
        title,
        edits,
        kind: move_kind(),
    })
}

fn build_local_function_removal_edit(
    module_info: &Module,
    context: &LocalFunctionContext<'_>,
    removal_range: TextRange,
    indent: &str,
    wrapper_kind: MethodWrapper,
) -> Option<(Module, TextRange, String)> {
    let replacement = match context.parent {
        ParentKind::Class(_) => {
            let name = context.function_def.name.id.as_str();
            let wrapper = match wrapper_kind {
                MethodWrapper::None => name.to_owned(),
                MethodWrapper::Staticmethod => format!("staticmethod({name})"),
                MethodWrapper::Classmethod => format!("classmethod({name})"),
            };
            format!("{indent}{name} = {wrapper}\n")
        }
        ParentKind::Function => {
            if needs_pass_after_removal(context.parent_body, context.function_def.range()) {
                format!("{indent}pass\n")
            } else {
                String::new()
            }
        }
        ParentKind::Module => return None,
    };
    Some((module_info.dupe(), removal_range, replacement))
}

fn build_removal_and_import_edits(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    ast: &ModModule,
    member_name: &str,
    target_handle: &Handle,
    import_format: ImportFormat,
    removal_range: TextRange,
) -> Option<(
    (Module, TextRange, String),
    Option<(Module, TextRange, String)>,
)> {
    let import_edit = build_import_edit(
        transaction,
        handle,
        module_info,
        ast,
        member_name,
        target_handle,
        import_format,
    );
    let removal_text = if import_edit
        .as_ref()
        .is_some_and(|edit| edit.1.start() == removal_range.start())
    {
        // If the import would be inserted exactly at the removal location (typically the first
        // statement), fold the import into the removal replacement to avoid overlapping edits.
        import_edit
            .as_ref()
            .map(|edit| edit.2.clone())
            .unwrap_or_default()
    } else {
        String::new()
    };
    let removal_edit = (module_info.dupe(), removal_range, removal_text);
    let import_edit = import_edit.and_then(|edit| {
        if edit.1.start() == removal_range.start() {
            None
        } else {
            Some(edit)
        }
    });
    Some((removal_edit, import_edit))
}

fn build_import_edit(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
    ast: &ModModule,
    member_name: &str,
    target_handle: &Handle,
    import_format: ImportFormat,
) -> Option<(Module, TextRange, String)> {
    if has_existing_import(ast, target_handle.module(), member_name) {
        return None;
    }
    let import_edit = insert_import_edit(
        ast,
        transaction.config_finder(),
        handle.dupe(),
        target_handle.dupe(),
        member_name,
        import_format,
    );
    Some((module_info.dupe(), import_edit.range, import_edit.new_text))
}

fn has_existing_import(ast: &ModModule, module_name: ModuleName, name: &str) -> bool {
    ast.body.iter().any(|stmt| match stmt {
        Stmt::ImportFrom(import_from) => {
            if let Some(module) = &import_from.module
                && ModuleName::from_name(&module.id) == module_name
            {
                import_from.names.iter().any(|alias| {
                    if alias.name.id.as_str() != name {
                        return false;
                    }
                    match &alias.asname {
                        None => true,
                        Some(asname) => asname.id.as_str() == name,
                    }
                })
            } else {
                false
            }
        }
        _ => false,
    })
}

fn build_module_insertion_edit(
    module_info: &Module,
    ast: &ModModule,
    member_text: &str,
    insert_position: Option<TextSize>,
) -> Option<(Module, TextRange, String)> {
    let source = module_info.contents();
    let position = insert_position
        .unwrap_or_else(|| module_insertion_point(ast, source).unwrap_or(TextSize::new(0)));
    let insert_text = prepare_insertion_text(source, position, member_text);
    Some((
        module_info.dupe(),
        TextRange::at(position, TextSize::new(0)),
        insert_text,
    ))
}

fn module_insertion_point(ast: &ModModule, source: &str) -> Option<TextSize> {
    if let Some(last_stmt) = ast.body.last() {
        Some(line_end_position(source, last_stmt.range().end()))
    } else {
        Some(TextSize::new(0))
    }
}

fn find_module_member<'a>(ast: &'a ModModule, selection: TextSize) -> Option<&'a Stmt> {
    ast.body.iter().find(|stmt| {
        stmt.range().contains(selection)
            && matches!(
                stmt,
                Stmt::FunctionDef(_) | Stmt::ClassDef(_) | Stmt::Assign(_) | Stmt::AnnAssign(_)
            )
    })
}

fn find_local_function_context<'a>(
    ast: &'a ModModule,
    selection: TextSize,
    parent: ParentKind<'a>,
) -> Option<LocalFunctionContext<'a>> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
            && let Some(found) = find_local_function_context_in_body(
                &class_def.body,
                selection,
                ParentKind::Class(class_def),
            )
        {
            return Some(found);
        }
        if let Stmt::FunctionDef(function_def) = stmt
            && function_def.range().contains(selection)
        {
            if let Some(found) = find_local_function_context_in_body(
                &function_def.body,
                selection,
                ParentKind::Function,
            ) {
                return Some(found);
            }
            if !matches!(parent, ParentKind::Module) {
                return Some(LocalFunctionContext {
                    function_def,
                    parent_body: &ast.body,
                    parent,
                });
            }
        }
    }
    None
}

fn find_local_function_context_in_body<'a>(
    body: &'a [Stmt],
    selection: TextSize,
    parent: ParentKind<'a>,
) -> Option<LocalFunctionContext<'a>> {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
            && let Some(found) = find_local_function_context_in_body(
                &class_def.body,
                selection,
                ParentKind::Class(class_def),
            )
        {
            return Some(found);
        }
        if let Stmt::FunctionDef(function_def) = stmt
            && function_def.range().contains(selection)
        {
            if let Some(found) = find_local_function_context_in_body(
                &function_def.body,
                selection,
                ParentKind::Function,
            ) {
                return Some(found);
            }
            if !matches!(parent, ParentKind::Module) {
                return Some(LocalFunctionContext {
                    function_def,
                    parent_body: body,
                    parent,
                });
            }
        }
    }
    None
}

fn function_text_for_top_level(
    source: &str,
    function_def: &StmtFunctionDef,
    wrapper_kind: MethodWrapper,
) -> Option<(String, String)> {
    let range = match wrapper_kind {
        MethodWrapper::None => function_def.range(),
        MethodWrapper::Staticmethod | MethodWrapper::Classmethod => {
            range_without_decorators(source, function_def.range(), &function_def.decorator_list)
        }
    };
    let (from_indent, _) = line_indent_and_start(source, range.start())?;
    let text = reindent_statement(source, range, &from_indent, "");
    Some((text, from_indent))
}

fn range_without_decorators(source: &str, range: TextRange, decorators: &[Decorator]) -> TextRange {
    let decorators_range = decorators
        .first()
        .map(|first| first.range().cover(decorators.last().unwrap().range()));
    decorators_range.map_or(range, |decorators_range| {
        let line_start = line_end_position(source, decorators_range.end());
        let start =
            first_non_whitespace_offset(source, line_start, range.end()).unwrap_or(line_start);
        TextRange::new(start, range.end())
    })
}

fn first_non_whitespace_offset(source: &str, start: TextSize, end: TextSize) -> Option<TextSize> {
    let start_idx = start.to_usize().min(source.len());
    let end_idx = end.to_usize().min(source.len());
    if start_idx >= end_idx {
        return None;
    }
    source[start_idx..end_idx]
        .char_indices()
        .find(|(_, ch)| !matches!(ch, ' ' | '\t' | '\r' | '\n'))
        .and_then(|(idx, _)| TextSize::try_from(start_idx + idx).ok())
}

fn method_wrapper_kind(function_def: &StmtFunctionDef) -> Option<MethodWrapper> {
    let mut kind = MethodWrapper::None;
    for decorator in &function_def.decorator_list {
        if decorator_matches_name(&decorator.expression, "staticmethod") {
            if kind != MethodWrapper::None {
                return None;
            }
            kind = MethodWrapper::Staticmethod;
        } else if decorator_matches_name(&decorator.expression, "classmethod") {
            if kind != MethodWrapper::None {
                return None;
            }
            kind = MethodWrapper::Classmethod;
        } else {
            // Only support static/class methods for now to avoid changing decorator semantics.
            return None;
        }
    }
    Some(kind)
}

fn contains_nonlocal_or_global(function_def: &StmtFunctionDef) -> bool {
    struct ScopeModifierFinder {
        found: bool,
    }

    impl<'a> ruff_python_ast::visitor::Visitor<'a> for ScopeModifierFinder {
        fn visit_stmt(&mut self, stmt: &'a Stmt) {
            if self.found {
                return;
            }
            match stmt {
                Stmt::Nonlocal(_) | Stmt::Global(_) => {
                    self.found = true;
                }
                Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
                _ => ruff_python_ast::visitor::walk_stmt(self, stmt),
            }
        }
    }

    let mut finder = ScopeModifierFinder { found: false };
    finder.visit_body(&function_def.body);
    finder.found
}

fn sibling_module_targets(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &Module,
) -> Option<Vec<(Handle, Module, std::sync::Arc<ModModule>)>> {
    let current_module = handle.module();
    let current_components = current_module.components();
    let parent_len = current_components.len().saturating_sub(1);
    let parent_prefix = &current_components[..parent_len];
    let mut targets = Vec::new();

    for module_name in transaction.modules() {
        if module_name == current_module || module_name.as_str() == "builtins" {
            continue;
        }
        let components = module_name.components();
        let is_sibling = if parent_len == 0 {
            components.len() == 1
        } else {
            components.len() == parent_len + 1 && &components[..parent_len] == parent_prefix
        };
        let is_parent_module =
            parent_len > 0 && components.len() == parent_len && components == parent_prefix;
        if !is_sibling && !is_parent_module {
            continue;
        }
        let Some(target_handle) = transaction
            .import_handle(handle, module_name, None)
            .finding()
        else {
            continue;
        };
        let Some(target_info) = transaction.get_module_info(&target_handle) else {
            continue;
        };
        if !matches!(
            target_info.path().details(),
            ModulePathDetails::FileSystem(_) | ModulePathDetails::Memory(_)
        ) {
            continue;
        }
        if target_info.path() == module_info.path() {
            continue;
        }
        let Some(target_ast) = transaction.get_ast(&target_handle) else {
            continue;
        };
        targets.push((target_handle, target_info, target_ast));
    }
    if targets.is_empty() {
        None
    } else {
        targets.sort_by(|a, b| a.0.module().as_str().cmp(b.0.module().as_str()));
        Some(targets)
    }
}

fn statement_removal_range(source: &str, stmt: &Stmt) -> Option<TextRange> {
    statement_removal_range_from_range(source, stmt.range())
}

fn statement_removal_range_from_range(source: &str, range: TextRange) -> Option<TextRange> {
    let (_, line_start) = line_indent_and_start(source, range.start())?;
    let line_end = line_end_position(source, range.end());
    Some(TextRange::new(line_start, line_end))
}

fn needs_pass_after_removal(body: &[Stmt], removed_range: TextRange) -> bool {
    let mut non_docstring = body.iter().filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next();
    non_docstring.next().is_none() && only_stmt.is_some_and(|stmt| stmt.range() == removed_range)
}

fn reindent_statement(
    source: &str,
    range: TextRange,
    from_indent: &str,
    to_indent: &str,
) -> String {
    let start = range.start().to_usize().min(source.len());
    let end = range.end().to_usize().min(source.len());
    let raw = if start < end { &source[start..end] } else { "" };
    let mut text = reindent_block(raw, from_indent, to_indent);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text
}
