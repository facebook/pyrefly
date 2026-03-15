/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use lsp_types::CodeActionKind;
use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AnyNodeRef;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ModModule;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::MethodInfo;
use super::extract_shared::code_at_range;
use super::extract_shared::first_parameter_name;
use super::extract_shared::function_has_decorator;
use super::types::LocalRefactorCodeAction;
use crate::state::lsp::FindPreference;
use crate::state::lsp::Transaction;

#[derive(Clone, Debug)]
struct MethodContext {
    info: MethodInfo,
    is_staticmethod: bool,
    is_classmethod: bool,
}

#[derive(Clone, Debug)]
struct FunctionContext<'a> {
    function_def: &'a StmtFunctionDef,
    method: Option<MethodContext>,
}

#[derive(Clone, Debug)]
struct ParamSlot {
    name: String,
    text: String,
}

#[derive(Clone, Copy, Debug)]
enum SignatureChange {
    Remove { index: usize },
    Move { index: usize, delta: isize },
}

pub(crate) fn change_signature_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let position = selection.start();
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let function_ctx = find_function_context(ast.as_ref(), position)?;
    if !supports_change_signature(&function_ctx) {
        return None;
    }

    let receiver_count = function_ctx
        .method
        .as_ref()
        .is_some_and(|method| !method.is_staticmethod) as usize;
    let selected_index = function_ctx
        .function_def
        .parameters
        .args
        .iter()
        .position(|param| param.name().range().contains_inclusive(position))?;
    if selected_index < receiver_count {
        return None;
    }

    let editable_params = function_ctx
        .function_def
        .parameters
        .args
        .iter()
        .skip(receiver_count)
        .map(|param| {
            Some(ParamSlot {
                name: param.name().id.to_string(),
                text: code_at_range(module_info.contents(), param.range())?.to_owned(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let receiver_text = if receiver_count == 1 {
        Some(
            code_at_range(
                module_info.contents(),
                function_ctx.function_def.parameters.args.first()?.range(),
            )?
            .to_owned(),
        )
    } else {
        None
    };
    let editable_index = selected_index - receiver_count;
    let selected_name = editable_params.get(editable_index)?.name.clone();
    let mut actions = Vec::new();

    let selected_param = function_ctx
        .function_def
        .parameters
        .args
        .get(selected_index)?;
    if !parameter_is_referenced_in_body(
        transaction,
        handle,
        module_info.path(),
        function_ctx.function_def,
        selected_param,
    ) {
        let remove_change = SignatureChange::Remove {
            index: editable_index,
        };
        if let Some(edits) = build_signature_edits(
            transaction,
            handle,
            &function_ctx,
            &editable_params,
            receiver_text.as_deref(),
            remove_change,
        ) {
            actions.push(LocalRefactorCodeAction {
                title: format!("Remove parameter `{selected_name}`"),
                edits,
                kind: CodeActionKind::REFACTOR_REWRITE,
            });
        }
    }

    if editable_index > 0 {
        let move_left = SignatureChange::Move {
            index: editable_index,
            delta: -1,
        };
        if let Some(edits) = build_signature_edits(
            transaction,
            handle,
            &function_ctx,
            &editable_params,
            receiver_text.as_deref(),
            move_left,
        ) {
            actions.push(LocalRefactorCodeAction {
                title: format!("Move parameter `{selected_name}` left"),
                edits,
                kind: CodeActionKind::REFACTOR_REWRITE,
            });
        }
    }

    if editable_index + 1 < editable_params.len() {
        let move_right = SignatureChange::Move {
            index: editable_index,
            delta: 1,
        };
        if let Some(edits) = build_signature_edits(
            transaction,
            handle,
            &function_ctx,
            &editable_params,
            receiver_text.as_deref(),
            move_right,
        ) {
            actions.push(LocalRefactorCodeAction {
                title: format!("Move parameter `{selected_name}` right"),
                edits,
                kind: CodeActionKind::REFACTOR_REWRITE,
            });
        }
    }

    (!actions.is_empty()).then_some(actions)
}

fn find_function_context(ast: &ModModule, position: TextSize) -> Option<FunctionContext<'_>> {
    let covering_nodes = Ast::locate_node(ast, position);
    for (idx, node) in covering_nodes.iter().enumerate() {
        if let AnyNodeRef::StmtFunctionDef(function_def) = node {
            let method = match covering_nodes.get(idx + 1) {
                Some(AnyNodeRef::StmtClassDef(class_def)) => {
                    Some(method_context_from_class(class_def, function_def)?)
                }
                _ => None,
            };
            return Some(FunctionContext {
                function_def,
                method,
            });
        }
    }
    None
}

fn method_context_from_class(
    class_def: &StmtClassDef,
    function_def: &StmtFunctionDef,
) -> Option<MethodContext> {
    let receiver_name = first_parameter_name(&function_def.parameters)?;
    Some(MethodContext {
        info: MethodInfo {
            class_name: class_def.name.id.to_string(),
            receiver_name,
        },
        is_staticmethod: function_has_decorator(function_def, "staticmethod"),
        is_classmethod: function_has_decorator(function_def, "classmethod"),
    })
}

fn parameter_is_referenced_in_body(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_path: &ModulePath,
    function_def: &StmtFunctionDef,
    parameter: &ruff_python_ast::ParameterWithDefault,
) -> bool {
    struct NameCollector {
        name: String,
        ranges: Vec<TextRange>,
    }

    impl Visitor<'_> for NameCollector {
        fn visit_stmt(&mut self, stmt: &Stmt) {
            walk_stmt(self, stmt);
        }

        fn visit_expr(&mut self, expr: &Expr) {
            if let Expr::Name(name) = expr
                && name.id.as_str() == self.name
                && !matches!(name.ctx, ExprContext::Invalid)
            {
                self.ranges.push(name.range());
            }
            walk_expr(self, expr);
        }
    }

    let mut collector = NameCollector {
        name: parameter.name().id.to_string(),
        ranges: Vec::new(),
    };
    for stmt in &function_def.body {
        collector.visit_stmt(stmt);
    }

    collector.ranges.into_iter().any(|range| {
        transaction
            .find_definition(handle, range.start(), FindPreference::default())
            .into_iter()
            .any(|definition| {
                definition.module.path() == module_path
                    && definition.definition_range == parameter.name().range()
            })
    })
}

fn supports_change_signature(function_ctx: &FunctionContext<'_>) -> bool {
    let parameters = &function_ctx.function_def.parameters;
    if !parameters.posonlyargs.is_empty()
        || !parameters.kwonlyargs.is_empty()
        || parameters.vararg.is_some()
        || parameters.kwarg.is_some()
    {
        return false;
    }
    if function_ctx.function_def.decorator_list.is_empty() {
        return true;
    }
    function_ctx
        .method
        .as_ref()
        .is_some_and(|method| method.is_staticmethod || method.is_classmethod)
}

fn build_signature_edits(
    transaction: &Transaction<'_>,
    handle: &Handle,
    function_ctx: &FunctionContext<'_>,
    editable_params: &[ParamSlot],
    receiver_text: Option<&str>,
    change: SignatureChange,
) -> Option<Vec<(pyrefly_python::module::Module, TextRange, String)>> {
    let module_info = transaction.get_module_info(handle)?;
    let new_order = apply_change_order(editable_params, change)?;
    let signature_text = render_signature(receiver_text, &new_order);
    let signature_range = parameters_inner_range(&function_ctx.function_def.parameters)?;
    let mut edits = vec![(module_info.dupe(), signature_range, signature_text)];
    edits.extend(build_callsite_edits(
        transaction,
        handle,
        &module_info,
        function_ctx,
        editable_params,
        &new_order,
    )?);
    Some(edits)
}

fn apply_change_order(params: &[ParamSlot], change: SignatureChange) -> Option<Vec<ParamSlot>> {
    let mut updated = params.to_vec();
    match change {
        SignatureChange::Remove { index } => {
            if index >= updated.len() {
                return None;
            }
            updated.remove(index);
        }
        SignatureChange::Move { index, delta } => {
            let target = index.checked_add_signed(delta)?;
            if index >= updated.len() || target >= updated.len() {
                return None;
            }
            updated.swap(index, target);
        }
    }
    Some(updated)
}

fn render_signature(receiver_text: Option<&str>, params: &[ParamSlot]) -> String {
    let mut parts = Vec::new();
    if let Some(receiver_text) = receiver_text {
        parts.push(receiver_text.to_owned());
    }
    parts.extend(params.iter().map(|param| param.text.clone()));
    parts.join(", ")
}

fn parameters_inner_range(parameters: &Parameters) -> Option<TextRange> {
    let start = parameters.range().start().checked_add(TextSize::from(1))?;
    let end = parameters.range().end().checked_sub(TextSize::from(1))?;
    Some(TextRange::new(start, end))
}

fn build_callsite_edits(
    transaction: &Transaction<'_>,
    handle: &Handle,
    module_info: &crate::module::module_info::ModuleInfo,
    function_ctx: &FunctionContext<'_>,
    old_params: &[ParamSlot],
    new_params: &[ParamSlot],
) -> Option<Vec<(pyrefly_python::module::Module, TextRange, String)>> {
    let definition = transaction
        .find_definition(
            handle,
            function_ctx.function_def.name.range.start(),
            FindPreference::default(),
        )
        .into_iter()
        .find(|def| {
            def.module.path() == module_info.path()
                && def
                    .definition_range
                    .contains_range(function_ctx.function_def.name.range)
        })?;

    let mut edits = Vec::new();
    for module_handle in transaction.handles() {
        let Some(other_module_info) = transaction.get_module_info(&module_handle) else {
            continue;
        };
        let Some(refs) = transaction.local_references_from_definition(
            &module_handle,
            definition.metadata.clone(),
            definition.definition_range,
            &definition.module,
            true,
        ) else {
            continue;
        };
        if refs.is_empty() {
            continue;
        }
        let Some(ast) = transaction.get_ast(&module_handle) else {
            continue;
        };
        let ref_set: std::collections::HashSet<TextRange> = refs.into_iter().collect();
        let mut module_edits = Vec::new();
        let mut failed = false;
        ast.as_ref().visit(&mut |expr| {
            if failed {
                return;
            }
            let Expr::Call(call) = expr else {
                return;
            };
            if !ref_set
                .iter()
                .any(|range| call.func.range().contains(range.start()))
            {
                return;
            }
            let Some(new_call) = rewrite_call_arguments(
                call,
                other_module_info.contents(),
                function_ctx,
                old_params,
                new_params,
            ) else {
                failed = true;
                return;
            };
            module_edits.push((other_module_info.dupe(), call.arguments.range(), new_call));
        });
        if failed {
            return None;
        }
        edits.extend(module_edits);
    }
    Some(edits)
}

fn rewrite_call_arguments(
    call: &ExprCall,
    source: &str,
    function_ctx: &FunctionContext<'_>,
    old_params: &[ParamSlot],
    new_params: &[ParamSlot],
) -> Option<String> {
    if call.arguments.args.iter().any(|arg| arg.is_starred_expr())
        || call.arguments.keywords.iter().any(|kw| kw.arg.is_none())
    {
        return None;
    }
    let explicit_receiver = explicit_receiver_text(call, source, function_ctx.method.as_ref())?;
    let positional_skip = explicit_receiver.is_some() as usize;
    let mut parts = Vec::new();
    if let Some(receiver) = explicit_receiver {
        parts.push(receiver);
    }
    for param in new_params {
        let old_index = old_params
            .iter()
            .position(|candidate| candidate.name == param.name)?;
        let argument = if let Some(keyword) = call.arguments.find_keyword(param.name.as_str()) {
            Some(&keyword.value)
        } else {
            call.arguments.find_positional(old_index + positional_skip)
        };
        let Some(argument) = argument else {
            continue;
        };
        let argument_text = code_at_range(source, argument.range())?.to_owned();
        parts.push(format!("{}={}", param.name, argument_text));
    }
    Some(format!("({})", parts.join(", ")))
}

fn explicit_receiver_text(
    call: &ExprCall,
    source: &str,
    method_ctx: Option<&MethodContext>,
) -> Option<Option<String>> {
    let Some(method_ctx) = method_ctx else {
        return Some(None);
    };
    if method_ctx.is_staticmethod || method_ctx.is_classmethod {
        return Some(None);
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(name) = attribute.value.as_ref() else {
        return Some(None);
    };
    if name.id.as_str() != method_ctx.info.class_name {
        return Some(None);
    }
    let receiver = call.arguments.find_positional(0)?;
    Some(Some(code_at_range(source, receiver.range())?.to_owned()))
}
