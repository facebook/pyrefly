/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::dedent_block_preserving_layout;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::types::Type;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use super::extract_shared::line_indent_and_start;
use super::extract_shared::selection_anchor;
use crate::binding::binding::KeyClass;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::lsp::Transaction;

const DEFAULT_INDENT: &str = "    ";

#[derive(Clone)]
struct AbstractMemberInfo {
    name: Name,
    ty: Type,
    defining_module: pyrefly_python::module::Module,
    docstring_range: Option<TextRange>,
}

/// Builds a quick fix that implements all inherited abstract members in a class.
pub(crate) fn implement_abstract_members_code_actions(
    transaction: &Transaction<'_>,
    handle: &Handle,
    selection: TextRange,
) -> Option<Vec<LocalRefactorCodeAction>> {
    let module_info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;
    let source = module_info.contents();
    let selection_point = selection_anchor(source, selection);
    let class_def = find_class_context(ast.as_ref(), selection_point)?;
    let (method_indent, insert_range) = insertion_point(class_def, source)?;
    let indent_unit = class_indent_unit(source, class_def, &method_indent);

    let bindings = transaction.get_bindings(handle)?;
    let key = KeyClass(ShortIdentifier::new(&class_def.name));
    let class_idx = bindings.key_to_idx_hashed_opt(starlark_map::Hashed::new(&key))?;

    let members = transaction
        .ad_hoc_solve(handle, |solver| {
            let class = solver.get_idx(class_idx).0.clone()?;
            let abstract_members = solver.get_abstract_members_for_class(&class);
            let mut infos = Vec::new();
            for name in abstract_members.unimplemented_abstract_methods() {
                let member = solver.get_class_member_with_defining_class(&class, name)?;
                infos.push(AbstractMemberInfo {
                    name: name.clone(),
                    ty: member.value.ty(),
                    defining_module: member.defining_class.module().dupe(),
                    docstring_range: member.defining_class.field_docstring_range(name),
                });
            }
            Some(infos)
        })
        .flatten()?;

    if members.is_empty() {
        return None;
    }

    let mut members = members;
    members.sort_by(|left, right| left.name.cmp(&right.name));
    let mut member_blocks = Vec::with_capacity(members.len());
    for member in members {
        member_blocks.push(build_member_block(&member, &method_indent, &indent_unit));
    }
    let mut combined = member_blocks.join("\n\n");
    combined.push('\n');

    Some(vec![LocalRefactorCodeAction {
        title: "Implement abstract members".to_owned(),
        edits: vec![(module_info.dupe(), insert_range, combined)],
        kind: lsp_types::CodeActionKind::QUICKFIX,
    }])
}

fn find_class_context<'a>(ast: &'a ModModule, selection: TextSize) -> Option<&'a StmtClassDef> {
    for stmt in &ast.body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
        {
            if let Some(inner) = find_class_in_body(&class_def.body, selection) {
                return Some(inner);
            }
            return Some(class_def);
        }
    }
    None
}

fn find_class_in_body<'a>(body: &'a [Stmt], selection: TextSize) -> Option<&'a StmtClassDef> {
    for stmt in body {
        if let Stmt::ClassDef(class_def) = stmt
            && class_def.range().contains(selection)
        {
            if let Some(inner) = find_class_in_body(&class_def.body, selection) {
                return Some(inner);
            }
            return Some(class_def);
        }
    }
    None
}

fn insertion_point(class_def: &StmtClassDef, source: &str) -> Option<(String, TextRange)> {
    if let Some(pass_stmt) = replaceable_pass_stmt(class_def) {
        let (indent, line_start) = line_indent_and_start(source, pass_stmt.range().start())?;
        let line_end = line_end_position(source, pass_stmt.range().end());
        return Some((indent, TextRange::new(line_start, line_end)));
    }
    for stmt in &class_def.body {
        if is_docstring_stmt(stmt) {
            continue;
        }
        let (indent, line_start) = line_indent_and_start(source, stmt.range().start())?;
        return Some((indent, TextRange::at(line_start, TextSize::new(0))));
    }
    if let Some(docstring) = class_def.body.first() {
        let (indent, _) = line_indent_and_start(source, docstring.range().start())?;
        let line_end = line_end_position(source, docstring.range().end());
        return Some((indent, TextRange::at(line_end, TextSize::new(0))));
    }
    let (class_indent, _) = line_indent_and_start(source, class_def.range().start())?;
    let insert_position = line_end_position(source, class_def.range().end());
    Some((
        format!("{class_indent}{DEFAULT_INDENT}"),
        TextRange::at(insert_position, TextSize::new(0)),
    ))
}

fn replaceable_pass_stmt(class_def: &StmtClassDef) -> Option<&Stmt> {
    let mut non_docstring = class_def
        .body
        .iter()
        .filter(|stmt| !is_docstring_stmt(stmt));
    let only_stmt = non_docstring.next()?;
    if non_docstring.next().is_some() {
        return None;
    }
    match only_stmt {
        Stmt::Pass(_) => Some(only_stmt),
        _ => None,
    }
}

fn class_indent_unit(source: &str, class_def: &StmtClassDef, method_indent: &str) -> String {
    let (class_indent, _) = match line_indent_and_start(source, class_def.range().start()) {
        Some(value) => value,
        None => return DEFAULT_INDENT.to_owned(),
    };
    if let Some(unit) = method_indent.strip_prefix(&class_indent)
        && !unit.is_empty()
    {
        return unit.to_owned();
    }
    DEFAULT_INDENT.to_owned()
}

fn line_end_position(source: &str, position: TextSize) -> TextSize {
    let mut idx = position.to_usize().min(source.len());
    let bytes = source.as_bytes();
    while idx < bytes.len() && bytes[idx] != b'\n' {
        idx += 1;
    }
    if idx < bytes.len() {
        idx += 1;
    }
    TextSize::try_from(idx).unwrap_or(position)
}

fn build_member_block(member: &AbstractMemberInfo, indent: &str, indent_unit: &str) -> String {
    let mut block = String::new();
    let property_getter = member.ty.is_property_setter_with_getter();
    let is_property = member.ty.is_property_getter()
        || property_getter.is_some()
        || member.ty.is_cached_property();
    let is_staticmethod = if is_property {
        block.push_str(indent);
        if member.ty.is_cached_property() {
            block.push_str("@cached_property\n");
        } else {
            block.push_str("@property\n");
        }
        false
    } else {
        let is_classmethod = member
            .ty
            .visit_toplevel_func_metadata(&|meta| meta.flags.is_classmethod);
        let is_staticmethod = member
            .ty
            .visit_toplevel_func_metadata(&|meta| meta.flags.is_staticmethod);
        if is_classmethod {
            block.push_str(indent);
            block.push_str("@classmethod\n");
        } else if is_staticmethod {
            block.push_str(indent);
            block.push_str("@staticmethod\n");
        }
        is_staticmethod
    };

    let signature_ty = property_getter.unwrap_or_else(|| member.ty.clone());
    let drop_receiver_annotation = !is_staticmethod;
    let (params, return_type) = format_member_signature(&signature_ty, drop_receiver_annotation);
    block.push_str(indent);
    block.push_str("def ");
    block.push_str(member.name.as_str());
    block.push('(');
    block.push_str(&params);
    block.push(')');
    if let Some(return_type) = return_type {
        block.push_str(" -> ");
        block.push_str(&return_type);
    }
    block.push_str(":\n");

    let body_indent = format!("{indent}{indent_unit}");
    if let Some(range) = member.docstring_range
        && let Some(docstring) =
            dedent_block_preserving_layout(member.defining_module.code_at(range))
    {
        let mut lines: Vec<&str> = docstring.lines().collect();
        while lines.last().is_some_and(|line| line.trim().is_empty()) {
            lines.pop();
        }
        for line in lines {
            block.push_str(&body_indent);
            block.push_str(line);
            block.push('\n');
        }
    }
    block.push_str(&body_indent);
    block.push_str("raise NotImplementedError");
    block
}

fn format_member_signature(ty: &Type, drop_receiver_annotation: bool) -> (String, Option<String>) {
    let callables = ty.callable_signatures();
    let Some(callable) = callables.first() else {
        return ("*args, **kwargs".to_owned(), None);
    };
    match &callable.params {
        Params::List(params) => {
            let mut display_types: Vec<&Type> =
                params.items().iter().map(|param| param.as_type()).collect();
            display_types.push(&callable.ret);
            let type_ctx = TypeDisplayContext::new(&display_types);
            let params_buf = format_param_list(params, &type_ctx, drop_receiver_annotation);
            let ret = type_ctx.display(&callable.ret).to_string();
            (params_buf, Some(ret))
        }
        _ => ("*args, **kwargs".to_owned(), None),
    }
}

fn format_param_list(
    params: &ParamList,
    type_ctx: &TypeDisplayContext,
    drop_receiver_annotation: bool,
) -> String {
    let mut buf = String::new();
    let mut named_posonly = false;
    let mut kwonly = false;
    for (i, param) in params.items().iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        if matches!(param, Param::PosOnly(Some(_), _, _)) {
            named_posonly = true;
        } else if named_posonly {
            named_posonly = false;
            buf.push_str("/, ");
        }
        if !kwonly && matches!(param, Param::KwOnly(..)) {
            kwonly = true;
            buf.push_str("*, ");
        }
        let should_strip_receiver = drop_receiver_annotation
            && i == 0
            && matches!(
                param,
                Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _)
                    if matches!(name.as_str(), "self" | "cls")
            );
        if should_strip_receiver {
            let name = match param {
                Param::PosOnly(Some(name), _, _) | Param::Pos(name, _, _) => name.as_str(),
                _ => "",
            };
            buf.push_str(name);
        } else {
            buf.push_str(&param.format_for_signature(type_ctx));
        }
    }
    if named_posonly {
        buf.push_str(", /");
    }
    buf
}
