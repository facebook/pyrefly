/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Extracts stub declarations from a type-checked module.
//!
//! Walks the module's AST in source order and uses the binding/answer
//! system to resolve types for each declaration.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::Param;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::Answers;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Key;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllEntry;
use crate::export::definitions::DunderAllKind;
use crate::state::state::Transaction;

/// A single module's stub content, in source order.
pub struct ModuleStub {
    pub items: Vec<StubItem>,
    /// Whether any item uses `Incomplete` (so we know whether to
    /// emit `from _typeshed import Incomplete`).
    pub uses_incomplete: bool,
}

pub enum StubItem {
    Import(StubImport),
    Function(StubFunction),
    Class(StubClass),
    Variable(StubVariable),
    TypeAlias(StubTypeAlias),
}

pub struct StubImport {
    pub text: String,
}

pub struct StubFunction {
    pub name: String,
    pub is_async: bool,
    pub type_params: Option<String>,
    pub decorators: Vec<String>,
    pub params: Vec<StubParam>,
    pub return_type: Option<String>,
    pub docstring: Option<String>,
}

pub struct StubParam {
    pub prefix: &'static str,
    pub name: String,
    pub annotation: Option<String>,
    pub default: Option<String>,
}

pub struct StubClass {
    pub name: String,
    pub type_params: Option<String>,
    pub bases: String,
    pub decorators: Vec<String>,
    pub body: Vec<StubItem>,
    pub docstring: Option<String>,
}

pub struct StubVariable {
    pub name: String,
    pub annotation: Option<String>,
    pub value: Option<String>,
}

pub struct StubTypeAlias {
    /// e.g. `type Vector = list[float]`.
    pub text: String,
}

/// Configuration for stub extraction.
pub struct ExtractConfig {
    pub include_private: bool,
    pub include_docstrings: bool,
}

/// Extract a `ModuleStub` from a type-checked module.
pub fn extract_module_stub(
    transaction: &Transaction,
    handle: &Handle,
    config: &ExtractConfig,
) -> Option<ModuleStub> {
    let bindings = transaction.get_bindings(handle)?;
    let answers = transaction.get_answers(handle)?;
    let ast = transaction.get_ast(handle)?;
    let module_info = transaction.get_module_info(handle)?;

    let function_map: HashMap<TextRange, DecoratedFunction> = bindings
        .keys::<KeyDecoratedFunction>()
        .map(|idx| {
            let dec = DecoratedFunction::from_bindings_answers(idx, &bindings, &answers);
            (dec.id_range(), dec)
        })
        .collect();

    let dunder_all = resolve_dunder_all(&ast.body, &module_info);

    let mut ctx = ExtractionContext {
        bindings: &bindings,
        answers: &answers,
        module_info: &module_info,
        config,
        uses_incomplete: false,
        function_map: &function_map,
        dunder_all: &dunder_all,
    };

    let items = extract_stmts(&ast.body, &mut ctx, false);

    Some(ModuleStub {
        items,
        uses_incomplete: ctx.uses_incomplete,
    })
}

struct ExtractionContext<'a> {
    bindings: &'a Bindings,
    answers: &'a Arc<Answers>,
    module_info: &'a Module,
    config: &'a ExtractConfig,
    uses_incomplete: bool,
    function_map: &'a HashMap<TextRange, DecoratedFunction>,
    /// When `__all__` is explicitly defined, only these names are exported
    /// at module level. `None` means no explicit `__all__` — use convention.
    dunder_all: &'a Option<HashSet<Name>>,
}

fn extract_stmts(stmts: &[Stmt], ctx: &mut ExtractionContext, in_class: bool) -> Vec<StubItem> {
    let mut items = Vec::new();
    let overloaded = collect_overloaded_names(stmts);

    for stmt in stmts {
        if is_overload_impl(stmt, &overloaded) {
            continue;
        }

        match stmt {
            Stmt::FunctionDef(func_def) => {
                if let Some(item) = extract_function(func_def, ctx, in_class) {
                    items.push(StubItem::Function(item));
                }
            }
            Stmt::ClassDef(class_def) => {
                if let Some(item) = extract_class(class_def, ctx) {
                    items.push(StubItem::Class(item));
                }
            }
            Stmt::Import(import) => {
                let text = source_text(ctx.module_info, import.range()).to_owned();
                items.push(StubItem::Import(StubImport { text }));
            }
            Stmt::ImportFrom(import) => {
                let text = source_text(ctx.module_info, import.range()).to_owned();
                items.push(StubItem::Import(StubImport { text }));
            }
            Stmt::AnnAssign(ann_assign) => {
                if let Some(item) = extract_ann_assign(ann_assign, ctx, in_class) {
                    items.push(StubItem::Variable(item));
                }
            }
            Stmt::Assign(assign) => {
                // TypeVar/NamedTuple/TypedDict calls and old-style type aliases
                // (e.g. `X = List[int]`, `X = int | str`) are preserved verbatim.
                if is_type_constructor_or_alias(assign) {
                    if let [Expr::Name(n)] = assign.targets.as_slice()
                        && !should_include_name(n.id.as_str(), ctx.config, in_class, ctx.dunder_all)
                    {
                        continue;
                    }
                    let text = source_text(ctx.module_info, assign.range()).to_owned();
                    items.push(StubItem::TypeAlias(StubTypeAlias { text }));
                } else {
                    for item in extract_assign(assign, ctx, in_class) {
                        items.push(StubItem::Variable(item));
                    }
                }
            }
            Stmt::TypeAlias(type_alias) => {
                if let Expr::Name(n) = type_alias.name.as_ref()
                    && !should_include_name(n.id.as_str(), ctx.config, in_class, ctx.dunder_all)
                {
                    continue;
                }
                let text = source_text(ctx.module_info, type_alias.range()).to_owned();
                items.push(StubItem::TypeAlias(StubTypeAlias { text }));
            }
            Stmt::If(if_stmt) => {
                if is_type_checking_guard(&if_stmt.test) {
                    items.extend(extract_stmts(&if_stmt.body, ctx, in_class));
                }
            }
            _ => {}
        }
    }

    items
}

fn is_type_checking_guard(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id == "TYPE_CHECKING",
        Expr::Attribute(attr) => attr.attr.as_str() == "TYPE_CHECKING",
        _ => false,
    }
}

fn extract_function(
    func_def: &StmtFunctionDef,
    ctx: &mut ExtractionContext,
    in_class: bool,
) -> Option<StubFunction> {
    let name = func_def.name.id.as_str();
    if !should_include_name(name, ctx.config, in_class, ctx.dunder_all) {
        return None;
    }

    let decorated = ctx.function_map.get(&func_def.name.range());

    let decorators: Vec<String> = func_def
        .decorator_list
        .iter()
        .map(|d| format!("@{}", source_text(ctx.module_info, d.expression.range())))
        .collect();

    let params = extract_params(func_def, decorated, ctx);
    let return_type = extract_return_type(func_def, decorated, ctx);
    let docstring = if ctx.config.include_docstrings {
        extract_docstring(&func_def.body)
    } else {
        None
    };

    // Extract PEP 695 type parameters (e.g. `def foo[T](...)`).
    let type_params = func_def
        .type_params
        .as_ref()
        .map(|tp| source_text(ctx.module_info, tp.range()).to_owned());

    Some(StubFunction {
        name: name.to_owned(),
        is_async: func_def.is_async,
        type_params,
        decorators,
        params,
        return_type,
        docstring,
    })
}

/// Enrich parameters with inferred types where source annotations are missing.
fn extract_params(
    func_def: &StmtFunctionDef,
    decorated: Option<&DecoratedFunction>,
    ctx: &mut ExtractionContext,
) -> Vec<StubParam> {
    let ast_params = &func_def.parameters;
    let mut result = Vec::new();

    let resolved_map: HashMap<&str, &Param> = decorated
        .map(|d| {
            d.undecorated
                .params
                .iter()
                .filter_map(|p| p.name().map(|n| (n.as_str(), p)))
                .collect()
        })
        .unwrap_or_default();

    for pwd in &ast_params.posonlyargs {
        result.push(make_param(
            "",
            &pwd.parameter.name.id,
            pwd.parameter.annotation.as_deref(),
            pwd.default.as_deref(),
            resolved_map.get(pwd.parameter.name.id.as_str()).copied(),
            ctx,
        ));
    }
    if !ast_params.posonlyargs.is_empty() {
        result.push(StubParam {
            prefix: "",
            name: "/".to_owned(),
            annotation: None,
            default: None,
        });
    }

    for pwd in &ast_params.args {
        result.push(make_param(
            "",
            &pwd.parameter.name.id,
            pwd.parameter.annotation.as_deref(),
            pwd.default.as_deref(),
            resolved_map.get(pwd.parameter.name.id.as_str()).copied(),
            ctx,
        ));
    }

    if let Some(vararg) = &ast_params.vararg {
        result.push(make_param(
            "*",
            &vararg.name.id,
            vararg.annotation.as_deref(),
            None,
            resolved_map.get(vararg.name.id.as_str()).copied(),
            ctx,
        ));
    } else if !ast_params.kwonlyargs.is_empty() {
        result.push(StubParam {
            prefix: "",
            name: "*".to_owned(),
            annotation: None,
            default: None,
        });
    }

    for pwd in &ast_params.kwonlyargs {
        result.push(make_param(
            "",
            &pwd.parameter.name.id,
            pwd.parameter.annotation.as_deref(),
            pwd.default.as_deref(),
            resolved_map.get(pwd.parameter.name.id.as_str()).copied(),
            ctx,
        ));
    }

    if let Some(kwarg) = &ast_params.kwarg {
        result.push(make_param(
            "**",
            &kwarg.name.id,
            kwarg.annotation.as_deref(),
            None,
            resolved_map.get(kwarg.name.id.as_str()).copied(),
            ctx,
        ));
    }

    result
}

/// Prefer source annotation, fall back to inferred type from the binding system.
fn make_param(
    prefix: &'static str,
    name: &Name,
    source_annotation: Option<&Expr>,
    default: Option<&Expr>,
    resolved: Option<&Param>,
    ctx: &mut ExtractionContext,
) -> StubParam {
    let annotation = if let Some(ann_expr) = source_annotation {
        Some(source_text(ctx.module_info, ann_expr.range()).to_owned())
    } else if let Some(param) = resolved {
        format_param_type(param, ctx)
    } else {
        None
    };

    let default_str = default.map(|d| format_default(d, ctx.module_info));

    StubParam {
        prefix,
        name: name.to_string(),
        annotation,
        default: default_str,
    }
}

/// Format a `Param`'s type for use in a stub, or return `None` for
/// `self`/`cls` parameters and unresolvable types.
fn format_param_type(param: &Param, ctx: &mut ExtractionContext) -> Option<String> {
    let ty = param.as_type();
    if let Some(name) = param.name()
        && (name == "self" || name == "cls")
    {
        return None;
    }
    format_type(ty, ctx)
}

/// Returns `Incomplete` for Any and unresolvable types.
fn format_type(ty: &Type, ctx: &mut ExtractionContext) -> Option<String> {
    if ty.is_any() {
        ctx.uses_incomplete = true;
        return Some("Incomplete".to_owned());
    }
    let s = ty.to_string();
    if s.contains("@") || s.contains("Unknown") {
        ctx.uses_incomplete = true;
        return Some("Incomplete".to_owned());
    }
    Some(s)
}

/// Uses source text for simple literals, `...` for everything else.
fn format_default(expr: &Expr, module_info: &Module) -> String {
    match expr {
        Expr::NoneLiteral(_) => "None".to_owned(),
        Expr::BooleanLiteral(b) => {
            if b.value {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Expr::NumberLiteral(_) | Expr::StringLiteral(_) | Expr::BytesLiteral(_) => {
            source_text(module_info, expr.range()).to_owned()
        }
        Expr::UnaryOp(u) => {
            if matches!(u.op, ruff_python_ast::UnaryOp::USub)
                && matches!(u.operand.as_ref(), Expr::NumberLiteral(_))
            {
                source_text(module_info, expr.range()).to_owned()
            } else {
                "...".to_owned()
            }
        }
        Expr::Tuple(t) if t.elts.is_empty() => "()".to_owned(),
        Expr::EllipsisLiteral(_) => "...".to_owned(),
        _ => "...".to_owned(),
    }
}

/// Prefer source annotation, fall back to inferred return type.
fn extract_return_type(
    func_def: &StmtFunctionDef,
    decorated: Option<&DecoratedFunction>,
    ctx: &mut ExtractionContext,
) -> Option<String> {
    if let Some(returns) = &func_def.returns {
        let expr: &Expr = returns;
        return Some(source_text(ctx.module_info, expr.range()).to_owned());
    }

    if decorated.is_some() {
        let short_id = ShortIdentifier::new(&func_def.name);
        let ret_key = Key::ReturnType(short_id);
        if let Some(idx) = ctx
            .bindings
            .key_to_idx_hashed_opt(starlark_map::Hashed::new(&ret_key))
            && let Some(ty) = ctx.answers.get_type_at(idx)
        {
            return format_type(&ty, ctx);
        }
    }

    None
}

fn extract_class(class_def: &StmtClassDef, ctx: &mut ExtractionContext) -> Option<StubClass> {
    let name = class_def.name.id.as_str();
    if !should_include_name(name, ctx.config, false, ctx.dunder_all) {
        return None;
    }

    let decorators: Vec<String> = class_def
        .decorator_list
        .iter()
        .map(|d| format!("@{}", source_text(ctx.module_info, d.expression.range())))
        .collect();

    let bases = if let Some(args) = &class_def.arguments {
        let mut parts: Vec<String> = Vec::new();
        for a in &args.args {
            let expr: &Expr = a;
            parts.push(source_text(ctx.module_info, expr.range()).to_owned());
        }
        for kw in &args.keywords {
            let val_expr: &Expr = &kw.value;
            if let Some(arg) = &kw.arg {
                parts.push(format!(
                    "{}={}",
                    arg.as_str(),
                    source_text(ctx.module_info, val_expr.range())
                ));
            } else {
                parts.push(format!(
                    "**{}",
                    source_text(ctx.module_info, val_expr.range())
                ));
            }
        }
        parts.join(", ")
    } else {
        String::new()
    };

    let docstring = if ctx.config.include_docstrings {
        extract_docstring(&class_def.body)
    } else {
        None
    };

    let type_params = class_def
        .type_params
        .as_ref()
        .map(|tp| source_text(ctx.module_info, tp.range()).to_owned());

    let body = extract_stmts(&class_def.body, ctx, true);

    Some(StubClass {
        name: name.to_owned(),
        type_params,
        bases,
        decorators,
        body,
        docstring,
    })
}

fn extract_ann_assign(
    ann_assign: &ruff_python_ast::StmtAnnAssign,
    ctx: &mut ExtractionContext,
    in_class: bool,
) -> Option<StubVariable> {
    let name = match ann_assign.target.as_ref() {
        Expr::Name(n) => n.id.as_str(),
        _ => return None,
    };
    if !should_include_name(name, ctx.config, in_class, ctx.dunder_all) {
        return None;
    }

    let annotation = source_text(ctx.module_info, ann_assign.annotation.range()).to_owned();

    let value = ann_assign
        .value
        .as_deref()
        .and_then(|v| simple_value_text(v, ctx.module_info));

    Some(StubVariable {
        name: name.to_owned(),
        annotation: Some(annotation),
        value,
    })
}

fn extract_assign(
    assign: &ruff_python_ast::StmtAssign,
    ctx: &mut ExtractionContext,
    in_class: bool,
) -> Vec<StubVariable> {
    let mut result = Vec::new();

    for target in &assign.targets {
        if let Expr::Name(name_expr) = target {
            let name = name_expr.id.as_str();
            if !should_include_name(name, ctx.config, in_class, ctx.dunder_all) {
                continue;
            }

            if name == "__all__" {
                continue;
            }

            let short_id = ShortIdentifier::expr_name(name_expr);
            let def_key = Key::Definition(short_id);
            let annotation = ctx
                .bindings
                .key_to_idx_hashed_opt(starlark_map::Hashed::new(&def_key))
                .and_then(|idx| ctx.answers.get_type_at(idx))
                .and_then(|ty| format_type(&ty, ctx));

            let value = simple_value_text(&assign.value, ctx.module_info);

            if annotation.is_some() || value.is_some() {
                result.push(StubVariable {
                    name: name.to_owned(),
                    annotation,
                    value,
                });
            }
        }
    }

    result
}

/// Returns `None` for complex expressions.
fn simple_value_text(expr: &Expr, module_info: &Module) -> Option<String> {
    match expr {
        Expr::NoneLiteral(_) => Some("None".to_owned()),
        Expr::BooleanLiteral(b) => Some(if b.value {
            "True".to_owned()
        } else {
            "False".to_owned()
        }),
        Expr::NumberLiteral(_) | Expr::StringLiteral(_) | Expr::BytesLiteral(_) => {
            Some(source_text(module_info, expr.range()).to_owned())
        }
        Expr::EllipsisLiteral(_) => Some("...".to_owned()),
        _ => None,
    }
}

fn extract_docstring(body: &[Stmt]) -> Option<String> {
    if let Some(Stmt::Expr(expr_stmt)) = body.first()
        && let Expr::StringLiteral(s) = expr_stmt.value.as_ref()
    {
        return Some(format!("\"\"\"{}\"\"\"", s.value));
    }
    None
}

/// Parse `__all__` from the module AST using the existing `Definitions`
/// infrastructure. Returns `Some(names)` when the module explicitly defines
/// `__all__` and it can be statically resolved, `None` otherwise.
fn resolve_dunder_all(body: &[Stmt], module_info: &Module) -> Option<HashSet<Name>> {
    let defs = Definitions::new(
        body,
        module_info.name(),
        module_info.path().is_init(),
        SysInfo::default(),
    );
    if !matches!(defs.dunder_all.kind, DunderAllKind::Specified) {
        return None;
    }
    let mut names = HashSet::new();
    for entry in &defs.dunder_all.entries {
        match entry {
            DunderAllEntry::Name(_, name) => {
                names.insert(name.clone());
            }
            DunderAllEntry::Remove(_, name) => {
                names.remove(name);
            }
            DunderAllEntry::Module(..) => {
                // Cross module __all__ re-exports require import resolution;
                // fall back to convention-based filtering.
                return None;
            }
        }
    }
    Some(names)
}

fn should_include_name(
    name: &str,
    config: &ExtractConfig,
    in_class: bool,
    dunder_all: &Option<HashSet<Name>>,
) -> bool {
    // At module level with an explicit `__all__`, only export listed names.
    if !in_class && let Some(all_names) = dunder_all {
        return all_names.contains(name);
    }

    // Convention-based filtering when no explicit `__all__` is present.
    // Dunder names are always part of the public protocol.
    if name.starts_with("__") && name.ends_with("__") {
        return true;
    }
    // Double-underscore names are name-mangled in classes but private at module level.
    if name.starts_with("__") && !in_class {
        return false;
    }
    if name.starts_with('_') && !config.include_private {
        return false;
    }
    true
}

/// Matches `@overload`, `@typing.overload`, and `@typing_extensions.overload`.
fn has_overload_decorator(func_def: &StmtFunctionDef) -> bool {
    func_def.decorator_list.iter().any(|d| {
        match &d.expression {
            Expr::Name(name) => name.id.as_str() == "overload",
            Expr::Attribute(attr) => {
                attr.attr.as_str() == "overload"
                    && matches!(&*attr.value, Expr::Name(base) if base.id.as_str() == "typing" || base.id.as_str() == "typing_extensions")
            }
            _ => false,
        }
    })
}

/// Collect function names that have at least one `@overload` variant,
/// so the non-overloaded implementation can be dropped from the stub.
fn collect_overloaded_names(stmts: &[Stmt]) -> HashSet<String> {
    let mut names = HashSet::new();
    for stmt in stmts {
        if let Stmt::FunctionDef(func) = stmt
            && has_overload_decorator(func)
        {
            names.insert(func.name.to_string());
        }
    }
    names
}

fn is_overload_impl(stmt: &Stmt, overloaded: &HashSet<String>) -> bool {
    if let Stmt::FunctionDef(func) = stmt {
        overloaded.contains(func.name.as_str()) && !has_overload_decorator(func)
    } else {
        false
    }
}

/// Returns `true` for calls to type-variable constructors (`TypeVar`,
/// `ParamSpec`, `TypeVarTuple`, `NewType`, `NamedTuple`, `TypedDict`).
fn is_type_constructor_call(value: &Expr) -> bool {
    if let Expr::Call(call) = value
        && let Expr::Name(name) = &*call.func
    {
        return matches!(
            name.id.as_str(),
            "TypeVar" | "ParamSpec" | "TypeVarTuple" | "NewType" | "NamedTuple" | "TypedDict"
        );
    }
    false
}

/// Subscripts (`List[int]`) or union pipes (`int | str`).
/// This is intentionally broad — false positives (e.g. `x = d["key"]`)
/// are benign since they just preserve the assignment verbatim.
fn is_old_style_type_alias(value: &Expr) -> bool {
    match value {
        Expr::Subscript(_) => true,
        Expr::BinOp(op) if op.op == Operator::BitOr => true,
        _ => false,
    }
}

/// Type constructor calls and old-style type aliases are preserved verbatim.
fn is_type_constructor_or_alias(assign: &ruff_python_ast::StmtAssign) -> bool {
    if let [Expr::Name(_)] = assign.targets.as_slice() {
        is_type_constructor_call(&assign.value) || is_old_style_type_alias(&assign.value)
    } else {
        false
    }
}

fn source_text(module_info: &Module, range: TextRange) -> &str {
    module_info.code_at(range)
}
