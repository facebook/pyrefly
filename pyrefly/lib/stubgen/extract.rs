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
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::Required;
use pyrefly_types::display::TypeDisplayContext;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;

use crate::alt::answers::Answers;
use crate::alt::class::class_field::ClassField;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::ClassSynthesizedFields;
use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllEntry;
use crate::export::definitions::DunderAllKind;
use crate::state::state::Transaction;
use crate::types::class::Class;

/// A single module's stub content, in source order.
pub struct ModuleStub {
    pub items: Vec<StubItem>,
    /// Whether any item uses `Incomplete` (so we know whether to
    /// emit `from _typeshed import Incomplete`).
    pub uses_incomplete: bool,
    pub uses_self: bool,
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
        uses_self: false,
        function_map: &function_map,
        dunder_all: &dunder_all,
        current_class: None,
    };

    let mut items = extract_stmts(&ast.body, &mut ctx, false);
    prune_stub_imports(&mut items);

    Some(ModuleStub {
        items,
        uses_incomplete: ctx.uses_incomplete,
        uses_self: ctx.uses_self,
    })
}

struct ExtractionContext<'a> {
    bindings: &'a Bindings,
    answers: &'a Arc<Answers>,
    module_info: &'a Module,
    config: &'a ExtractConfig,
    uses_incomplete: bool,
    uses_self: bool,
    function_map: &'a HashMap<TextRange, DecoratedFunction>,
    /// When `__all__` is explicitly defined, only these names are exported
    /// at module level. `None` means no explicit `__all__` — use convention.
    dunder_all: &'a Option<HashSet<Name>>,
    /// Innermost class being extracted (`None` at module scope).
    current_class: Option<Class>,
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
            Stmt::If(if_stmt) if is_type_checking_guard(&if_stmt.test) => {
                items.extend(extract_stmts(&if_stmt.body, ctx, in_class));
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
        .filter(|d| !(in_class && is_computed_field_decorator(&d.expression)))
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
    if ty.any(|sub_type| matches!(sub_type, Type::SelfType(_))) {
        ctx.uses_self = true;
    }
    let mut display = TypeDisplayContext::new(&[ty]);
    display.render_self_type_as_self();
    let s = display.display(ty).to_string();
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

/// Names already represented as variables in the extracted class body.
fn stub_class_level_variable_names(body: &[StubItem]) -> HashSet<&str> {
    let mut out = HashSet::new();
    for item in body {
        if let StubItem::Variable(v) = item {
            out.insert(v.name.as_str());
        }
    }
    out
}

/// Instance attributes inferred from methods (e.g. `self.name` in `__init__`) that
/// are not already declared in the class body, materialized as stub `name: T` lines.
///
/// Emitted in alphabetical order by field name (stable regardless of map / inference order).
fn extract_instance_attr_stubs_from_class_fields(
    class_def: &StmtClassDef,
    class_body: &[StubItem],
    ctx: &mut ExtractionContext,
) -> Vec<StubVariable> {
    let Some(def_index) = ctx.bindings.class_def_index(class_def) else {
        return Vec::new();
    };
    let already = stub_class_level_variable_names(class_body);
    let class_fields = match ctx.bindings.get_class_fields(def_index) {
        Some(f) => f,
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    for (name, _) in class_fields.iter() {
        if already.contains(name.as_str()) {
            continue;
        }
        if !should_include_name(name.as_str(), ctx.config, true, ctx.dunder_all) {
            continue;
        }
        let key = KeyClassField(def_index, name.clone());
        let Some(field_idx) = ctx.bindings.key_to_idx_hashed_opt(Hashed::new(&key)) else {
            continue;
        };
        let Some(field) = ctx.answers.get_idx(field_idx) else {
            continue;
        };
        if !field.is_simple_instance_attribute() {
            continue;
        }
        let Some(ann) = format_type(&field.ty(), ctx) else {
            continue;
        };
        out.push(StubVariable {
            name: name.to_string(),
            annotation: Some(ann),
            value: None,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Inserts synthesized instance attribute stubs before `__init__` when present,
/// otherwise at the start of the class body.
fn merge_instance_field_stubs(
    synthesized: Vec<StubVariable>,
    mut body: Vec<StubItem>,
) -> Vec<StubItem> {
    if synthesized.is_empty() {
        return body;
    }
    let init_idx = body
        .iter()
        .position(|item| matches!(item, StubItem::Function(f) if f.name == "__init__"));
    let synth: Vec<StubItem> = synthesized.into_iter().map(StubItem::Variable).collect();
    match init_idx {
        Some(i) => {
            body.splice(i..i, synth);
            body
        }
        None => {
            body.splice(0..0, synth);
            body
        }
    }
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

    let resolved_class = resolve_class_for_stub(class_def, ctx);
    let outer_class = ctx.current_class.clone();
    ctx.current_class = resolved_class.clone();

    let body = extract_class_body(class_def, ctx, resolved_class.as_ref());

    ctx.current_class = outer_class;

    let extra = extract_instance_attr_stubs_from_class_fields(class_def, &body, ctx);
    let body = merge_instance_field_stubs(extra, body);

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

    let value = match ann_assign.value.as_deref() {
        None => None,
        Some(v) => {
            if let Some(simple) = simple_value_text(v, ctx.module_info) {
                Some(simple)
            } else if in_class {
                complex_class_attribute_stub_value(name, ctx)
            } else {
                None
            }
        }
    };

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

fn is_computed_field_decorator(expr: &Expr) -> bool {
    match expr {
        Expr::Name(n) => n.id.as_str() == "computed_field",
        Expr::Attribute(a) => a.attr.as_str() == "computed_field",
        _ => false,
    }
}

fn resolve_class_for_stub(class_def: &StmtClassDef, ctx: &ExtractionContext) -> Option<Class> {
    let key = KeyClass(ShortIdentifier::new(&class_def.name));
    let idx = ctx.bindings.key_to_idx_hashed_opt(Hashed::new(&key))?;
    let answered = ctx.answers.get_idx(idx)?;
    answered.0.clone()
}

fn extract_class_body(
    class_def: &StmtClassDef,
    ctx: &mut ExtractionContext,
    resolved_class: Option<&Class>,
) -> Vec<StubItem> {
    let body = &class_def.body;
    let has_explicit_init = body.iter().any(|s| {
        matches!(
            s,
            Stmt::FunctionDef(f) if f.name.as_str() == "__init__"
        )
    });

    if has_explicit_init {
        return extract_stmts(body, ctx, true);
    }

    let first_fn = body.iter().position(|s| matches!(s, Stmt::FunctionDef(_)));
    match first_fn {
        None => {
            let mut items = extract_stmts(body, ctx, true);
            maybe_push_synthetic_init(class_def, &mut items, resolved_class, ctx);
            items
        }
        Some(i) => {
            let leading = &body[..i];
            let tail = &body[i..];
            let mut items = extract_stmts(leading, ctx, true);
            maybe_push_synthetic_init(class_def, &mut items, resolved_class, ctx);
            items.extend(extract_stmts(tail, ctx, true));
            items
        }
    }
}

fn maybe_push_synthetic_init(
    class_def: &StmtClassDef,
    items: &mut Vec<StubItem>,
    resolved_class: Option<&Class>,
    ctx: &mut ExtractionContext,
) {
    if let Some(cls) = resolved_class
        && let Some(init_fn) = synthetic_init_stub_fn(class_def, cls, ctx)
    {
        items.push(StubItem::Function(init_fn));
    }
}

fn synthetic_init_stub_fn(
    class_def: &StmtClassDef,
    cls: &Class,
    ctx: &mut ExtractionContext,
) -> Option<StubFunction> {
    // NamedTuple's synthesized `__init__` is a `(self, *args, **kwargs)` placeholder; the
    // field-accurate signature lives on `__new__`. Skip the placeholder rather than emit it.
    if class_metadata_for(cls, ctx).is_some_and(|m| m.named_tuple_metadata().is_some()) {
        return None;
    }
    let idx = ctx
        .bindings
        .key_to_idx_hashed_opt(Hashed::new(&KeyClassSynthesizedFields(cls.index())))?;
    let synthesized: Arc<ClassSynthesizedFields> = ctx.answers.get_idx(idx)?;
    let init_field = synthesized.get(&dunder::INIT)?;
    let ty = init_field.inner.ty();
    let callable = *ty.callable_signatures().first()?;
    let Params::List(list) = &callable.params else {
        return None;
    };
    let items = list.items();
    let ast_defaults = ann_assign_value_exprs_by_name(&class_def.body);
    let ann_for_init_param = init_param_annotation_overrides_from_class_body(class_def, cls, ctx);
    let mut params = Vec::new();
    let mut inserted_kwonly_star = false;
    let has_explicit_kw_sep = items.iter().any(|p| matches!(p, Param::Varargs(None, _)));

    for p in items {
        if matches!(p, Param::Kwargs(..)) {
            continue;
        }
        if matches!(p, Param::KwOnly(..)) && !inserted_kwonly_star && !has_explicit_kw_sep {
            params.push(StubParam {
                prefix: "",
                name: "*".to_owned(),
                annotation: None,
                default: None,
            });
            inserted_kwonly_star = true;
        }
        let mut stub = synthesized_param_to_stub(p, ctx);
        if let Some(text) = ann_for_init_param.get(stub.name.as_str()) {
            stub.annotation = Some(text.clone());
        }
        if stub.default.as_deref() == Some("...")
            && let Some(expr) = ast_defaults.get(stub.name.as_str())
        {
            let fd = format_default(expr, ctx.module_info);
            if fd != "..." {
                stub.default = Some(fd);
            }
        }
        params.push(stub);
    }

    let return_type = format_type(&callable.ret, ctx);

    Some(StubFunction {
        name: "__init__".to_owned(),
        is_async: false,
        type_params: None,
        decorators: Vec::new(),
        params,
        return_type,
        docstring: None,
    })
}

fn class_metadata_for(cls: &Class, ctx: &ExtractionContext) -> Option<Arc<ClassMetadata>> {
    let idx = ctx
        .bindings
        .key_to_idx_hashed_opt(Hashed::new(&KeyClassMetadata(cls.index())))?;
    ctx.answers.get_idx(idx)
}

/// Map annotated assignment targets to their RHS expression for class-body field stubs.
fn ann_assign_value_exprs_by_name(body: &[Stmt]) -> HashMap<String, &Expr> {
    let mut m = HashMap::new();
    for stmt in body {
        let Stmt::AnnAssign(ann) = stmt else {
            continue;
        };
        let Expr::Name(n) = ann.target.as_ref() else {
            continue;
        };
        if let Some(v) = ann.value.as_deref() {
            m.insert(n.id.to_string(), v);
        }
    }
    m
}

/// Maps synthesized `__init__` parameter names to annotation source text from the class body.
///
/// Pydantic's lax coercion types are wider than the authored annotations; stubs should reflect the
/// latter. Includes [`validation_alias`](https://docs.pydantic.dev/latest/concepts/alias/)
/// keywords when they participate in `__init__`.
fn init_param_annotation_overrides_from_class_body(
    class_def: &StmtClassDef,
    cls: &Class,
    ctx: &ExtractionContext,
) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for stmt in &class_def.body {
        let Stmt::AnnAssign(ann) = stmt else {
            continue;
        };
        let Expr::Name(n) = ann.target.as_ref() else {
            continue;
        };
        let field_name = n.id.clone();
        let Some(idx) = ctx
            .bindings
            .key_to_idx_hashed_opt(Hashed::new(&KeyClassField(cls.index(), field_name.clone())))
        else {
            continue;
        };
        let Some(class_field): Option<Arc<ClassField>> = ctx.answers.get_idx(idx) else {
            continue;
        };
        if class_field.is_init_var() {
            continue;
        }
        let flags = class_field.dataclass_flags_of(ctx.answers.heap());
        if !flags.init {
            continue;
        }
        let ann_text = source_text(ctx.module_info, ann.annotation.range()).to_owned();
        if flags.init_by_name {
            m.insert(field_name.as_str().to_owned(), ann_text.clone());
        }
        if let Some(alias) = &flags.init_by_alias {
            m.insert(alias.as_str().to_owned(), ann_text);
        }
    }
    m
}

fn synthesized_param_to_stub(param: &Param, ctx: &mut ExtractionContext) -> StubParam {
    match param {
        Param::PosOnly(maybe_name, ty, req) => {
            let name = maybe_name
                .as_ref()
                .map(|n| n.as_str().to_owned())
                .unwrap_or_else(|| "_".to_owned());
            StubParam {
                prefix: "",
                name,
                annotation: annotation_for_param(param, ty, ctx),
                default: required_to_stub_default(req, ctx),
            }
        }
        Param::Pos(name, ty, req) => StubParam {
            prefix: "",
            name: name.as_str().to_owned(),
            annotation: annotation_for_param(param, ty, ctx),
            default: required_to_stub_default(req, ctx),
        },
        Param::Varargs(maybe_name, ty) => {
            if let Some(name) = maybe_name {
                StubParam {
                    prefix: "*",
                    name: name.as_str().to_owned(),
                    annotation: format_type(ty, ctx),
                    default: None,
                }
            } else {
                StubParam {
                    prefix: "",
                    name: "*".to_owned(),
                    annotation: None,
                    default: None,
                }
            }
        }
        Param::KwOnly(name, ty, req) => StubParam {
            prefix: "",
            name: name.as_str().to_owned(),
            annotation: annotation_for_param(param, ty, ctx),
            default: required_to_stub_default(req, ctx),
        },
        Param::Kwargs(maybe_name, ty) => StubParam {
            prefix: "**",
            name: maybe_name
                .as_ref()
                .map(|n| n.as_str().to_owned())
                .unwrap_or_else(|| "kwargs".to_owned()),
            annotation: format_type(ty, ctx),
            default: None,
        },
    }
}

fn annotation_for_param(param: &Param, ty: &Type, ctx: &mut ExtractionContext) -> Option<String> {
    if let Some(n) = param.name()
        && (n.as_str() == "self" || n.as_str() == "cls")
    {
        return None;
    }
    format_type(ty, ctx)
}

fn required_to_stub_default(req: &Required, ctx: &mut ExtractionContext) -> Option<String> {
    match req {
        Required::Required => None,
        Required::Optional(None) => Some("...".to_owned()),
        Required::Optional(Some(dv)) => dv.display.clone().or_else(|| format_type(&dv.ty, ctx)),
    }
}

/// Render the class-body default for a field whose RHS is not a simple literal
/// (e.g. `field(default=None)`, `Field(default_factory=list)`). We elide to `= ...`
/// when the dataclass / Pydantic field analysis says the field has a default at all;
/// otherwise no default is emitted. The AST shape of the RHS is irrelevant — the
/// checker's `dataclass_flags_of` already reflects whether `default` / `default_factory`
/// were supplied (and ignores positional `...` per Pydantic's runtime semantics).
fn complex_class_attribute_stub_value(name: &str, ctx: &mut ExtractionContext) -> Option<String> {
    let cls = ctx.current_class.as_ref()?;
    let key = KeyClassField(cls.index(), Name::new(name));
    let idx = ctx.bindings.key_to_idx_hashed_opt(Hashed::new(&key))?;
    let class_field: Arc<ClassField> = ctx.answers.get_idx(idx)?;
    let flags = class_field.dataclass_flags_of(ctx.answers.heap());
    flags.default.as_ref().map(|_| "...".to_owned())
}

fn prune_stub_imports(items: &mut Vec<StubItem>) {
    let body = collect_stub_text_for_import_prune(items);
    prune_imports_in_items(items, &body);
}

fn collect_stub_text_for_import_prune(items: &[StubItem]) -> String {
    let mut out = String::new();
    for item in items {
        append_stub_item_text(item, &mut out);
    }
    out
}

fn append_stub_item_text(item: &StubItem, out: &mut String) {
    match item {
        StubItem::Import(_) => {}
        StubItem::Function(f) => {
            for d in &f.decorators {
                out.push_str(d);
            }
            out.push_str(&f.name);
            for p in &f.params {
                out.push_str(&p.name);
                if let Some(a) = &p.annotation {
                    out.push_str(a);
                }
                if let Some(d) = &p.default {
                    out.push_str(d);
                }
            }
            if let Some(r) = &f.return_type {
                out.push_str(r);
            }
        }
        StubItem::Class(c) => {
            out.push_str(&c.name);
            out.push_str(&c.bases);
            for d in &c.decorators {
                out.push_str(d);
            }
            append_stub_items_text(&c.body, out);
        }
        StubItem::Variable(v) => {
            out.push_str(&v.name);
            if let Some(a) = &v.annotation {
                out.push_str(a);
            }
            if let Some(val) = &v.value {
                out.push_str(val);
            }
        }
        StubItem::TypeAlias(t) => {
            out.push_str(&t.text);
        }
    }
}

fn append_stub_items_text(items: &[StubItem], out: &mut String) {
    for item in items {
        append_stub_item_text(item, out);
    }
}

fn prune_imports_in_items(items: &mut Vec<StubItem>, body: &str) {
    items.retain_mut(|item| {
        if let StubItem::Import(imp) = item {
            let new_text = prune_import_line(&imp.text, body);
            if new_text.is_empty() {
                return false;
            }
            imp.text = new_text;
        }
        if let StubItem::Class(c) = item {
            let class_body = collect_stub_text_for_import_prune(&c.body);
            prune_imports_in_items(&mut c.body, &class_body);
        }
        true
    });
}

fn prune_import_line(line: &str, body: &str) -> String {
    let trimmed = line.trim();
    let indent_len = line.len() - trimmed.len();
    let indent = &line[..indent_len];

    if trimmed == "import functools" && !contains_whole_word(body, "functools") {
        return String::new();
    }
    if let Some(rest) = trimmed.strip_prefix("from dataclasses import") {
        return prune_from_import_rest(indent, "dataclasses", rest, "field", body);
    }
    if let Some(rest) = trimmed.strip_prefix("from pydantic import") {
        let line = prune_from_import_rest(indent, "pydantic", rest, "Field", body);
        return prune_from_import_rest_line(line, indent, "pydantic", "computed_field", body);
    }
    line.to_owned()
}

fn prune_from_import_rest(
    indent: &str,
    module: &str,
    after_import: &str,
    symbol: &str,
    body: &str,
) -> String {
    let keep_symbol = contains_whole_word(body, symbol);
    let names: Vec<String> = after_import
        .split(',')
        .filter_map(|raw| {
            let name = raw.trim();
            if name.is_empty() {
                return None;
            }
            if !keep_symbol && name == symbol {
                return None;
            }
            Some(name.to_owned())
        })
        .collect();
    if names.is_empty() {
        return String::new();
    }
    format!("{}from {} import {}", indent, module, names.join(", "))
}

/// Apply [`prune_from_import_rest`] when `line` is a `from module import ...` line.
fn prune_from_import_rest_line(
    line: String,
    indent: &str,
    module: &str,
    symbol: &str,
    body: &str,
) -> String {
    if line.trim().is_empty() {
        return line;
    }
    let trimmed = line.trim();
    let prefix = format!("from {module} import");
    let Some(rest) = trimmed.strip_prefix(&prefix) else {
        return line;
    };
    prune_from_import_rest(indent, module, rest, symbol, body)
}

fn contains_whole_word(haystack: &str, word: &str) -> bool {
    for (idx, _) in haystack.match_indices(word) {
        let before_ok = idx == 0
            || !haystack
                .as_bytes()
                .get(idx - 1)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
        let after = idx + word.len();
        let after_ok = after >= haystack.len()
            || !haystack
                .as_bytes()
                .get(after)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn source_text(module_info: &Module, range: TextRange) -> &str {
    module_info.code_at(range)
}
