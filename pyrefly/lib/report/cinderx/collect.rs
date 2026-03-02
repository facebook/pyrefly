/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Expression walker for the CinderX type report.
//!
//! Walks every expression in a module's AST, looks up the inferred type,
//! and builds a deduplicated `TypeTable` plus a list of `LocatedType` entries.

use std::collections::HashMap;
use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::facet::FacetKind;
use pyrefly_types::type_info::TypeInfo;
use pyrefly_types::types::BoundMethod;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Type;
use pyrefly_util::visit::Visit;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_python_ast::Stmt;
use ruff_python_ast::statement_visitor::StatementVisitor;
use ruff_python_ast::statement_visitor::walk_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;

use crate::alt::answers::Answers;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClassField;
use crate::binding::bindings::Bindings;
use crate::module::module_info::ModuleInfo;
use crate::report::cinderx::convert::type_to_structured;
use crate::report::cinderx::types::LocatedType;
use crate::report::cinderx::types::TypeTable;
use crate::report::cinderx::types::TypeTableEntry;
use crate::state::state::Transaction;

/// Collected per-module type data for the CinderX report.
pub(crate) struct ModuleTypeData {
    /// Deduplicated type table entries.
    pub entries: Vec<TypeTableEntry>,
    /// Per-expression located type references into the table.
    pub locations: Vec<LocatedType>,
    /// All classes encountered during type conversion, for MRO collection.
    pub classes: Vec<Class>,
}

/// Collect per-expression types for a single module.
///
/// Walks every expression in the AST, looks up the inferred type via
/// `Answers`, converts it to a `StructuredType` entry in a shared
/// `TypeTable`, and records a `LocatedType` mapping source location
/// to table index.
///
/// Returns `None` if any of the required data (AST, answers, bindings,
/// module info) is unavailable for the given handle.
pub(crate) fn collect_module_types(
    transaction: &Transaction,
    handle: &Handle,
) -> Option<ModuleTypeData> {
    let ast = transaction.get_ast(handle)?;
    let module_info = transaction.get_module_info(handle)?;
    let answers = transaction.get_answers(handle)?;
    let bindings = transaction.get_bindings(handle)?;

    let mut contextual_types = build_contextual_types(&ast, &answers, &bindings);
    collect_call_contextual_types(&ast, &answers, &bindings, &mut contextual_types);

    let mut collector = ExpressionCollector {
        module_info: &module_info,
        answers: &answers,
        bindings: &bindings,
        transaction,
        handle,
        contextual_types: &contextual_types,
        table: TypeTable::new(),
        locations: Vec::new(),
        pending_class_traits: Vec::new(),
    };
    collector.walk(&ast);

    // Extract unique classes from pending_class_traits for MRO collection.
    let classes: Vec<Class> = collector
        .pending_class_traits
        .into_iter()
        .map(|(_, cls)| cls)
        .collect();

    Some(ModuleTypeData {
        entries: collector.table.into_entries(),
        locations: collector.locations,
        classes,
    })
}

/// Known CinderX primitive type names from `__static__`.
///
/// These are simple subclasses of `int` or `float` that map to C-level
/// primitives in the CinderX compiler. Other classes exported by the
/// `__static__` module (e.g. `chkdict`, `StaticGeneric`, `Array`) are
/// not primitives and should not receive contextual typing.
const STATIC_PRIMITIVE_NAMES: &[&str] = &[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "cbool", "char",
    "double", "single",
];

/// Check whether a type is a `__static__` primitive (CinderX C-level type).
fn is_static_primitive(ty: &Type) -> bool {
    match ty {
        Type::ClassType(ct) => {
            ct.class_object().module_name().as_str() == "__static__"
                && STATIC_PRIMITIVE_NAMES.contains(&ct.class_object().name().as_str())
        }
        _ => false,
    }
}

/// Look up the declared type of an attribute on a class via `KeyClassField`.
///
/// Given an `ExprAttribute` (e.g. `self.x`), looks up the type of the base
/// expression from the trace, extracts the class, and resolves the attribute
/// via `KeyClassField`. Returns `None` if any step fails (e.g. the base is
/// not a class instance, or the attribute is not a known class field in this
/// module's bindings).
fn lookup_attr_type(attr: &ExprAttribute, answers: &Answers, bindings: &Bindings) -> Option<Type> {
    let base_ty = answers.get_type_trace(attr.value.range())?;
    let class_def_index = match &base_ty {
        Type::ClassType(ct) | Type::SelfType(ct) => ct.class_object().index(),
        _ => return None,
    };
    let key = KeyClassField(class_def_index, attr.attr.id.clone());
    let idx = bindings.key_to_idx_hashed_opt(Hashed::new(&key))?;
    let class_field = answers.get_idx(idx)?;
    Some(class_field.ty())
}

/// Statement visitor that builds a map from RHS expression ranges to their
/// contextual (annotation) types for `AnnAssign` and `Assign` statements
/// targeting `__static__` primitive types.
struct ContextualTypeCollector<'a> {
    answers: &'a Answers,
    bindings: &'a Bindings,
    contextual_types: HashMap<TextRange, Type>,
}

impl<'a> StatementVisitor<'a> for ContextualTypeCollector<'a> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        if let Stmt::AnnAssign(ann) = stmt
            && let Some(ref value) = ann.value
        {
            let target_type = match ann.target.as_ref() {
                Expr::Name(name) => {
                    let key = Key::Definition(ShortIdentifier::expr_name(name));
                    if self.bindings.is_valid_key(&key) {
                        self.answers.get_type_at(self.bindings.key_to_idx(&key))
                    } else {
                        None
                    }
                }
                Expr::Attribute(attr) => lookup_attr_type(attr, self.answers, self.bindings),
                _ => None,
            };
            if let Some(ty) = target_type
                && is_static_primitive(&ty)
            {
                self.contextual_types.insert(value.range(), ty);
            }
        }
        if let Stmt::Assign(assign) = stmt {
            for target in &assign.targets {
                let target_type = match target {
                    Expr::Name(name) => {
                        let key = Key::BoundName(ShortIdentifier::expr_name(name));
                        let valid_key = if self.bindings.is_valid_key(&key) {
                            Some(key)
                        } else {
                            let key = Key::Definition(ShortIdentifier::expr_name(name));
                            if self.bindings.is_valid_key(&key) {
                                Some(key)
                            } else {
                                None
                            }
                        };
                        valid_key.and_then(|key| {
                            self.answers.get_type_at(self.bindings.key_to_idx(&key))
                        })
                    }
                    Expr::Attribute(attr) => lookup_attr_type(attr, self.answers, self.bindings),
                    _ => None,
                };
                if let Some(ty) = target_type
                    && is_static_primitive(&ty)
                {
                    self.contextual_types.insert(assign.value.range(), ty);
                }
            }
        }
        walk_stmt(self, stmt);
    }
}

/// Build a map from RHS expression ranges to contextual types for qualifying
/// `AnnAssign` and `Assign` statements (those targeting `__static__` primitive types).
fn build_contextual_types(
    ast: &Arc<ModModule>,
    answers: &Answers,
    bindings: &Bindings,
) -> HashMap<TextRange, Type> {
    let mut collector = ContextualTypeCollector {
        answers,
        bindings,
        contextual_types: HashMap::new(),
    };
    collector.visit_body(&ast.body);
    collector.contextual_types
}

/// Walk all expressions looking for `Expr::Call` nodes and collect contextual
/// types for positional arguments whose corresponding parameter is a `__static__`
/// primitive type.
///
/// Skips calls where any argument is a starred expression (`*args`) or any
/// keyword is a splat (`**kwargs`), since argument-to-parameter matching is
/// unreliable in those cases.
fn collect_call_contextual_types(
    ast: &Arc<ModModule>,
    answers: &Answers,
    bindings: &Bindings,
    contextual_types: &mut HashMap<TextRange, Type>,
) {
    ast.visit(&mut |x: &Expr| {
        if let Expr::Call(call) = x {
            // Skip if any arg is starred (*args).
            if call
                .arguments
                .args
                .iter()
                .any(|a| matches!(a, Expr::Starred(_)))
            {
                return;
            }
            // Skip if any keyword is a splat (**kwargs).
            if call.arguments.keywords.iter().any(|kw| kw.arg.is_none()) {
                return;
            }

            // Look up the callee type.
            let callee_type = if let Expr::Name(name) = call.func.as_ref() {
                let key = Key::BoundName(ShortIdentifier::expr_name(name));
                if bindings.is_valid_key(&key) {
                    answers.get_type_at(bindings.key_to_idx(&key))
                } else {
                    let key = Key::Definition(ShortIdentifier::expr_name(name));
                    if bindings.is_valid_key(&key) {
                        answers.get_type_at(bindings.key_to_idx(&key))
                    } else {
                        None
                    }
                }
            } else {
                answers.get_type_trace(call.func.range())
            };

            let Some(callee_type) = callee_type else {
                return;
            };

            // Extract parameters and the number of leading params to skip
            // (1 for bound methods to skip `self`, 0 otherwise).
            let (params, skip_count): (&Params, usize) = match &callee_type {
                Type::Function(f) => (&f.signature.params, 0),
                Type::BoundMethod(box BoundMethod {
                    func: BoundMethodType::Function(f),
                    ..
                }) => (&f.signature.params, 1),
                _ => return,
            };

            let Params::List(param_list) = params else {
                return;
            };
            let items = param_list.items();

            for (i, arg) in call.arguments.args.iter().enumerate() {
                let param_idx = i + skip_count;
                if param_idx >= items.len() {
                    break;
                }
                let param = &items[param_idx];

                // Break if we hit a variadic argument - matching past that is complicated.
                if matches!(param, Param::VarArg(_, _) | Param::Kwargs(_, _)) {
                    break;
                }
                if is_static_primitive(param.as_type()) {
                    contextual_types.insert(arg.range(), param.as_type().clone());
                }
            }

            // Process keyword arguments by matching keyword name to parameter name.
            //
            // For valid code, this should not be able to overlap with the positional arguments
            // (and Pyrefly would raise a type error) although we don't try to handle that here.
            for kw in call.arguments.keywords.iter() {
                // kw.arg is Option<Identifier> — None means **kwargs splat,
                // which we already skip above.
                if let Some(ref kw_name) = kw.arg {
                    // Find the parameter with this name.
                    for param in items {
                        if let Some(param_name) = param.name()
                            && param_name.as_str() == kw_name.as_str()
                            && is_static_primitive(param.as_type())
                        {
                            contextual_types.insert(kw.value.range(), param.as_type().clone());
                            break;
                        }
                    }
                }
            }
        }
    });
}

/// Walk an expression chain of `Expr::Attribute` and `Expr::Subscript`
/// nodes to find the root `Expr::Name` and collect `FacetKind`s from
/// root to leaf.
///
/// Handles all three facet kinds:
/// - `Attribute(name)` from `x.foo`
/// - `Index(n)` from `x[0]` (integer literal subscript)
/// - `Key(s)` from `x["key"]` (string literal subscript)
///
/// Returns `None` if the chain doesn't root at an `Expr::Name`
/// (e.g. `f().attr`) or contains a non-literal subscript.
fn extract_facet_chain(expr: &Expr) -> Option<(&ExprName, Vec<FacetKind>)> {
    if !matches!(expr, Expr::Attribute(_) | Expr::Subscript(_)) {
        return None;
    }
    let mut chain = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attr) => {
                chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                current = attr.value.as_ref();
            }
            Expr::Subscript(sub) => {
                match sub.slice.as_ref() {
                    Expr::NumberLiteral(ExprNumberLiteral {
                        value: Number::Int(idx),
                        ..
                    }) if idx.as_i64().is_some() => {
                        chain.push(FacetKind::Index(idx.as_i64().unwrap()));
                    }
                    Expr::StringLiteral(lit) => {
                        chain.push(FacetKind::Key(lit.value.to_string()));
                    }
                    _ => return None,
                }
                current = sub.value.as_ref();
            }
            Expr::Name(name) => {
                chain.reverse();
                return Some((name, chain));
            }
            _ => return None,
        }
    }
}

/// Check whether any level in a facet chain has a facet narrow.
///
/// Walks the facet tree level by level using `type_at_facet` to check
/// for narrows and `at_facet` to descend. Returns `true` as soon as
/// any level has a narrowed type.
fn has_facet_narrow_in_chain(type_info: &TypeInfo, chain: &[FacetKind]) -> bool {
    let mut current = type_info.clone();
    for facet in chain {
        if current.type_at_facet(facet).is_some() {
            return true;
        }
        current = current.at_facet(facet, Type::never);
    }
    false
}

/// Holds all the context needed to walk expressions and collect types
/// for the CinderX report.
///
/// Groups the read-only per-module data (answers, bindings, module info,
/// transaction context) together with the mutable output accumulators
/// (type table, located types, pending classes) to avoid threading many
/// arguments through every recursive call.
struct ExpressionCollector<'a> {
    module_info: &'a ModuleInfo,
    answers: &'a Answers,
    bindings: &'a Bindings,
    transaction: &'a Transaction<'a>,
    handle: &'a Handle,
    contextual_types: &'a HashMap<TextRange, Type>,
    table: TypeTable,
    locations: Vec<LocatedType>,
    pending_class_traits: Vec<(usize, Class)>,
}

impl<'a> ExpressionCollector<'a> {
    /// Try to find a binding key for a name expression.
    ///
    /// Checks `BoundName` (use/load site) first, then `Definition`
    /// (store site), returning `None` if neither exists in bindings.
    fn try_find_key_for_name(&self, name: &ExprName) -> Option<Key> {
        let key = Key::BoundName(ShortIdentifier::expr_name(name));
        if self.bindings.is_valid_key(&key) {
            return Some(key);
        }
        let key = Key::Definition(ShortIdentifier::expr_name(name));
        if self.bindings.is_valid_key(&key) {
            return Some(key);
        }
        None
    }

    /// Look up the inferred type for an expression.
    ///
    /// For `Expr::Name` nodes, looks up via the binding key for precise
    /// results. For all other expressions, falls back to the trace map
    /// which is populated during solving.
    fn lookup_type(&self, x: &Expr) -> Option<Type> {
        if let Expr::Name(name) = x
            && let Some(key) = self.try_find_key_for_name(name)
            && let Some(ty) = self.answers.get_type_at(self.bindings.key_to_idx(&key))
        {
            return Some(ty);
        }
        self.answers.get_type_trace(x.range())
    }

    /// Recursive expression visitor: looks up type, converts to structured
    /// form, records location, then recurses into child expressions.
    ///
    /// For attribute/subscript chains rooted at a name (`x.attr`, `x[0]`,
    /// `x["key"].bar`, etc.), checks whether any level in the chain has a
    /// facet narrow. If so, re-resolves the full chain on the unnarrowed
    /// base type to populate `unnarrowed_type` and `is_narrowed_mismatch`
    /// on the `LocatedType`.
    fn visit_expr(&mut self, x: &Expr, parent: Option<&Expr>) {
        if let Some(ty) = self.lookup_type(x) {
            let range = x.range();
            let location = self
                .module_info
                .lined_buffer()
                .python_ast_range_for_expr(range, x, parent);
            let type_index =
                type_to_structured(&ty, &mut self.table, &mut self.pending_class_traits);

            // Detect facet narrows on attribute/subscript chains (x.attr, x[0], etc.).
            // When any level in the chain has a facet narrow, re-resolve the full
            // chain on the unnarrowed base type so CinderX can handle the unsound
            // narrow appropriately.
            let (unnarrowed_type, is_narrowed_mismatch) = if let Some((name, chain)) =
                extract_facet_chain(x)
                && let Some(key) = self.try_find_key_for_name(name)
                && let Some(type_info) = self.answers.get_idx(self.bindings.key_to_idx(&key))
                && type_info.has_facets()
                && has_facet_narrow_in_chain(&type_info, &chain)
            {
                // Some level in the chain has a facet narrow.
                // Re-resolve the full chain on the unnarrowed base type.
                let base_type = type_info.ty().clone();
                let unnarrowed_ty =
                    self.transaction
                        .ad_hoc_solve(self.handle, "cinderx_unnarrow", |solver| {
                            let errors = solver.error_swallower();
                            let mut current_ty = base_type;
                            for facet in &chain {
                                current_ty = match facet {
                                    FacetKind::Attribute(name) => solver.attr_infer_for_type(
                                        &current_ty,
                                        name,
                                        range,
                                        &errors,
                                        None,
                                    ),
                                    FacetKind::Index(idx) => {
                                        let synth = Expr::NumberLiteral(ExprNumberLiteral {
                                            node_index: AtomicNodeIndex::default(),
                                            range: TextRange::empty(TextSize::from(0)),
                                            value: Number::Int(Int::from(*idx as u64)),
                                        });
                                        solver.subscript_infer_for_type(
                                            &current_ty,
                                            &synth,
                                            range,
                                            &errors,
                                        )
                                    }
                                    FacetKind::Key(key) => {
                                        let synth =
                                            Ast::str_expr(key, TextRange::empty(TextSize::from(0)));
                                        solver.subscript_infer_for_type(
                                            &current_ty,
                                            &synth,
                                            range,
                                            &errors,
                                        )
                                    }
                                };
                            }
                            current_ty
                        });
                match unnarrowed_ty {
                    Some(unnarrowed_ty) => {
                        let unnarrowed_idx = type_to_structured(
                            &unnarrowed_ty,
                            &mut self.table,
                            &mut self.pending_class_traits,
                        );
                        let is_mismatch =
                            self.table.hash_at(type_index) != self.table.hash_at(unnarrowed_idx);
                        (Some(unnarrowed_idx), is_mismatch)
                    }
                    // ad_hoc_solve returned None (module data unavailable); degrade gracefully.
                    None => (None, false),
                }
            } else {
                (None, false)
            };

            // Check if this expression flows into a slot with a contextual type
            // (e.g. a literal assigned to a `__static__` primitive variable).
            let contextual_type = self.contextual_types.get(&x.range()).map(|ctx_ty| {
                type_to_structured(ctx_ty, &mut self.table, &mut self.pending_class_traits)
            });

            self.locations.push(LocatedType {
                location,
                type_index,
                unnarrowed_type,
                is_narrowed_mismatch,
                contextual_type,
            });
        }

        x.recurse(&mut |child| self.visit_expr(child, Some(x)));
    }

    /// Walk all top-level expressions in the AST.
    fn walk(&mut self, ast: &Arc<ModModule>) {
        ast.visit(&mut |x: &Expr| self.visit_expr(x, None));
    }
}
