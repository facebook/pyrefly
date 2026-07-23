/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_graph::index::Idx;
use pyrefly_python::dunder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType;
use pyrefly_types::types::Type;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::includes::Includes;
use pyrefly_util::lock::Mutex;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_python_ast::Expr;
use ruff_python_ast::Parameters;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::Hashed;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::Answers;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingExport;
use crate::binding::binding::BindingUndecoratedFunction;
use crate::binding::binding::ClassBinding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::Key;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassMetadata;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyDecorator;
use crate::binding::binding::KeyExport;
use crate::binding::binding::KeyUndecoratedFunction;
use crate::binding::binding::ReturnTypeKind;
use crate::binding::bindings::Bindings;
use crate::commands::check::Handles;
use crate::commands::coverage::types::*;
use crate::error::error::Error;
use crate::export::exports::ExportLocation;
use crate::export::exports::Exports;
use crate::module::finder::DirEntryCache;
use crate::module::finder::find_import_filtered;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;

/// All parameters with merge keys: positional-only → index, keyword-capable → name,
/// variadic → singleton, implicit receiver → `None`.
fn params_with_keys(
    params: &Parameters,
    has_implicit_receiver: bool,
) -> Vec<(Option<ParamKey>, &ruff_python_ast::Parameter)> {
    let mut result = Vec::new();
    for (i, p) in params.posonlyargs.iter().enumerate() {
        result.push((Some(ParamKey::Positional(i + 1)), &p.parameter));
    }
    for p in params.args.iter().chain(&params.kwonlyargs) {
        result.push((
            Some(ParamKey::Named(p.parameter.name.to_string())),
            &p.parameter,
        ));
    }
    if let Some(v) = &params.vararg {
        result.push((Some(ParamKey::VarPositional), v));
    }
    if let Some(v) = &params.kwarg {
        result.push((Some(ParamKey::VarKeyword), v));
    }
    if has_implicit_receiver && let Some((key, _)) = result.first_mut() {
        *key = None;
    }
    result
}

fn range_to_location(module: &Module, range: TextRange) -> Location {
    let pos = module.lined_buffer().display_pos(range.start(), None);
    Location {
        line: pos.line_within_file().get() as usize,
        column: pos.column().get() as usize,
    }
}

/// Collect the module's type-ignore suppressions, each located at the `#` starting its comment.
fn parse_suppressions(module: &Module) -> Vec<ReportSuppression> {
    let mut suppressions = Vec::new();
    for (_, supps) in module.ignore().iter() {
        for supp in supps {
            let offset = module.lined_buffer().line_start(supp.comment_line())
                + TextSize::try_from(supp.comment_offset()).unwrap();
            suppressions.push(ReportSuppression {
                kind: supp.tool(),
                codes: supp.error_codes().to_vec(),
                location: range_to_location(module, TextRange::empty(offset)),
            });
        }
    }
    suppressions
}

fn has_function_ancestor(parent: &NestingContext) -> bool {
    let mut current = parent;
    loop {
        if current.is_function() {
            return true;
        }
        match current.parent() {
            Some(p) => current = p,
            None => return false,
        }
    }
}

/// True if the class, or an enclosing class, was removed by a module-scope `del`.
/// After `del X` at module scope the entire `X.*` subtree is unreachable, so a nested
/// class (and its methods/attrs) must be excluded along with `X` itself. We therefore
/// test the outermost enclosing name: function-nested classes are already filtered by
/// `has_function_ancestor`, so every context reached here is a class.
fn is_deleted_class(module: &Module, bindings: &Bindings, cls: &ClassBinding) -> bool {
    let mut nesting = &cls.parent;
    let outermost = loop {
        match nesting.parent() {
            None => break cls.def.name.id.as_str(),
            Some(parent) if parent.is_toplevel() => {
                let short_id = nesting
                    .identifier()
                    .expect("non-toplevel NestingContext must have an identifier");
                break module.code_at(short_id.range());
            }
            Some(parent) => nesting = parent,
        }
    };
    bindings
        .module_deletes()
        .iter()
        .any(|n| n.as_str() == outermost)
}

/// Build a class's fully-qualified name: module prefix, nesting context, then class name. Returns
/// e.g. `"pkg.mod.Outer.Inner"` for a nested class or `"pkg.mod.MyClass"` for a top-level one.
fn class_fqn(
    module: &Module,
    parent: &NestingContext,
    class_name: impl std::fmt::Display,
) -> String {
    let prefix = module_prefix(module);
    let parent_path = module.display(parent).to_string();
    if parent_path.is_empty() {
        format!("{prefix}{class_name}")
    } else {
        format!("{prefix}{parent_path}.{class_name}")
    }
}

/// The module name with a trailing `.`, prefixed to symbol FQNs.
fn module_prefix(module: &Module) -> String {
    format!("{}.", module.name())
}

/// Merge overloads with the same qualified name into one entry,
/// keeping the best annotation quality per deduplicated slot.
fn merge_overloads(functions: &mut Vec<Function>) {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, func) in functions.iter().enumerate() {
        if func.property_role.is_none() {
            groups.entry(func.name.clone()).or_default().push(i);
        }
    }

    let mut to_remove: SmallSet<usize> = SmallSet::new();
    for indices in groups.into_values().filter(|g| g.len() >= 2) {
        let mut param_slots: HashMap<ParamKey, SlotRank> = HashMap::new();
        let mut return_rank = SlotRank::Skip;

        for &idx in &indices {
            let func = &functions[idx];
            return_rank = return_rank.max(func.return_rank);

            for param in &func.parameters {
                if let Some(key) = &param.merge_key {
                    let entry = param_slots.entry(key.clone()).or_insert(SlotRank::Skip);
                    *entry = (*entry).max(param.into());
                }
            }
        }

        // Fold deduplicated slots (Skip contributes zero counts).
        let slots = std::iter::once(return_rank)
            .chain(param_slots.values().copied())
            .map(SlotCounts::from)
            .fold(SlotCounts::default(), SlotCounts::merge);
        let n_params = param_slots
            .into_values()
            .filter(|r| *r != SlotRank::Skip)
            .count();

        functions[indices[0]].slots = slots;
        functions[indices[0]].n_params = n_params;
        functions[indices[0]].is_type_known = slots.n_untyped == 0 && slots.n_any == 0;
        to_remove.extend(indices[1..].iter().copied());
    }

    if !to_remove.is_empty() {
        let mut i = 0;
        functions.retain(|_| {
            let keep = !to_remove.contains(&i);
            i += 1;
            keep
        });
    }
}

/// Classify an annotation slot from resolver output. Bare qualifiers
/// (e.g. `Final`) have unresolved annotation types but still count as typed.
fn classify_annotation_rank(has_annotation: bool, resolved_is_known: Option<bool>) -> SlotRank {
    match resolved_is_known {
        Some(is_known) => SlotRank::classify(true, is_known),
        None if has_annotation => SlotRank::Typed,
        None => SlotRank::Untyped,
    }
}

/// Annotation source text and slot classification for an optional annotation binding.
fn classify_annotation(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    annotation_idx: Option<Idx<KeyAnnotation>>,
) -> (Option<String>, SlotCounts) {
    let annotation_text = annotation_idx.and_then(|idx| match bindings.get(idx) {
        BindingAnnotation::AnnotateExpr(_, expr, _) => {
            Some(module.code_at(expr.range()).to_owned())
        }
        _ => None,
    });
    let resolved_ty = annotation_idx.and_then(|idx| {
        answers
            .get_idx(idx)
            .and_then(|awt| awt.annotation.ty.as_ref().map(is_type_known))
    });
    let slots = classify_annotation_rank(annotation_text.is_some(), resolved_ty).into();
    (annotation_text, slots)
}

/// Returns true if the name is public: does not start with `_`, or is a dunder (`__x__`).
/// Matches typestats `is_public_name`.
fn is_public_name(name: &str) -> bool {
    !name.starts_with('_') || name.ends_with("__")
}

/// A module is public when every dotted-path component is public.
fn is_public_module(module: ModuleName) -> bool {
    module.as_str().split('.').all(is_public_name)
}

/// True if `fqn` is in `public_fqns` directly, or a class-level prefix is and every segment
/// below the module is itself a public name (no private nested leaks).
fn is_public_fqn(fqn: &str, module_prefix: &str, public_fqns: &HashSet<String>) -> bool {
    if let Some(relative) = fqn.strip_prefix(module_prefix)
        && !relative.split('.').all(is_public_name)
        && !public_fqns.contains(fqn)
    {
        return false;
    }

    public_fqns.contains(fqn)
        || std::iter::successors(fqn.rsplit_once('.').map(|(prefix, _)| prefix), |prefix| {
            prefix.rsplit_once('.').map(|(parent, _)| parent)
        })
        .take_while(|prefix| prefix.starts_with(module_prefix))
        .any(|prefix| public_fqns.contains(prefix))
}

/// Module-level dunders that typestats always excludes from the report.
const EXCLUDED_MODULE_DUNDERS: &[&str] = &[
    // User-level module hooks.
    "__all__",
    "__dir__",
    "__doc__",
    "__getattr__",
    // CPython-injected globals.
    // Keep in sync with `IMPLICIT_GLOBALS` in `crates/pyrefly_types/src/globals.rs`.
    "__annotations__",
    "__builtins__",
    "__cached__",
    "__debug__",
    "__dict__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__path__",
    "__spec__",
];

/// Walk re-exports to the defining module's handle and name, `None` on cycle/miss.
fn trace_export_origin(
    handle: &Handle,
    mut cur_name: Name,
    transaction: &Transaction,
) -> Option<(Handle, Name)> {
    let mut seen = SmallSet::new();
    let mut cur_handle = handle.clone();

    loop {
        if !seen.insert((cur_handle.module(), cur_name.clone())) {
            return None;
        }

        match transaction.get_exports(&cur_handle).get(&cur_name) {
            Some(ExportLocation::ThisModule(_)) => {
                return Some((cur_handle, cur_name));
            }
            None => return None,
            Some(ExportLocation::OtherModule(other_module, alias)) => {
                if let Some(alias) = alias {
                    cur_name = alias.clone();
                }
                cur_handle = transaction
                    .import_handle(&cur_handle, *other_module, None)
                    .finding()?;
            }
        }
    }
}

/// Collect origin FQNs of all publicly exported names across public modules, plus the deduped
/// handles of cross-module re-export origins.
fn compute_public_fqns(
    handles: &[Handle],
    transaction: &Transaction,
) -> (HashSet<String>, SmallSet<Handle>) {
    let mut fqns = HashSet::new();
    let mut origins = SmallSet::new();
    for handle in handles.iter().filter(|h| is_public_module(h.module())) {
        let exports_data = transaction.get_exports_data(handle);
        let exports = transaction.get_exports(handle);

        // prioritize `__all__` if present, otherwise local defs + `import x as x`
        let names: Vec<Name> =
            if let Some(all_iter) = exports_data.get_explicit_dunder_all_names_iter() {
                all_iter.cloned().collect()
            } else {
                exports
                    .iter()
                    .filter_map(|(name, loc)| {
                        let is_local = matches!(loc, ExportLocation::ThisModule(_));
                        let is_reexport = exports_data.is_explicit_reexport(name);
                        (is_public_name(name.as_str()) && (is_local || is_reexport))
                            .then_some(name.clone())
                    })
                    .collect()
            };

        // collect both the local and traced origin FQN so a file-scoped run matches the module
        for name in names {
            if EXCLUDED_MODULE_DUNDERS.contains(&name.as_str()) {
                continue;
            }
            fqns.insert(format!("{}.{}", handle.module(), name));
            if let Some((origin_handle, origin_name)) =
                trace_export_origin(handle, name, transaction)
            {
                fqns.insert(format!("{}.{}", origin_handle.module(), origin_name));
                if origin_handle != *handle {
                    origins.insert(origin_handle);
                }
            }
        }
    }
    (fqns, origins)
}

/// Retain only publicly reachable symbols and recalculate aggregates.
fn filter_module_report_to_public(report: &mut ModuleReport, public_fqns: &HashSet<String>) {
    let module_prefix = format!("{}.", report.name);

    report
        .symbol_reports
        .retain(|sym| is_public_fqn(sym.name(), &module_prefix, public_fqns));
    report
        .names
        .retain(|n| is_public_fqn(n, &module_prefix, public_fqns));

    // type ignores attach to the module, not symbols, so filtering leaves them untouched
    report.slots = report
        .symbol_reports
        .iter()
        .fold(SlotCounts::default(), |acc, sym| acc.merge(*sym.slots()));
    report.symbols = SymbolCounts {
        n_type_ignores: report.symbols.n_type_ignores,
        ..count_symbols(&report.symbol_reports, &module_prefix)
    };

    report.coverage = report.slots.coverage();
    report.strict_coverage = report.slots.strict_coverage();
}

/// True if the first parameter is the implicit receiver (`self`/`cls`),
/// excluded from slot counting. `__new__` is a staticmethod but still takes cls.
fn has_implicit_receiver(
    fun: &BindingUndecoratedFunction,
    answers: &Answers,
    undecorated_idx: Idx<KeyUndecoratedFunction>,
) -> bool {
    fun.class_key.is_some() && {
        let is_staticmethod = answers
            .get_idx(undecorated_idx)
            .is_some_and(|u| u.metadata.flags.is_staticmethod);
        !is_staticmethod || fun.def.name.as_str() == dunder::NEW
    }
}

/// Whether the class is a schema class (dataclass, enum,
/// TypedDict, NamedTuple, pydantic model, or attrs class). Fields of schema classes are
/// IMPLICIT — they have 0 typable slots because the class definition itself governs their types.
///
/// Only frameworks where annotations are structurally required are included.
/// Django, marshmallow, and factory_boy use descriptors with optional annotations,
/// so their fields count toward coverage.
fn is_schema_class(bindings: &Bindings, answers: &Answers, cls_binding: &ClassBinding) -> bool {
    let metadata_key = KeyClassMetadata(cls_binding.def_index);
    answers
        .get_idx(bindings.key_to_idx(&metadata_key))
        .is_some_and(|metadata| {
            metadata.dataclass_metadata().is_some()
                || metadata.is_enum()
                || metadata.is_typed_dict()
                || metadata.named_tuple_metadata().is_some()
                || metadata.is_pydantic_model()
                || metadata.is_attrs_class()
        })
}

fn parse_variables(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    exports: &SmallMap<Name, ExportLocation>,
    dunder_all: &SmallSet<Name>,
    functions: &[Function],
    classes: &[ReportClass],
) -> Vec<Variable> {
    fn untyped_if_call(expr: &Expr) -> SlotCounts {
        if let Expr::Call(_) = expr {
            SlotCounts::untyped()
        } else {
            SlotCounts::default()
        }
    }

    /// True if the binding resolves to an import in at least one flow branch.
    fn involves_import(bindings: &Bindings, idx: Idx<Key>, seen: &mut SmallSet<Idx<Key>>) -> bool {
        if !seen.insert(idx) {
            return false; // LoopPhi can be cyclic
        }
        match bindings.get(idx) {
            Binding::Module(..) | Binding::Import(..) => true,
            Binding::Forward(i)
            | Binding::PromoteForward(i)
            | Binding::ForwardToFirstUse(i)
            | Binding::Narrow(i, _, _) => involves_import(bindings, *i, seen),
            Binding::Phi(_, branches) => branches
                .iter()
                .any(|b| involves_import(bindings, b.value_key, seen)),
            Binding::LoopPhi(prior, members) => {
                involves_import(bindings, *prior, seen)
                    || members.iter().any(|i| involves_import(bindings, *i, seen))
            }
            _ => false,
        }
    }

    let module_prefix = module_prefix(module);
    let deleted = bindings.module_deletes();
    // Collect names already reported as functions or classes so we can skip them.
    let reported_names: SmallSet<&str> = functions
        .iter()
        .map(|f| f.name.as_str())
        .chain(classes.iter().map(|c| c.name.as_str()))
        .collect();

    let mut variables = Vec::new();
    for idx in bindings.keys::<KeyExport>() {
        let KeyExport(name) = bindings.idx_to_key(idx);
        // Skip non-public module-level names (unless listed in `__all__`), excluded dunders, and `del`eted names.
        let name_str = name.as_str();
        if (!is_public_name(name_str) && !dunder_all.contains(name))
            || EXCLUDED_MODULE_DUNDERS.contains(&name_str)
            || deleted.contains(name)
        {
            continue;
        }
        let qualified_name = format!("{module_prefix}{name}");
        if reported_names.contains(qualified_name.as_str()) {
            continue;
        }
        let binding = bindings.get(idx);
        let range = match exports.get(name) {
            Some(ExportLocation::ThisModule(export)) => export.location,
            _ => continue,
        };
        let (annotation, slots) = match binding {
            BindingExport::AnnotatedForward(annot_idx, key_idx) => {
                // IMPLICIT: type aliases are type-level constructs with 0 slots.
                if matches!(
                    bindings.get(*key_idx),
                    Binding::TypeAlias(_) | Binding::TypeAliasRef(_)
                ) {
                    (None, SlotCounts::default())
                } else {
                    classify_annotation(module, bindings, answers, Some(*annot_idx))
                }
            }
            BindingExport::Forward(idx) | BindingExport::PromoteForward(idx) => {
                match bindings.get(*idx) {
                    // Skip injected implicit globals
                    Binding::Global(_) => continue,
                    // IMPLICIT: special type forms and type aliases have 0 slots
                    Binding::TypeVar(_)
                    | Binding::ParamSpec(_)
                    | Binding::TypeVarTuple(_)
                    | Binding::TypeAlias(_)
                    | Binding::TypeAliasRef(_) => (None, SlotCounts::default()),
                    // Functions and classes are handled by parse_functions/parse_classes;
                    // skip them here even when excluded (e.g. @type_check_only).
                    Binding::Function(..) | Binding::ClassDef(..) => continue,
                    // IMPLICIT: non-call assignments have 0 slots;
                    // call assignments are untyped (1 slot)
                    Binding::NameAssign(na) => (None, untyped_if_call(na.expr.as_ref())),
                    Binding::MultiTargetAssign(_, rhs_idx, _, _) => match bindings.get(*rhs_idx) {
                        Binding::Function(..) | Binding::ClassDef(..) => continue,
                        Binding::Expr(_, expr) => (None, untyped_if_call(expr.as_ref())),
                        _ => {
                            unreachable!(
                                "MultiTargetAssign RHS should be Expr, Function, or ClassDef"
                            );
                        }
                    },
                    // Skip optional imports (`try: import x; except _: x = None`) like plain imports.
                    Binding::Phi(..) | Binding::LoopPhi(..)
                        if involves_import(bindings, *idx, &mut SmallSet::new()) =>
                    {
                        continue;
                    }
                    _ => (None, SlotCounts::untyped()),
                }
            }
        };
        variables.push(Variable {
            name: qualified_name,
            annotation,
            slots,
            location: range_to_location(module, range),
            range,
        });
    }
    variables.sort_by_key(|a| a.location);
    variables
}

/// The MRO of `class`, or `Cyclic` if unresolved.
fn class_mro(bindings: &Bindings, answers: &Answers, class: &Class) -> Arc<ClassMro> {
    answers
        .get_idx(bindings.key_to_idx(&KeyClassMro(class.index())))
        .unwrap_or_else(|| Arc::new(ClassMro::Cyclic))
}

/// Slots for `field_name` from the nearest base class annotating it in `class_idx`'s MRO (gh-3997).
///
/// Walks the MRO nearest-first and returns the first ancestor that *annotates* `field_name`, so the
/// closest base's type quality wins. Ancestors that merely assign the field without an annotation
/// contribute nothing (an unannotated base can't upgrade the subclass attr); returns `None` when no
/// base annotates it.
fn inherited_annotation_slots(
    bindings: &Bindings,
    answers: &Answers,
    transaction: &Transaction,
    handle: &Handle,
    class_idx: Idx<KeyClass>,
    field_name: &Name,
) -> Option<SlotCounts> {
    let class = answers.get_idx(class_idx).and_then(|r| r.0.clone())?;
    class_mro(bindings, answers, &class)
        .ancestors_no_object()
        .iter()
        .find_map(|ancestor| {
            let cls = ancestor.class_object();
            let h = Handle::new(
                cls.module_name(),
                cls.module_path().dupe(),
                handle.sys_info().dupe(),
            );
            let b = transaction.get_bindings(&h)?;
            let idx = b.key_to_idx_hashed_opt(Hashed::new(&KeyClassField(
                cls.index(),
                field_name.clone(),
            )))?;
            let annot = match &b.get(idx).definition {
                ClassFieldDefinition::DeclaredByAnnotation { annotation, .. } => Some(*annotation),
                ClassFieldDefinition::DefinedInMethod { annotation, .. }
                | ClassFieldDefinition::AssignedInBody { annotation, .. } => *annotation,
                _ => None,
            }?;
            let m = transaction.get_module_info(&h)?;
            let a = transaction.get_answers(&h)?;
            Some(classify_annotation(&m, &b, &a, Some(annot)).1)
        })
}

/// Extract instance attributes assigned in `__init__`/`__new__`/`__post_init__`,
/// plus schema class body fields (dataclass, enum, TypedDict, NamedTuple).
///
/// For each class field that is either:
/// - `DefinedInMethod` from a recognized attribute-defining method (e.g. `__init__`), or
/// - `DeclaredByAnnotation` in the class body AND initialized in such a method,
///
/// emit a `Variable` (reported as `SymbolReport::Attr`).
///
/// For schema class body fields (dataclass fields, enum members, TypedDict/NamedTuple
/// fields), emit a `Variable` with `SlotCounts::default()` (0 typable) to match
/// typestats IMPLICIT classification.
fn parse_instance_attrs(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    transaction: &Transaction,
    handle: &Handle,
    tco_classes: &SmallSet<Idx<KeyClass>>,
) -> Vec<Variable> {
    let mut attrs = Vec::new();

    for field_idx in bindings.keys::<KeyClassField>() {
        let field = bindings.get(field_idx);

        // Skip private class attrs (single-underscore prefix).
        if !is_public_name(field.name.as_str()) {
            continue;
        }

        // Skip class-body dunder attrs with implicit types (__slots__, __doc__, etc.)
        if is_implicit_dunder_attr(field.name.as_str()) {
            continue;
        }

        // Skip attrs of @type_check_only classes.
        if tco_classes.contains(&field.class_idx) {
            continue;
        }

        let cls_binding = match bindings.get(field.class_idx) {
            BindingClass::ClassDef(cls) => cls,
            BindingClass::FunctionalClassDef(..) => continue,
        };
        if has_function_ancestor(&cls_binding.parent)
            || is_deleted_class(module, bindings, cls_binding)
        {
            continue;
        }

        // Only count instance attrs from recognized methods (__init__, etc.)
        // or a schema class body field. We handle recognized-method fields with full
        // slot classification, and schema body fields as IMPLICIT (0 typable).
        let (annotation, slots) = match &field.definition {
            ClassFieldDefinition::DefinedInMethod {
                annotation, method, ..
            } => {
                if !method.recognized_attribute_defining_method {
                    continue;
                }
                if annotation.is_none()
                    && let Some(slots) = inherited_annotation_slots(
                        bindings,
                        answers,
                        transaction,
                        handle,
                        field.class_idx,
                        &field.name,
                    )
                {
                    (None, slots)
                } else {
                    classify_annotation(module, bindings, answers, *annotation)
                }
            }
            // Schema class fields are always IMPLICIT regardless of whether they're
            // also initialized in a recognized method — the class definition governs
            // their types.
            ClassFieldDefinition::DeclaredByAnnotation { .. }
            | ClassFieldDefinition::AssignedInBody { .. }
                if is_schema_class(bindings, answers, cls_binding) =>
            {
                (None, SlotCounts::default())
            }
            // Non-schema fields only count when initialized in a recognized method,
            // or declared in a stub.
            ClassFieldDefinition::DeclaredByAnnotation {
                annotation,
                initialized_in_recognized_method,
            } => {
                if !initialized_in_recognized_method && !module.path().is_interface() {
                    continue;
                }
                classify_annotation(module, bindings, answers, Some(*annotation))
            }
            _ => continue,
        };

        let class_name = class_fqn(module, &cls_binding.parent, &cls_binding.def.name);
        let range = field.range;
        attrs.push(Variable {
            name: format!("{}.{}", class_name, field.name),
            annotation,
            slots,
            location: range_to_location(module, range),
            range,
        });
    }
    attrs.sort_by_key(|a| a.location);
    attrs
}

fn parse_functions(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    exports: &SmallMap<Name, ExportLocation>,
    dunder_all: &SmallSet<Name>,
    tco_classes: &SmallSet<Idx<KeyClass>>,
) -> Vec<Function> {
    let mut functions = Vec::new();
    let module_prefix = module_prefix(module);
    let deleted = bindings.module_deletes();

    for idx in bindings.keys::<Key>() {
        if let Key::Definition(id) = bindings.idx_to_key(idx)
            && let Binding::Function(x, _pred, _class_meta) = bindings.get(idx)
        {
            let decorated = bindings.get(*x);
            let fun = bindings.get(decorated.undecorated_idx);
            // Skip functions nested inside other functions, even when the name collides with an
            // exported module-level function (gh-4018).
            if fun.outer_funcs.is_some() {
                continue;
            }
            // Skip @type_check_only decorated functions.
            if has_type_check_only_decorator(&fun.decorators, bindings) {
                continue;
            }
            // Skip @no_type_check decorated functions — their bodies are
            // not analyzed, so coverage metrics are meaningless.
            if has_no_type_check_decorator(&fun.decorators, bindings) {
                continue;
            }
            // Skip methods of @type_check_only decorated classes.
            if fun.class_key.is_some_and(|ck| tco_classes.contains(&ck)) {
                continue;
            }
            // Skip overload implementation signatures — only @overload
            // decorated signatures are part of the public API.
            if let Some(pred) = _pred
                && let Binding::Function(pred_x, _, _) = bindings.get(*pred)
            {
                let pred_is_overload = answers
                    .get_idx(bindings.get(*pred_x).undecorated_idx)
                    .is_some_and(|u| u.metadata.flags.is_overload);
                let this_is_overload = answers
                    .get_idx(decorated.undecorated_idx)
                    .is_some_and(|u| u.metadata.flags.is_overload);
                if pred_is_overload && !this_is_overload {
                    continue;
                }
            }
            let range = fun.def.range;
            let location = range_to_location(module, range);
            let func_name = if let Some(class_key) = fun.class_key {
                match bindings.get(class_key) {
                    BindingClass::ClassDef(cls) => {
                        // Skip methods of function-nested and `del`eted classes
                        if has_function_ancestor(&cls.parent)
                            || is_deleted_class(module, bindings, cls)
                        {
                            continue;
                        }
                        // Skip private class methods (single-underscore prefix).
                        if !is_public_name(fun.def.name.as_str()) {
                            continue;
                        }
                        let class_qname = class_fqn(module, &cls.parent, &cls.def.name);
                        format!("{class_qname}.{}", fun.def.name)
                    }
                    BindingClass::FunctionalClassDef(..) => {
                        continue;
                    }
                }
            } else {
                // Keep only public, exported, non-deleted, non-excluded-dunder module-level functions.
                if !exports.contains_key(&fun.def.name.id)
                    || (!is_public_name(fun.def.name.as_str())
                        && !dunder_all.contains(&fun.def.name.id))
                    || deleted.contains(&fun.def.name.id)
                    || EXCLUDED_MODULE_DUNDERS.contains(&fun.def.name.as_str())
                {
                    continue;
                }
                format!("{}{}", module_prefix, fun.def.name)
            };

            let return_idx = bindings.key_to_idx(&Key::ReturnType(*id));
            let return_annotation = return_annotation_range(bindings, return_idx)
                .map(|range| module.code_at(range).to_owned());

            let resolved_return_ty = return_annotation
                .as_ref()
                .and_then(|_| answers.get_type_at(return_idx).map(|t| is_type_known(&t)));
            let is_return_type_known =
                classify_annotation_rank(return_annotation.is_some(), resolved_return_ty)
                    == SlotRank::Typed;

            let mut parameters = Vec::new();
            let implicit_receiver = has_implicit_receiver(fun, answers, decorated.undecorated_idx);
            let all_params = params_with_keys(&fun.def.parameters, implicit_receiver);
            let mut all_params_type_known = true;

            let property_role = answers
                .get_type_at(idx)
                .and_then(|t| t.property_metadata().map(|m| m.role.clone()));
            let is_property_deleter = matches!(property_role, Some(PropertyRole::DeleterDecorator));

            // Implicit dunder returns (e.g. __init__ → None) and property setter/deleter
            // returns are trivial, so they are excluded even when explicitly annotated.
            let skip_return = is_property_deleter
                || matches!(
                    property_role,
                    Some(PropertyRole::Setter | PropertyRole::SetterDecorator)
                )
                || (fun.class_key.is_some() && is_implicit_dunder_return(fun.def.name.as_str()));
            let return_rank = if skip_return {
                SlotRank::Skip
            } else {
                SlotRank::classify(return_annotation.is_some(), is_return_type_known)
            };
            let mut func_slots = SlotCounts::from(return_rank);
            let mut n_params = 0usize;
            let mut non_self_index = 0usize;

            for (merge_key, param) in &all_params {
                let param_name = param.name.as_str();
                let is_self = merge_key.is_none();
                let param_annotation = param
                    .annotation
                    .as_ref()
                    .map(|ann| module.code_at(ann.range()).to_owned());

                let resolved_param_ty = if param.annotation.is_some() {
                    let annot_key = KeyAnnotation::Annotation(ShortIdentifier::new(&param.name));
                    // Use fallible lookup to handle @no_type_check functions gracefully.
                    // The binding pass skips creating KeyAnnotation entries for parameters
                    // of @no_type_check functions since their bodies are not analyzed.
                    bindings
                        .key_to_idx_hashed_opt(Hashed::new(&annot_key))
                        .and_then(|annot_idx| answers.get_idx(annot_idx))
                        .and_then(|awt| awt.annotation.ty.as_ref().map(is_type_known))
                } else {
                    None
                };
                let is_param_type_known = is_self
                    || classify_annotation_rank(param_annotation.is_some(), resolved_param_ty)
                        == SlotRank::Typed;

                // Implicit dunder params are always excluded, even when annotated.
                let is_implicit_param = !is_self
                    && fun.class_key.is_some()
                    && is_implicit_dunder_param(fun.def.name.as_str(), non_self_index);

                // self/cls and implicit params are excluded from slot counting.
                let effective_key = if is_self || is_implicit_param {
                    None
                } else {
                    merge_key.clone()
                };

                if !is_self {
                    non_self_index += 1;
                }

                if !is_param_type_known && effective_key.is_some() {
                    all_params_type_known = false;
                }

                // Deleters have 0 typables; skip parameter slots entirely.
                if effective_key.is_some() && !is_property_deleter {
                    func_slots = func_slots.merge(
                        SlotRank::classify(param_annotation.is_some(), is_param_type_known).into(),
                    );
                    n_params += 1;
                }

                parameters.push(Parameter {
                    name: param_name.to_owned(),
                    annotation: param_annotation,
                    is_type_known: is_param_type_known,
                    merge_key: effective_key,
                    location: range_to_location(module, param.range),
                });
            }

            let is_fully_annotated = return_annotation.is_some()
                && parameters
                    .iter()
                    .all(|p| p.merge_key.is_none() || p.annotation.is_some());
            let is_type_known = is_fully_annotated && is_return_type_known && all_params_type_known;

            functions.push(Function {
                name: func_name,
                return_annotation,
                return_rank,
                parameters,
                is_type_known,
                property_role,
                n_params,
                slots: func_slots,
                location,
                range,
            });
        }
    }
    // Resolve method aliases: class body assignments like `__rand__ = __and__`
    // that point to an existing method. Emit a duplicate Function for the alias.
    for field_idx in bindings.keys::<KeyClassField>() {
        let field = bindings.get(field_idx);
        if let ClassFieldDefinition::AssignedInBody {
            alias_of: Some(target_name),
            ..
        } = &field.definition
        {
            let cls = match bindings.get(field.class_idx) {
                BindingClass::ClassDef(cls) => cls,
                BindingClass::FunctionalClassDef(..) => continue,
            };
            if has_function_ancestor(&cls.parent) {
                continue;
            }
            let class_prefix = class_fqn(module, &cls.parent, &cls.def.name);
            let target_qualified = format!("{}.{}", class_prefix, target_name);
            if let Some(target_func) = functions.iter().find(|f| f.name == target_qualified) {
                let alias_name = format!("{}.{}", class_prefix, field.name);
                let range = field.range;
                let location = range_to_location(module, range);
                functions.push(Function {
                    name: alias_name,
                    slots: target_func.slots,
                    location,
                    range,
                    return_annotation: target_func.return_annotation.clone(),
                    return_rank: target_func.return_rank,
                    parameters: target_func.parameters.clone(),
                    is_type_known: target_func.is_type_known,
                    property_role: target_func.property_role.clone(),
                    n_params: target_func.n_params,
                });
            }
        }
    }
    functions
}

fn return_annotation_range(bindings: &Bindings, return_idx: Idx<Key>) -> Option<TextRange> {
    if let Binding::ReturnType(ret) = bindings.get(return_idx)
        && let ReturnTypeKind::ShouldValidateAnnotation { range, .. }
        | ReturnTypeKind::ShouldTrustAnnotation { range, .. } = &ret.kind
    {
        Some(*range)
    } else {
        None
    }
}

/// Only a bare `Any` counts as unknown; container types like `list[Any]` are known.
fn is_type_known(ty: &Type) -> bool {
    !ty.is_any()
}

/// Dunder methods whose return type is fully determined by the protocol
/// and therefore don't need an explicit annotation for coverage.
fn is_implicit_dunder_return(name: &str) -> bool {
    matches!(
        name,
        "__init__"
            | "__del__"
            | "__init_subclass__"
            | "__post_init__"
            | "__bool__"
            | "__len__"
            | "__length_hint__"
            | "__hash__"
            | "__int__"
            | "__float__"
            | "__complex__"
            | "__index__"
            | "__str__"
            | "__repr__"
            | "__format__"
            | "__bytes__"
            | "__sizeof__"
            | "__contains__"
            | "__setattr__"
            | "__delattr__"
            | "__setitem__"
            | "__delitem__"
            | "__dir__"
            | "__instancecheck__"
            | "__subclasscheck__"
            | "__set__"
            | "__delete__"
            | "__set_name__"
            | "__buffer__"
            | "__release_buffer__"
            | "__mro_entries__"
            | "__subclasses__"
    )
}

/// Dunder method parameters whose types are fully determined by the
/// protocol (e.g. `__exit__`'s exception triple, `__getattr__`'s `name: str`).
/// Positions are 0-indexed after excluding self/cls.
fn is_implicit_dunder_param(name: &str, non_self_param_pos: usize) -> bool {
    match name {
        // __exit__/__aexit__(self, exc_type, exc_val, exc_tb)
        "__exit__" | "__aexit__" => non_self_param_pos <= 2,
        // First non-self param is protocol-fixed (str, int, or memoryview)
        "__getattr__" | "__getattribute__" | "__delattr__" | "__setattr__" | "__format__"
        | "__buffer__" | "__release_buffer__" => non_self_param_pos == 0,
        // __set_name__(self, owner, name: str) — name at position 1
        "__set_name__" => non_self_param_pos == 1,
        _ => false,
    }
}

/// Class-body dunder attributes whose types are implicit (determined by
/// the Python runtime). These should be excluded from coverage counting.
fn is_implicit_dunder_attr(name: &str) -> bool {
    matches!(
        name,
        "__annotations__"
            | "__base__"
            | "__bases__"
            | "__class__"
            | "__dict__"
            | "__doc__"
            | "__firstlineno__"
            | "__match_args__"
            | "__module__"
            | "__mro__"
            | "__name__"
            | "__objclass__"
            | "__qualname__"
            | "__slots__"
            | "__static_attributes__"
            | "__type_params__"
            | "__weakref__"
    )
}

/// Check whether a decorator expression matches the given name.
/// Handles `@name`, `@module.name`, and call forms like `@name(...)`.
fn is_decorator_named(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Name(n) => n.id.as_str() == name,
        Expr::Attribute(attr) => attr.attr.as_str() == name,
        Expr::Call(call) => is_decorator_named(&call.func, name),
        _ => false,
    }
}

/// Check if any decorator in the list is `@type_check_only`.
fn has_type_check_only_decorator(decorators: &[Idx<KeyDecorator>], bindings: &Bindings) -> bool {
    has_decorator_named(decorators, bindings, "type_check_only")
}

/// Check if any decorator in the list is `@no_type_check`.
fn has_no_type_check_decorator(decorators: &[Idx<KeyDecorator>], bindings: &Bindings) -> bool {
    has_decorator_named(decorators, bindings, "no_type_check")
}

/// Check if any decorator in the list matches the given name.
fn has_decorator_named(decorators: &[Idx<KeyDecorator>], bindings: &Bindings, name: &str) -> bool {
    decorators.iter().any(|&dec_idx| {
        let decorator = bindings.get(dec_idx);
        is_decorator_named(&decorator.expr, name)
    })
}

/// Names in an explicit, statically-resolvable `__all__`; `None` if inferred or unresolvable.
fn collect_dunder_all(transaction: &Transaction, handle: &Handle) -> Option<SmallSet<Name>> {
    transaction
        .get_exports_data(handle)
        .get_explicit_dunder_all_names_iter()
        .map(|it| it.cloned().collect())
}

/// The `(module_prefix, __all__ FQNs)` that gate which `.py`-only symbols a stub merge keeps,
/// or `None` when the stub has no explicit `__all__` (leaving the merge unfiltered).
fn stub_merge_filter(
    transaction: &Transaction,
    handle: &Handle,
) -> Option<(String, HashSet<String>)> {
    collect_dunder_all(transaction, handle).map(|all| {
        let module_prefix = format!("{}.", handle.module());
        let fqns = all.iter().map(|n| format!("{module_prefix}{n}")).collect();
        (module_prefix, fqns)
    })
}

/// Collect all class keys that have the `@type_check_only` decorator.
fn collect_type_check_only_classes(bindings: &Bindings) -> SmallSet<Idx<KeyClass>> {
    let mut tco_classes = SmallSet::new();
    for idx in bindings.keys::<Key>() {
        if let Binding::ClassDef(class_key, decorators) = bindings.get(idx)
            && has_type_check_only_decorator(decorators, bindings)
        {
            tco_classes.insert(*class_key);
        }
    }
    tco_classes
}

/// Determine whether a function name represents a method (contains '.', i.e. `Cls.method`).
fn is_method(name: &str, module_prefix: &str) -> bool {
    let without_prefix = name.strip_prefix(module_prefix).unwrap_or(name);
    without_prefix.contains('.')
}

/// Count symbols by kind. The caller sets `n_type_ignores` separately.
fn count_symbols(symbol_reports: &[SymbolReport], module_prefix: &str) -> SymbolCounts {
    let mut symbols = SymbolCounts::default();
    for sym in symbol_reports {
        match sym {
            SymbolReport::Function { name, n_params, .. } if is_method(name, module_prefix) => {
                symbols.n_methods += 1;
                symbols.n_method_params += *n_params;
            }
            SymbolReport::Function { n_params, .. } => {
                symbols.n_functions += 1;
                symbols.n_function_params += *n_params;
            }
            SymbolReport::Class { .. } => symbols.n_classes += 1,
            SymbolReport::Attr { .. } => symbols.n_attrs += 1,
            SymbolReport::Property { .. } => symbols.n_properties += 1,
        }
    }
    symbols
}

/// Calculate the aggregate summary by summing per-module symbol counts.
pub fn calculate_summary(module_reports: &[ModuleReport]) -> ReportSummary {
    let mut slots = SlotCounts::default();
    let mut symbols = SymbolCounts::default();
    for module in module_reports {
        slots = slots.merge(module.slots);
        symbols = symbols.merge(module.symbols);
    }
    ReportSummary {
        n_modules: module_reports.len(),
        slots,
        coverage: slots.coverage(),
        strict_coverage: slots.strict_coverage(),
        symbols,
    }
}

fn parse_classes(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    tco_classes: &SmallSet<Idx<KeyClass>>,
) -> Vec<ReportClass> {
    let mut classes = Vec::new();

    for class_idx in bindings.keys::<KeyClass>() {
        // Skip @type_check_only classes.
        if tco_classes.contains(&class_idx) {
            continue;
        }
        let cls_binding = match bindings.get(class_idx) {
            BindingClass::ClassDef(cls) => cls,
            BindingClass::FunctionalClassDef(..) => continue,
        };
        let parent = &cls_binding.parent;
        let name = &cls_binding.def.name;
        // Skip classes nested inside functions, since they are not public symbols.
        if has_function_ancestor(parent) {
            continue;
        }
        if is_deleted_class(module, bindings, cls_binding) {
            continue;
        }
        // Skip classes that fail to resolve.
        if answers.get_idx(class_idx).is_none_or(|c| c.0.is_none()) {
            continue;
        }
        classes.push(ReportClass {
            name: class_fqn(module, parent, name),
            location: range_to_location(module, cls_binding.def.range),
        });
    }
    classes.sort();
    classes
}

/// Returns the set of `.py` paths that should be skipped because a
/// corresponding `.pyi` file also appears in `handles`.
fn py_paths_shadowed_by_pyi(handles: &[Handle]) -> SmallSet<PathBuf> {
    handles
        .iter()
        .filter(|h| h.path().is_interface())
        .map(|h| h.path().as_path().with_extension("py"))
        .collect()
}

/// `module.Cls.member` names for each public class, including MRO-inherited ones.
fn collect_class_members(
    module: &Module,
    bindings: &Bindings,
    answers: &Answers,
    transaction: &Transaction,
    handle: &Handle,
    tco_classes: &SmallSet<Idx<KeyClass>>,
) -> SmallSet<String> {
    let mut members = SmallSet::new();
    for idx in bindings.keys::<KeyClass>() {
        if tco_classes.contains(&idx) {
            continue;
        }
        let BindingClass::ClassDef(binding) = bindings.get(idx) else {
            continue;
        };
        if has_function_ancestor(&binding.parent) {
            continue;
        }
        let Some(cls) = answers.get_idx(idx).and_then(|r| r.0.clone()) else {
            continue;
        };

        let fqname = class_fqn(module, &binding.parent, &binding.def.name);

        let mro = class_mro(bindings, answers, &cls);
        let ancestors = mro.ancestors_no_object();
        for obj in std::iter::once(&cls).chain(ancestors.iter().map(ClassType::class_object)) {
            if obj.module_name().as_str() == "builtins" {
                continue;
            }
            if let Some(fields) = transaction.get_class_fields(handle, obj) {
                for name in fields.names() {
                    members.insert(format!("{fqname}.{name}"));
                }
            }
        }
    }
    members
}

/// FQNs the stub re-exports from other modules
///
/// In a `.pyi` an import is only a re-export when its name is listed in `__all__` or written
/// as a `from x import Y as Y` alias. A plain `from x import Y` is a private implementation detail,
/// and does not obscure a same-named definition in the `.py`.
fn collect_reexport_fqns(
    module: &Module,
    exports: &SmallMap<Name, ExportLocation>,
    exports_data: &Exports,
    dunder_all: &SmallSet<Name>,
) -> SmallSet<String> {
    let module_prefix = module_prefix(module);
    exports
        .iter()
        .filter(|&(name, loc)| {
            matches!(loc, ExportLocation::OtherModule(..))
                && (dunder_all.contains(name) || exports_data.is_explicit_reexport(name))
        })
        .map(|(name, _)| format!("{module_prefix}{name}"))
        .collect()
}

/// Stub-side inputs of `merge_uncovered_py_symbols`, captured while the stub's
/// bindings/answers are still live.
struct StubMergeData {
    class_members: SmallSet<String>,
    all_filter: Option<(String, HashSet<String>)>,
    reexports: SmallSet<String>,
}

struct ModuleSymbols {
    module: Module,
    functions: Vec<Function>,
    variables: Vec<Variable>,
    classes: Vec<ReportClass>,
    suppressions: Vec<ReportSuppression>,
    /// `Some` only when collected with `for_stub_merge`.
    stub_merge: Option<StubMergeData>,
}

impl ModuleSymbols {
    /// Parse a solved module's symbols into plain data; `for_stub_merge` also captures
    /// `StubMergeData`.
    fn collect(transaction: &Transaction, handle: &Handle, for_stub_merge: bool) -> Option<Self> {
        let bindings = transaction.get_bindings(handle)?;
        let module = transaction.get_module_info(handle)?;
        let answers = transaction.get_answers(handle)?;
        let exports = transaction.get_exports(handle);
        let dunder_all = collect_dunder_all(transaction, handle).unwrap_or_default();
        let tco_classes = collect_type_check_only_classes(&bindings);
        let mut functions = parse_functions(
            &module,
            &bindings,
            &answers,
            &exports,
            &dunder_all,
            &tco_classes,
        );
        merge_overloads(&mut functions);
        let classes = parse_classes(&module, &bindings, &answers, &tco_classes);
        let mut variables = parse_variables(
            &module,
            &bindings,
            &answers,
            &exports,
            &dunder_all,
            &functions,
            &classes,
        );
        variables.extend(parse_instance_attrs(
            &module,
            &bindings,
            &answers,
            transaction,
            handle,
            &tco_classes,
        ));
        let suppressions = parse_suppressions(&module);
        let stub_merge = for_stub_merge.then(|| StubMergeData {
            class_members: collect_class_members(
                &module,
                &bindings,
                &answers,
                transaction,
                handle,
                &tco_classes,
            ),
            all_filter: stub_merge_filter(transaction, handle),
            reexports: collect_reexport_fqns(
                &module,
                &exports,
                &transaction.get_exports_data(handle),
                &dunder_all,
            ),
        });
        Some(ModuleSymbols {
            module,
            functions,
            variables,
            classes,
            suppressions,
            stub_merge,
        })
    }

    fn line_count(&self) -> usize {
        self.module.lined_buffer().line_index().line_count()
    }

    /// When this `.pyi` stub only covers a subset of its `.py` counterpart's public
    /// symbols, add the uncovered `py` symbols so that completeness metrics reflect
    /// the full module interface. Merged symbols count as fully untyped, since type
    /// checkers ignore the `.py` when a stub exists.
    fn merge_uncovered_py_symbols(&mut self, py: ModuleSymbols) {
        let StubMergeData {
            class_members,
            all_filter,
            reexports,
        } = self
            .stub_merge
            .take()
            .expect("stub must be collected with `for_stub_merge` to merge `.py` symbols");

        // Dedupe py-side symbols against all stub-side names regardless of kind, so a name defined
        // as a function or class in the stub isn't re-added as an untyped attr by a .py re-export.
        let stub_names: SmallSet<String> = self
            .functions
            .iter()
            .map(|f| &f.name)
            .chain(self.variables.iter().map(|v| &v.name))
            .chain(self.classes.iter().map(|c| &c.name))
            .cloned()
            .collect();
        // Keep py-only names the stub neither defines nor omits from an explicit `__all__`.
        let keep = |name: &str| {
            !stub_names.contains(name)
                && all_filter
                    .as_ref()
                    .is_none_or(|(prefix, fqns)| is_public_fqn(name, prefix, fqns))
        };
        // A re-exported name owns its whole `Name.*` subtree: when the stub re-exports a class,
        // the .py class AND its attributes must drop together.
        let is_reexported = |name: &str| {
            reexports.iter().any(|reexport| {
                name == reexport.as_str()
                    || name
                        .strip_prefix(reexport.as_str())
                        .is_some_and(|rest| rest.starts_with('.'))
            })
        };
        for mut py_func in py.functions {
            if keep(&py_func.name)
                && !class_members.contains(&py_func.name)
                && !is_reexported(&py_func.name)
            {
                py_func.slots = py_func.slots.as_untyped();
                self.functions.push(py_func);
            }
        }
        for mut py_var in py.variables {
            if keep(&py_var.name)
                && !class_members.contains(&py_var.name)
                && !is_reexported(&py_var.name)
            {
                py_var.slots = py_var.slots.as_untyped();
                self.variables.push(py_var);
            }
        }
        for py_class in py.classes {
            if keep(&py_class.name) && !is_reexported(&py_class.name) {
                self.classes.push(py_class);
            }
        }
    }
}

fn build_module_report(
    name: String,
    path: String,
    derived_name: &str,
    line_count: usize,
    functions: &[Function],
    variables: &[Variable],
    classes: &[ReportClass],
    suppressions: Vec<ReportSuppression>,
) -> ModuleReport {
    let mut symbol_reports = Vec::new();
    let mut total_slots = SlotCounts::default();
    let mut names = Vec::new();

    for var in variables {
        total_slots = total_slots.merge(var.slots);
        names.push(var.name.clone());
        symbol_reports.push(SymbolReport::Attr {
            name: var.name.clone(),
            slots: var.slots,
            location: var.location,
        });
    }

    // Merge same-name property accessors into a single report entry.
    let mut property_map: Vec<(String, SlotCounts, Location)> = Vec::new();
    for func in functions {
        if func.property_role.is_some() {
            if let Some(entry) = property_map.iter_mut().find(|(n, _, _)| *n == func.name) {
                entry.1 = entry.1.merge(func.slots);
            } else {
                property_map.push((func.name.clone(), func.slots, func.location));
            }
        } else {
            total_slots = total_slots.merge(func.slots);
            names.push(func.name.clone());
            symbol_reports.push(SymbolReport::Function {
                name: func.name.clone(),
                slots: func.slots,
                n_params: func.n_params,
                location: func.location,
            });
        }
    }
    for (name, slots, location) in &property_map {
        total_slots = total_slots.merge(*slots);
        names.push(name.clone());
        symbol_reports.push(SymbolReport::Property {
            name: name.clone(),
            slots: *slots,
            location: *location,
        });
    }

    for cls in classes {
        names.push(cls.name.clone());
        symbol_reports.push(SymbolReport::Class {
            name: cls.name.clone(),
            slots: SlotCounts::default(),
            location: cls.location,
        });
    }

    // Overloads and property accessors produce duplicate names.
    let mut seen = SmallSet::new();
    names.retain(|n| seen.insert(n.clone()));

    // Match prefixes against the derived (file-based) name: symbol names are
    // built from it and only rewritten to the override name later.
    let module_prefix = format!("{}.", derived_name);
    let symbols = SymbolCounts {
        n_type_ignores: suppressions.len(),
        ..count_symbols(&symbol_reports, &module_prefix)
    };

    // A `--module` override renames the module, so rewrite symbol prefixes to match.
    if name != derived_name {
        let rewrite = |s: &mut String| {
            if let Some(rest) = s.strip_prefix(&module_prefix) {
                *s = format!("{name}.{rest}");
            }
        };
        for n in &mut names {
            rewrite(n);
        }
        for sym in &mut symbol_reports {
            rewrite(sym.name_mut());
        }
    }

    ModuleReport {
        name,
        path,
        names,
        line_count,
        symbol_reports,
        type_ignores: suppressions,
        coverage: total_slots.coverage(),
        strict_coverage: total_slots.strict_coverage(),
        slots: total_slots,
        symbols,
    }
}

fn is_untyped(slots: &SlotCounts, strict: bool) -> bool {
    slots.n_untyped > 0 || (strict && slots.n_any > 0)
}

fn untyped_error(
    module: &Module,
    slots: &SlotCounts,
    strict: bool,
    range: TextRange,
    name: &str,
) -> Error {
    // Under --strict, `Any`-annotated slots also count as untyped (gh-4024).
    let n_untyped = slots.n_untyped + if strict { slots.n_any } else { 0 };
    let (kind, desc) = if n_untyped == slots.n_typable {
        (ErrorKind::CoverageMissing, "is untyped")
    } else {
        (ErrorKind::CoveragePartial, "is not fully typed")
    };
    Error::new(
        module.dupe(),
        range,
        format!("`{name}` {desc}"),
        Vec::new(),
        kind,
    )
}

fn collect_untyped_errors(
    errors: &mut Vec<Error>,
    module: &Module,
    functions: &[Function],
    variables: &[Variable],
    strict: bool,
    public_fqns: Option<&HashSet<String>>,
) {
    let module_prefix = format!("{}.", module.name());
    let is_public =
        |name: &str| public_fqns.is_none_or(|fqns| is_public_fqn(name, &module_prefix, fqns));
    // Property accessors share a name; emit one error per property, as `report` merges them.
    let mut seen_properties = SmallSet::new();
    for func in functions {
        if is_untyped(&func.slots, strict)
            && is_public(&func.name)
            && (func.property_role.is_none() || seen_properties.insert(&func.name))
        {
            errors.push(untyped_error(
                module,
                &func.slots,
                strict,
                func.range,
                &func.name,
            ));
        }
    }
    for var in variables {
        if is_untyped(&var.slots, strict) && is_public(&var.name) {
            errors.push(untyped_error(
                module, &var.slots, strict, var.range, &var.name,
            ));
        }
    }
}

pub fn collect_module_reports(
    files_to_check: Box<dyn Includes>,
    config_finder: ConfigFinder,
    prefer_stubs: bool,
    module_name_override: Option<String>,
    public_only: bool,
    untyped_strict: Option<bool>,
    thread_count: ThreadCount,
) -> anyhow::Result<(Vec<ModuleReport>, Vec<Error>)> {
    let expanded_file_list = config_finder.checkpoint(files_to_check.files_iter())?;
    let state = State::new(config_finder, thread_count);
    let holder = Forgetter::new(state, false);
    let handles = Handles::new(expanded_file_list);
    let (handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());

    if !sourcedb_errors.is_empty() {
        for error in sourcedb_errors {
            error.print();
        }
        return Err(anyhow::anyhow!("Failed to query sourcedb."));
    }

    let shadowed = if prefer_stubs {
        py_paths_shadowed_by_pyi(&handles)
    } else {
        SmallSet::new()
    };

    // Map each .pyi to its corresponding .py: first co-located,
    // then by module-name lookup in site-package-path.
    let pyi_to_py: HashMap<PathBuf, Handle> = if prefer_stubs {
        let py_by_path: HashMap<PathBuf, &Handle> = handles
            .iter()
            .filter(|h| !h.path().is_interface())
            .map(|h| (h.path().as_path().to_path_buf(), h))
            .collect();
        let mut map: HashMap<PathBuf, Handle> = handles
            .iter()
            .filter(|h| h.path().is_interface())
            .filter_map(|h| {
                let py_path = h.path().as_path().with_extension("py");
                py_by_path
                    .get(&py_path)
                    .map(|&py_h| (h.path().as_path().to_path_buf(), py_h.clone()))
            })
            .collect();
        // Fall back to site-package-path for stubs-only packages.
        for h in handles.iter().filter(|h| h.path().is_interface()) {
            let pyi_path = h.path().as_path().to_path_buf();
            if map.contains_key(&pyi_path) {
                continue;
            }
            let config = holder
                .as_ref()
                .config_finder()
                .python_file(h.module_kind(), h.path());
            if let Some(py_module_path) = find_import_filtered(
                &config,
                h.module(),
                None,
                Some(ModuleStyle::Executable),
                &DirEntryCache::new(),
                None,
            )
            .finding()
            {
                let py_handle = config.handle_from_module_path(py_module_path);
                map.insert(pyi_path, py_handle);
            }
        }
        map
    } else {
        HashMap::new()
    };
    let config_finder = holder.as_ref().config_finder();
    let dir_cache = DirEntryCache::new();
    // gh-3632: skip files whose module name isn't importable (shadowed parent).
    let importable = |handle: &Handle| {
        handle.module() == ModuleName::unknown() || {
            let config = config_finder.python_file(handle.module_kind(), handle.path());
            find_import_filtered(&config, handle.module(), None, None, &dir_cache, None)
                .finding()
                .is_some()
        }
    };
    let mut targets: Vec<Handle> = handles
        .iter()
        .filter(|h| !shadowed.contains(h.path().as_path()) && importable(h))
        .cloned()
        .collect();
    // Targets plus each stub's `.py` counterpart; the flag marks stubs that merge one.
    let to_collect: HashMap<Handle, bool> = targets
        .iter()
        .map(|h| {
            let merges_py = pyi_to_py.contains_key(&h.path().as_path().to_path_buf());
            (h.dupe(), merges_py)
        })
        .chain(pyi_to_py.values().map(|h| (h.dupe(), false)))
        .collect();
    let run_set: Vec<Handle> = to_collect.keys().cloned().collect();
    let collected: Mutex<HashMap<Handle, ModuleSymbols>> = Mutex::new(HashMap::new());

    let mut forgetter = Forgetter::new(
        holder.as_ref().new_transaction(Require::Exports, None),
        true,
    );
    let transaction = forgetter.as_mut();
    // Collect each module the moment it solves, before the run evicts its bindings/answers,
    // so peak memory holds only the solver's working set (gh-3989).
    transaction.set_solutions_hook(Some(Box::new(|handle, transaction| {
        let Some(&for_stub_merge) = to_collect.get(handle) else {
            return;
        };
        if let Some(symbols) = ModuleSymbols::collect(transaction, handle, for_stub_merge) {
            collected.lock().insert(handle.dupe(), symbols);
        }
    })));
    transaction.run(&run_set, Require::Errors, None);
    // Later lazy computation must not fire the hook.
    transaction.set_solutions_hook(None);

    // gh-4034: a symbol re-exported from a public module stays counted even when
    // `--project-excludes` drops its defining file; collect such origins in a 2nd pass.
    let public_fqns = public_only.then(|| {
        let (fqns, origins) = compute_public_fqns(&handles, transaction);
        // Dedupe by path and module name so an origin that discovery already covers
        // (e.g. an excluded .pyi of a discovered .py) isn't reported twice.
        let mut seen_paths: HashSet<&Path> = handles.iter().map(|h| h.path().as_path()).collect();
        let mut seen_modules: HashSet<ModuleName> = handles.iter().map(|h| h.module()).collect();
        let extras: Vec<Handle> = origins
            .iter()
            .filter(|h| {
                let path = h.path().as_path();
                seen_paths.insert(path)
                    && seen_modules.insert(h.module())
                    && files_to_check.covers_ignoring_excludes(path)
                    && !config_finder
                        .python_file(h.module_kind(), h.path())
                        .site_package_path()
                        .any(|sp| path.starts_with(sp))
            })
            .cloned()
            .collect();
        if !extras.is_empty() {
            // `Everything` retains bindings/answers so the extras can be collected after the run
            transaction.run(&extras, Require::Everything, None);
            collected.lock().extend(extras.iter().filter_map(|handle| {
                ModuleSymbols::collect(transaction, handle, false).map(|s| (handle.dupe(), s))
            }));
            targets.extend(extras);
        }
        fqns
    });

    let mut module_reports: Vec<ModuleReport> = Vec::new();
    let mut errors: Vec<Error> = Vec::new();
    let mut collected = collected.lock();
    for handle in &targets {
        let Some(mut symbols) = collected.remove(handle) else {
            continue;
        };
        // Per source module, so stub-merged `.py` symbols render against their own file.
        if let Some(strict) = untyped_strict {
            collect_untyped_errors(
                &mut errors,
                &symbols.module,
                &symbols.functions,
                &symbols.variables,
                strict,
                public_fqns.as_ref(),
            );
        }

        // When a .pyi stub shadows a .py file, include uncovered .py symbols.
        if let Some(py_handle) = pyi_to_py.get(&handle.path().as_path().to_path_buf())
            && let Some(py_symbols) = collected.remove(py_handle)
        {
            let py_module = py_symbols.module.dupe();
            let own_functions = symbols.functions.len();
            let own_variables = symbols.variables.len();
            symbols.merge_uncovered_py_symbols(py_symbols);
            if let Some(strict) = untyped_strict {
                collect_untyped_errors(
                    &mut errors,
                    &py_module,
                    &symbols.functions[own_functions..],
                    &symbols.variables[own_variables..],
                    strict,
                    public_fqns.as_ref(),
                );
            }
        }

        let derived_name = handle.module().to_string();
        let name = module_name_override.clone().unwrap_or(derived_name.clone());
        let path = handle.path().as_path().display().to_string();
        module_reports.push(build_module_report(
            name,
            path,
            &derived_name,
            symbols.line_count(),
            &symbols.functions,
            &symbols.variables,
            &symbols.classes,
            symbols.suppressions,
        ));
    }

    if let Some(public_fqns) = &public_fqns {
        for report in &mut module_reports {
            filter_module_report_to_public(report, public_fqns);
        }
        module_reports.retain(|r| !r.symbol_reports.is_empty());
    }

    // `handles` iterate in nondeterministic `HashSet` order; path disambiguates `--module`.
    module_reports.sort_by(|a, b| (&a.name, &a.path).cmp(&(&b.name, &b.path)));
    errors.sort_by_key(|e| (e.path().to_string(), e.range().start()));

    // Surface config warnings, like `check` does.
    config_finder.print_errors();

    Ok((module_reports, errors))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;
    use pyrefly_util::globs::FilteredGlobs;
    use pyrefly_util::globs::Globs;
    use pyrefly_util::globs::HiddenDirFilter;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
    use tempfile::TempDir;

    use super::*;
    use crate::commands::config_finder::default_config_finder;
    use crate::state::require::Require;
    use crate::test::util::TestEnv;

    /// Load a checked-in test file from the COVERAGE_TEST_PATH directory.
    /// Normalizes `\r\n` to `\n` so snapshots pass on Windows.
    fn load_test_file(name: &str) -> String {
        let path =
            std::env::var("COVERAGE_TEST_PATH").expect("COVERAGE_TEST_PATH env var must be set");
        std::fs::read_to_string(PathBuf::from(path).join(name))
            .unwrap_or_else(|e| panic!("failed to read test file {name}: {e}"))
            .replace("\r\n", "\n")
    }

    /// Compare serialized JSON output against a checked-in expected file.
    /// When `COVERAGE_TEST_WRITE_PATH` is set, writes the actual output to that
    /// directory instead of comparing (use this to update snapshots).
    fn compare_snapshot<T: serde::Serialize>(name: &str, actual: &T) {
        let actual_json = serde_json::to_string_pretty(actual).unwrap();

        if let Ok(write_path) = std::env::var("COVERAGE_TEST_WRITE_PATH") {
            let out = PathBuf::from(write_path).join(name);
            std::fs::write(&out, format!("{}\n", actual_json.trim()))
                .unwrap_or_else(|e| panic!("failed to write snapshot {}: {e}", out.display()));
            println!("Updated snapshot: {}", out.display());
            return;
        }

        let expected_json = load_test_file(name);
        pretty_assertions::assert_eq!(
            actual_json.trim(),
            expected_json.trim(),
            "Snapshot mismatch for {name}. To update, run with \
             COVERAGE_TEST_WRITE_PATH set to the test_files directory."
        );
    }

    /// Run the production parse pipeline on a checked-in test file, as module `test`.
    fn parse_test_module(file: &str, mut env: TestEnv) -> (ModuleSymbols, String) {
        let code = load_test_file(file);
        let module_path = format!(
            "test.{}",
            file.rsplit_once('.').map_or("py", |(_, ext)| ext)
        );
        env.add_with_path("test", &module_path, &code);

        let (state, handle_fn) = env
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let symbols = ModuleSymbols::collect(&state.transaction(), &handle, false).unwrap();
        (symbols, module_path)
    }

    fn build_module_report_for_test_with_env(file: &str, env: TestEnv) -> ModuleReport {
        let (p, module_path) = parse_test_module(file, env);
        build_module_report(
            "test".to_owned(),
            module_path,
            "test",
            p.line_count(),
            &p.functions,
            &p.variables,
            &p.classes,
            p.suppressions,
        )
    }

    /// Build a `ModuleReport` from a checked-in Python test file,
    /// mirroring the production pipeline in `collect_module_reports`.
    fn build_module_report_for_test(py_file: &str) -> ModuleReport {
        build_module_report_for_test_with_env(py_file, TestEnv::new())
    }

    /// Build a `ModuleReport` for the `--module` override case (file parses as `test`).
    fn build_module_report_with_override(py_file: &str, override_name: &str) -> ModuleReport {
        let (p, module_path) = parse_test_module(py_file, TestEnv::new());
        build_module_report(
            override_name.to_owned(),
            module_path,
            "test",
            p.line_count(),
            &p.functions,
            &p.variables,
            &p.classes,
            p.suppressions,
        )
    }

    /// Merge a `.pyi` stub's symbols with its `.py` source, mirroring the production
    /// pipeline in `collect_module_reports` when `prefer_stubs` is true and both
    /// files exist for the same module.
    fn merged_stub_symbols(pyi_file: &str, py_file: &str) -> ModuleSymbols {
        let pyi_code = load_test_file(pyi_file);
        let (pyi_state, pyi_handle_fn) = TestEnv::one_with_path("test", "test.pyi", &pyi_code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let pyi_handle = pyi_handle_fn("test");
        let mut stub = ModuleSymbols::collect(&pyi_state.transaction(), &pyi_handle, true).unwrap();

        let (py, _) = parse_test_module(py_file, TestEnv::new());
        stub.merge_uncovered_py_symbols(py);
        stub
    }

    fn build_stub_module_report(pyi_file: &str, py_file: &str) -> ModuleReport {
        let stub = merged_stub_symbols(pyi_file, py_file);
        build_module_report(
            "test".to_owned(),
            "test.pyi".to_owned(),
            "test",
            stub.line_count(),
            &stub.functions,
            &stub.variables,
            &stub.classes,
            stub.suppressions,
        )
    }

    #[test]
    fn test_report_suppressions() {
        let report = build_module_report_for_test("suppressions.py");
        compare_snapshot("suppressions.expected.json", &report);
    }

    #[test]
    fn test_report_variables() {
        let report = build_module_report_for_test("variables.py");
        compare_snapshot("variables.expected.json", &report);
    }

    /// gh-3773: optional imports must not be reported as untyped variables.
    #[test]
    fn test_report_optional_imports() {
        let report = build_module_report_for_test("optional_imports.py");
        compare_snapshot("optional_imports.expected.json", &report);
    }

    #[test]
    fn test_report_multi_target_aliases() {
        let report = build_module_report_for_test("multi_target_aliases.py");
        compare_snapshot("multi_target_aliases.expected.json", &report);
    }

    #[test]
    fn test_report_functions() {
        let report = build_module_report_for_test("functions.py");
        compare_snapshot("functions.expected.json", &report);
    }

    #[test]
    fn test_report_incomplete_methods() {
        let report = build_module_report_for_test("incomplete_methods.py");
        compare_snapshot("incomplete_methods.expected.json", &report);
    }

    #[test]
    fn test_report_nested_classes() {
        let report = build_module_report_for_test("nested_classes.py");
        compare_snapshot("nested_classes.expected.json", &report);
    }

    #[test]
    fn test_report_inheritance() {
        let report = build_module_report_for_test("inheritance.py");
        compare_snapshot("inheritance.expected.json", &report);
    }

    #[test]
    fn test_report_nested_exclusions() {
        let report = build_module_report_for_test("nested_exclusions.py");
        compare_snapshot("nested_exclusions.expected.json", &report);
    }

    /// When a .pyi stub only covers a subset of the .py file's exports,
    /// the uncovered symbols appear as unannotated and reduce completeness.
    #[test]
    fn test_report_partial_stub() {
        let report = build_stub_module_report("partial_stub.pyi", "partial_stub.py");
        compare_snapshot("partial_stub.expected.json", &report);
    }

    /// gh-3524: stub functions/classes must not be re-added as untyped attrs by .py re-exports.
    #[test]
    fn test_report_stub_reexport() {
        let report = build_stub_module_report("stub_reexport.pyi", "stub_reexport.py");
        compare_snapshot("stub_reexport.expected.json", &report);
    }

    /// gh-3626: a .py-only symbol the stub omits from an explicit `__all__` must not be merged in.
    #[test]
    fn test_report_stub_dunder_all_filters_non_all() {
        let report = build_stub_module_report("dunder_all_stub.pyi", "dunder_all_stub.py");
        compare_snapshot("dunder_all_stub.expected.json", &report);
    }

    /// gh-3641: a stub re-export (import) must not be re-added as untyped by the .py def.
    #[test]
    fn test_report_stub_reexport_import() {
        let report =
            build_stub_module_report("stub_reexport_import.pyi", "stub_reexport_import.py");
        compare_snapshot("stub_reexport_import.expected.json", &report);
    }

    /// gh-3641: a plain `from x import Y` in a stub (not in `__all__`, not an `as` alias) is a
    /// private import, so a same-named public `.py` def must still appear in the report.
    #[test]
    fn test_report_stub_private_import() {
        let report = build_stub_module_report("stub_private_import.pyi", "stub_private_import.py");
        compare_snapshot("stub_private_import.expected.json", &report);
    }

    /// When a stub re-exports a class, the .py class is suppressed; its methods and attrs must
    /// drop with it rather than dangling under a class that no longer appears in the report.
    #[test]
    fn test_report_stub_reexport_class() {
        let report = build_stub_module_report("stub_reexport_class.pyi", "stub_reexport_class.py");
        compare_snapshot("stub_reexport_class.expected.json", &report);
    }

    /// gh-3778: `.py`-only symbols count as fully untyped, even when annotated in the `.py`.
    #[test]
    fn test_report_stub_ignores_py_annotations() {
        let report = build_stub_module_report(
            "stub_ignores_py_annotations.pyi",
            "stub_ignores_py_annotations.py",
        );
        compare_snapshot("stub_ignores_py_annotations.expected.json", &report);
    }

    /// gh-3519: don't double-count methods whose stub coverage is inherited.
    #[test]
    fn test_report_inherited_method_via_stub() {
        let report =
            build_stub_module_report("stub_inherited_methods.pyi", "stub_inherited_methods.py");
        compare_snapshot("stub_inherited_methods.expected.json", &report);
    }

    #[test]
    fn test_report_stub_class_attrs() {
        let report = build_module_report_for_test("stub_class_attrs.pyi");
        compare_snapshot("stub_class_attrs.expected.json", &report);
    }

    /// Stubs-only packages: .py discovered via site-package-path, merged like co-located stubs.
    #[test]
    fn test_report_external_stub_merge() {
        use pyrefly_config::config::ConfigFile;

        let site_dir = tempfile::TempDir::new().unwrap();
        let py_code = load_test_file("partial_stub.py");
        std::fs::write(site_dir.path().join("test.py"), &py_code).unwrap();

        let mut config = ConfigFile::default();
        config.python_environment.site_package_path = Some(vec![site_dir.path().to_path_buf()]);
        config.interpreters.skip_interpreter_query = true;
        config.configure();

        let py_module_path = find_import_filtered(
            &config,
            ModuleName::from_str("test"),
            None,
            Some(ModuleStyle::Executable),
            &DirEntryCache::new(),
            None,
        )
        .finding()
        .expect("should discover test.py in site-packages");
        assert_eq!(py_module_path.as_path(), site_dir.path().join("test.py"));

        // the merge should produce the same report as the co-located case
        let report = build_stub_module_report("partial_stub.pyi", "partial_stub.py");
        compare_snapshot("partial_stub.expected.json", &report);
    }

    /// gh-3632: skip a file shadowed by a same-named module (it can't be imported).
    #[test]
    fn test_report_skips_unimportable_shadowed_module() {
        use pyrefly_config::config::ConfigFile;

        let site = tempfile::TempDir::new().unwrap();
        let dir = site.path();
        std::fs::write(dir.join("lapack_lite.pyi"), "def f() -> int: ...\n").unwrap();
        std::fs::create_dir(dir.join("lapack_lite")).unwrap();
        std::fs::write(dir.join("lapack_lite").join("fortran.pyi"), "x = 1\n").unwrap();

        let mut config = ConfigFile::default();
        config.python_environment.site_package_path = Some(vec![dir.to_path_buf()]);
        config.interpreters.skip_interpreter_query = true;
        config.configure();

        let cache = DirEntryCache::new();
        let find = |m| {
            find_import_filtered(&config, ModuleName::from_str(m), None, None, &cache, None)
                .finding()
                .is_some()
        };
        assert!(find("lapack_lite"), "real module importable");
        assert!(!find("lapack_lite.fortran"), "shadowed file skipped");
    }

    /// When both test.py and test.pyi exist, the .py file is shadowed.
    #[test]
    fn test_pyi_shadows_py_in_report() {
        let sys_info = SysInfo::default();
        let py_handle = Handle::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.py")),
            sys_info.dupe(),
        );
        let py_handle2 = Handle::new(
            ModuleName::from_str("test2"),
            ModulePath::memory(PathBuf::from("test2.py")),
            sys_info.dupe(),
        );
        let pyi_handle = Handle::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.pyi")),
            sys_info.dupe(),
        );
        let handles = vec![py_handle, py_handle2, pyi_handle];
        let shadowed = py_paths_shadowed_by_pyi(&handles);

        assert!(
            shadowed.contains(PathBuf::from("test.py").as_path()),
            "test.py should be shadowed when test.pyi exists"
        );
        assert!(
            !shadowed.contains(PathBuf::from("test.pyi").as_path()),
            "test.pyi should not be shadowed"
        );
        assert!(
            !shadowed.contains(PathBuf::from("test2.py").as_path()),
            "test2.py should not be shadowed"
        );
    }

    #[test]
    fn test_is_untyped() {
        // Non-strict: only a fully-unannotated slot counts; `Any` is covered.
        assert!(!is_untyped(&SlotCounts::typed(), false));
        assert!(!is_untyped(&SlotCounts::any(), false));
        assert!(is_untyped(&SlotCounts::untyped(), false));
        // Strict: `Any`-typed slots count as untyped too.
        assert!(!is_untyped(&SlotCounts::typed(), true));
        assert!(is_untyped(&SlotCounts::any(), true));
        assert!(is_untyped(&SlotCounts::untyped(), true));
    }

    #[test]
    fn test_check_findings_match_report_symbols() {
        for file in [
            "any_annotations.py",
            "any_detection.py",
            "decorators.py",
            "dunder_attrs.py",
            "dunder_implicit.py",
            "functions.py",
            "incomplete_methods.py",
            "inheritance.py",
            "inherited_attrs.py",
            "instance_attrs.py",
            "method_aliases.py",
            "optional_imports.py",
            "overloads.py",
            "overloads_partial.py",
            "partial_any.py",
            "property_basic.py",
            "schema_classes_methods.py",
            "stub_ignores_py_annotations.pyi",
            "variables.py",
        ] {
            // `.pyi` entries are stub-merged with their `.py` source.
            let (p, module_path) = if let Some(stem) = file.strip_suffix(".pyi") {
                (
                    merged_stub_symbols(file, &format!("{stem}.py")),
                    "test.pyi".to_owned(),
                )
            } else {
                parse_test_module(file, TestEnv::new())
            };

            let report = build_module_report(
                "test".to_owned(),
                module_path,
                "test",
                p.line_count(),
                &p.functions,
                &p.variables,
                &p.classes,
                Vec::new(),
            );
            for strict in [false, true] {
                let mut want: Vec<&str> = report
                    .symbol_reports
                    .iter()
                    .filter(|s| is_untyped(s.slots(), strict))
                    .map(|s| s.name())
                    .collect();
                want.sort();

                let mut errors = Vec::new();
                collect_untyped_errors(
                    &mut errors,
                    &p.module,
                    &p.functions,
                    &p.variables,
                    strict,
                    None,
                );
                let mut got: Vec<&str> = errors
                    .iter()
                    .map(|e| e.msg_header().split('`').nth(1).unwrap())
                    .collect();
                got.sort();

                assert_eq!(want, got, "check/report desync in {file} (strict={strict})");
            }
        }
    }

    /// gh-4024: under `--strict` a fully-`Any` symbol has no typed slots left,
    /// so it must be reported as coverage-missing, not coverage-partial.
    #[test]
    fn test_untyped_error_kinds() {
        let (p, _) = parse_test_module("any_annotations.py", TestEnv::new());
        let kinds = |strict: bool| {
            let mut errors = Vec::new();
            collect_untyped_errors(
                &mut errors,
                &p.module,
                &p.functions,
                &p.variables,
                strict,
                None,
            );
            errors
                .iter()
                .map(|e| {
                    (
                        e.msg_header().split('`').nth(1).unwrap().to_owned(),
                        e.error_kind(),
                    )
                })
                .collect::<Vec<_>>()
        };
        // Non-strict: every annotated slot counts as covered, so nothing is reported.
        assert_eq!(kinds(false), vec![]);
        // Strict: fully-`Any` symbols are missing; mixed `Any`/concrete is partial.
        assert_eq!(
            kinds(true),
            vec![
                ("test.func_any".to_owned(), ErrorKind::CoverageMissing),
                ("test.func_mixed".to_owned(), ErrorKind::CoveragePartial),
                ("test.x".to_owned(), ErrorKind::CoverageMissing),
            ]
        );
    }

    #[test]
    fn test_slot_counts() {
        // Verify slot classification
        let typed = SlotCounts::typed();
        assert_eq!(typed.n_typable, 1);
        assert_eq!(typed.n_typed, 1);
        assert_eq!(typed.n_any, 0);
        assert_eq!(typed.n_untyped, 0);

        let any = SlotCounts::any();
        assert_eq!(any.n_typable, 1);
        assert_eq!(any.n_typed, 0);
        assert_eq!(any.n_any, 1);
        assert_eq!(any.n_untyped, 0);

        let untyped = SlotCounts::untyped();
        assert_eq!(untyped.n_typable, 1);
        assert_eq!(untyped.n_typed, 0);
        assert_eq!(untyped.n_any, 0);
        assert_eq!(untyped.n_untyped, 1);

        // Merge
        let merged = typed.merge(any).merge(untyped);
        assert_eq!(merged.n_typable, 3);
        assert_eq!(merged.n_typed, 1);
        assert_eq!(merged.n_any, 1);
        assert_eq!(merged.n_untyped, 1);

        // Coverage
        assert!((merged.coverage() - 66.66666666666667).abs() < 0.001);
        assert!((merged.strict_coverage() - 33.33333333333333).abs() < 0.001);

        // Empty → 100%
        let empty = SlotCounts::default();
        assert_eq!(empty.coverage(), 100.0);
        assert_eq!(empty.strict_coverage(), 100.0);
    }

    #[test]
    fn test_report_any_detection() {
        let report = build_module_report_for_test("any_detection.py");
        compare_snapshot("any_detection.expected.json", &report);
    }

    #[test]
    fn test_report_multi_tool_suppressions() {
        let report = build_module_report_for_test("multi_tool_suppressions.py");
        compare_snapshot("multi_tool_suppressions.expected.json", &report);
    }

    #[test]
    fn test_is_fully_annotated() {
        /// Helper to create a Function for testing annotation completeness.
        fn make_function(name: &str, has_return: bool, params: Vec<(&str, bool)>) -> Function {
            Function {
                name: name.to_owned(),
                return_annotation: if has_return {
                    Some("int".to_owned())
                } else {
                    None
                },
                return_rank: SlotRank::classify(has_return, has_return),
                parameters: params
                    .into_iter()
                    .enumerate()
                    .map(|(i, (param_name, annotated))| {
                        let is_self =
                            i == 0 && (matches!(param_name, "self" | "cls") || name == "__new__");
                        Parameter {
                            name: param_name.to_owned(),
                            annotation: if annotated {
                                Some("str".to_owned())
                            } else {
                                None
                            },
                            is_type_known: annotated,
                            merge_key: if is_self {
                                None
                            } else {
                                Some(ParamKey::Named(param_name.to_owned()))
                            },
                            location: Location { line: 1, column: 1 },
                        }
                    })
                    .collect(),
                is_type_known: false, // Not relevant for annotation-only tests
                property_role: None,
                n_params: 0,
                slots: SlotCounts::default(),
                location: Location { line: 1, column: 1 },
                range: TextRange::default(),
            }
        }

        /// Name-based approximation of `has_implicit_receiver` for test convenience.
        fn is_fully_annotated(function: &Function) -> bool {
            if function.return_annotation.is_none() {
                return false;
            }
            function
                .parameters
                .iter()
                .all(|p| p.merge_key.is_none() || p.annotation.is_some())
        }

        // Fully annotated function
        assert!(is_fully_annotated(&make_function(
            "foo",
            true,
            vec![("x", true)]
        )));

        // self as first param is exempt
        assert!(is_fully_annotated(&make_function(
            "bar",
            true,
            vec![("self", false), ("y", true)]
        )));

        // cls as first param is exempt
        assert!(is_fully_annotated(&make_function(
            "cls_method",
            true,
            vec![("cls", false)]
        )));

        // Missing return annotation
        assert!(!is_fully_annotated(&make_function(
            "no_return",
            false,
            vec![]
        )));

        // Missing parameter annotation
        assert!(!is_fully_annotated(&make_function(
            "missing_param",
            true,
            vec![("x", false)]
        )));

        // "self" as a non-first parameter should NOT be exempt
        assert!(!is_fully_annotated(&make_function(
            "bad_self",
            true,
            vec![("x", true), ("self", false)]
        )));

        // "cls" as a non-first parameter should NOT be exempt
        assert!(!is_fully_annotated(&make_function(
            "bad_cls",
            true,
            vec![("x", true), ("cls", false)]
        )));

        // __new__ first param is exempt regardless of name
        assert!(is_fully_annotated(&make_function(
            "__new__",
            true,
            vec![("_cls", false), ("x", true)]
        )));

        // __new__ with standard "cls" name is also exempt
        assert!(is_fully_annotated(&make_function(
            "__new__",
            true,
            vec![("cls", false), ("x", true)]
        )));
    }

    // ──── Phase 1: Tests that validate existing pyrefly behaviour ────

    /// Any-typed annotations: typed for coverage, untyped for strict_coverage.
    #[test]
    fn test_report_any_annotations() {
        let report = build_module_report_for_test("any_annotations.py");
        compare_snapshot("any_annotations.expected.json", &report);
    }

    /// String annotations ("int") and Annotated unwrapping.
    #[test]
    fn test_report_string_annotations() {
        let report = build_module_report_for_test("string_annotations.py");
        compare_snapshot("string_annotations.expected.json", &report);
    }

    // ──── Phase 2: TDD baseline tests (ported from typestats) ────
    //
    // These tests capture pyrefly's CURRENT behaviour for scenarios from typestats.
    // Snapshots may need updating when later diffs improve property/overload/schema handling.

    /// @property getter/setter/deleter merged into one report per property.
    #[test]
    fn test_report_property_basic() {
        let report = build_module_report_for_test("property_basic.py");
        compare_snapshot("property_basic.expected.json", &report);
    }

    /// @overload decorated functions and methods.
    ///
    /// Only the @overload signatures are reported; the implementation
    /// signature is excluded because it is not part of the public API.
    #[test]
    fn test_report_overloads() {
        let report = build_module_report_for_test("overloads.py");
        compare_snapshot("overloads.expected.json", &report);
    }

    /// @overload merging: partial annotations, different param counts, non-overloaded.
    #[test]
    fn test_report_overloads_partial() {
        let report = build_module_report_for_test("overloads_partial.py");
        compare_snapshot("overloads_partial.expected.json", &report);
    }

    /// @overload merging with a fallback returning `Any`.
    /// Best-wins semantics mean the merged return is `Typed`, not `Any`.
    /// Regression test for https://github.com/facebook/pyrefly/issues/3257.
    #[test]
    fn test_report_overloads_any_fallback() {
        let report = build_module_report_for_test("overloads_any_fallback.py");
        compare_snapshot("overloads_any_fallback.expected.json", &report);
    }

    /// @dataclass, Enum, TypedDict, NamedTuple: schema class fields.
    /// Current: schema class fields are reported as regular variables.
    /// Typestats: schema fields are IMPLICIT (0 typable).
    #[test]
    fn test_report_schema_classes() {
        let report = build_module_report_for_test("schema_classes.py");
        compare_snapshot("schema_classes.expected.json", &report);
    }

    /// Instance attributes: self.x in __init__.
    /// Current: instance attrs are not reported (only class-body and top-level).
    /// Typestats: self.x in __init__ is collected as a class member.
    #[test]
    fn test_report_instance_attrs() {
        let report = build_module_report_for_test("instance_attrs.py");
        compare_snapshot("instance_attrs.expected.json", &report);
    }

    /// Dunder methods with implicit return types (__init__, __bool__, __len__, etc.)
    /// should have their return slot excluded from coverage counting.
    #[test]
    fn test_report_dunder_implicit() {
        let report = build_module_report_for_test("dunder_implicit.py");
        compare_snapshot("dunder_implicit.expected.json", &report);
    }

    /// Dunder methods with implicit parameter types (__exit__ exception triple,
    /// __getattr__ name, __setattr__ name). Unannotated implicit params get 0 slots;
    /// explicit annotations on implicit params count normally.
    #[test]
    fn test_report_dunder_params() {
        let report = build_module_report_for_test("dunder_params.py");
        compare_snapshot("dunder_params.expected.json", &report);
    }

    /// Class-body dunder attrs (__slots__, __doc__, __module__, etc.) are
    /// excluded from coverage counting — their types are implicit.
    #[test]
    fn test_report_dunder_attrs() {
        let report = build_module_report_for_test("dunder_attrs.py");
        compare_snapshot("dunder_attrs.expected.json", &report);
    }

    /// Protocol classes define structural interfaces. The class itself should
    /// have n_typable=0, while its methods still count toward coverage.
    #[test]
    fn test_report_protocol() {
        let report = build_module_report_for_test("protocol.py");
        compare_snapshot("protocol.expected.json", &report);
    }

    /// @staticmethod, @classmethod decorator handling.
    /// Current: decorators are reported as regular methods.
    #[test]
    fn test_report_decorators() {
        let report = build_module_report_for_test("decorators.py");
        compare_snapshot("decorators.expected.json", &report);
    }

    /// @type_check_only exclusion: decorated functions and classes are
    /// entirely excluded from the report.
    #[test]
    fn test_report_type_check_only() {
        let report = build_module_report_for_test("type_check_only.py");
        compare_snapshot("type_check_only.expected.json", &report);
    }

    /// Class method aliases: __rand__ = __and__.
    /// Current: aliases are not reported (only the original method def).
    /// Typestats: method aliases are copied as Function symbols.
    #[test]
    fn test_report_method_aliases() {
        let report = build_module_report_for_test("method_aliases.py");
        compare_snapshot("method_aliases.expected.json", &report);
    }

    /// Non-public names and excluded module dunders are filtered from the report.
    #[test]
    fn test_report_private_filtering() {
        let report = build_module_report_for_test("private_filtering.py");
        compare_snapshot("private_filtering.expected.json", &report);
    }

    /// Leading-underscore names listed in `__all__` are part of the public API (issue #3578).
    #[test]
    fn test_report_private_in_all() {
        let report = build_module_report_for_test("private_in_all.py");
        compare_snapshot("private_in_all.expected.json", &report);
    }

    /// PEP 562 module hooks written as `def` are excluded from the report (issue #4020).
    #[test]
    fn test_report_module_dunder_hooks() {
        let report = build_module_report_for_test("module_dunder_hooks.py");
        compare_snapshot("module_dunder_hooks.expected.json", &report);
    }

    /// CPython-injected module globals are excluded even when control flow
    /// wraps them in Phi bindings (issue #3505).
    #[test]
    fn test_report_implicit_module_globals() {
        let report = build_module_report_for_test("implicit_module_globals.py");
        compare_snapshot("implicit_module_globals.expected.json", &report);
    }

    /// Toplevel names removed by `del` must not appear in the report (issue #3576).
    #[test]
    fn test_report_del_module_level() {
        let report = build_module_report_for_test("del_module_level.py");
        compare_snapshot("del_module_level.expected.json", &report);
    }

    /// Methods and instance attrs of a `del`eted class — and its whole nested
    /// subtree — must not appear either (issue #4021).
    #[test]
    fn test_report_del_class() {
        let report = build_module_report_for_test("del_class.py");
        compare_snapshot("del_class.expected.json", &report);
    }

    /// --module name override: symbol counts (n_functions vs n_methods) must
    /// be correct even when the output module name differs from the derived name.
    #[test]
    fn test_report_module_name_override() {
        let report = build_module_report_with_override("functions.py", "my.package.module");
        compare_snapshot("module_name_override.expected.json", &report);
    }

    /// Inherited attrs should only appear under the defining class, not subclasses.
    #[test]
    fn test_report_inherited_attrs() {
        let report = build_module_report_for_test("inherited_attrs.py");
        compare_snapshot("inherited_attrs.expected.json", &report);
    }

    /// `list[Any]` etc. count as typed; only bare `Any` is "any".
    #[test]
    fn test_report_partial_any() {
        let report = build_module_report_for_test("partial_any.py");
        compare_snapshot("partial_any.expected.json", &report);
    }

    /// Bare `Final` should be typed, not `Any`
    #[test]
    fn test_report_bare_final() {
        let report = build_module_report_for_test("bare_final.py");
        let attr_slots = |name: &str| {
            report
                .symbol_reports
                .iter()
                .find_map(|s| match s {
                    SymbolReport::Attr { name: n, slots, .. } if n == name => Some(*slots),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("no attr symbol named {name}"))
        };

        for name in [
            "test.golden",
            "test.golden_ratio",
            "test.pi",
            "test.name",
            "test.Constants.rate",
            "test.Constants.count",
        ] {
            let slots = attr_slots(name);
            assert_eq!(slots.n_typable, 1, "{name} should have 1 typable slot");
            assert_eq!(slots.n_typed, 1, "{name} should be typed");
            assert_eq!(slots.n_any, 0, "{name} should not be any");
        }
    }

    #[test]
    fn test_report_bare_list_annotations() {
        let report = build_module_report_for_test("bare_list_annotations.py");
        let function_slots = report
            .symbol_reports
            .iter()
            .find_map(|s| match s {
                SymbolReport::Function { name, slots, .. } if name == "test.f" => Some(*slots),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no function symbol named test.f"));

        assert_eq!(function_slots.n_typable, 2);
        assert_eq!(function_slots.n_typed, 2);
        assert_eq!(function_slots.n_any, 0);
        assert_eq!(function_slots.n_untyped, 0);
    }

    /// Type aliases (explicit `TypeAlias`, bare assignments, PEP 695, TypeAliasType)
    /// are type-level constructs and should have 0 typable slots.
    #[test]
    fn test_report_type_aliases() {
        let report = build_module_report_for_test("type_aliases.py");
        compare_snapshot("type_aliases.expected.json", &report);
    }

    /// @no_type_check functions should be excluded from coverage reporting
    /// entirely, since their bodies are not analyzed.
    #[test]
    fn test_report_no_type_check_excluded() {
        let code = r#"
from typing import no_type_check

@no_type_check
def f(x: int):
    pass

def g(x: int) -> int:
    return x
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let symbols = ModuleSymbols::collect(&state.transaction(), &handle, false).unwrap();

        // Only g should be reported; f is excluded due to @no_type_check.
        assert_eq!(symbols.functions.len(), 1);
        assert_eq!(symbols.functions[0].name, "test.g");
    }

    // ──── --public-only tests ────

    #[test]
    fn test_is_public_module() {
        let public = |s| is_public_module(ModuleName::from_str(s));
        assert!(public("pkg"));
        assert!(public("pkg.sub"));
        assert!(!public("pkg._internal"));
        assert!(!public("_pkg"));
        assert!(!public("pkg._internal.sub"));
    }

    #[test]
    fn test_is_fqn_public() {
        let fqns: HashSet<String> = ["pkg.Foo", "pkg.bar", "pkg._explicit"]
            .into_iter()
            .map(String::from)
            .collect();
        let public = |s| is_public_fqn(s, "pkg.", &fqns);
        assert!(public("pkg.Foo"));
        assert!(public("pkg.bar"));
        assert!(public("pkg.Foo.method")); // class member of a public class
        assert!(public("pkg.Foo.Inner.attr"));
        // Private name directly listed (e.g. via __all__) is public.
        assert!(public("pkg._explicit"));
        assert!(!public("other.Foo"));
        assert!(!public("pkg.Baz"));
        assert!(!public("pkg.Baz.method"));
        // Private nested members do not leak through a public parent.
        assert!(!public("pkg.Foo._private"));
        assert!(!public("pkg.Foo._Inner.attr"));
        assert!(!public("pkg._Hidden.Foo"));
    }

    #[test]
    fn test_filter_module_report_to_public() {
        let loc = |line| Location { line, column: 1 };
        let mut report = ModuleReport {
            name: "pkg".to_owned(),
            path: "pkg/__init__.py".to_owned(),
            names: vec!["pkg.Foo".into(), "pkg.bar".into(), "pkg._private".into()],
            line_count: 10,
            symbol_reports: vec![
                SymbolReport::Class {
                    name: "pkg.Foo".into(),
                    slots: SlotCounts::default(),
                    location: loc(1),
                },
                SymbolReport::Function {
                    name: "pkg.Foo.method".into(),
                    slots: SlotCounts::typed(),
                    n_params: 2,
                    location: loc(2),
                },
                SymbolReport::Function {
                    name: "pkg.bar".into(),
                    slots: SlotCounts::untyped(),
                    n_params: 1,
                    location: loc(3),
                },
                SymbolReport::Attr {
                    name: "pkg._private".into(),
                    slots: SlotCounts::typed(),
                    location: loc(4),
                },
            ],
            type_ignores: vec![],
            slots: SlotCounts::default(),
            coverage: 100.0,
            strict_coverage: 100.0,
            symbols: SymbolCounts {
                n_functions: 2,
                n_methods: 0,
                n_function_params: 123,
                n_method_params: 456,
                n_classes: 1,
                n_attrs: 1,
                n_properties: 0,
                n_type_ignores: 0,
            },
        };

        let public_fqns: HashSet<String> = ["pkg.Foo", "pkg.bar"]
            .into_iter()
            .map(String::from)
            .collect();
        filter_module_report_to_public(&mut report, &public_fqns);

        assert_eq!(report.names, vec!["pkg.Foo", "pkg.bar"]);
        assert_eq!(report.symbol_reports.len(), 3); // Foo, Foo.method, bar
        assert_eq!(report.symbols.n_functions, 1);
        assert_eq!(report.symbols.n_methods, 1);
        assert_eq!(report.symbols.n_function_params, 1);
        assert_eq!(report.symbols.n_method_params, 2);
        assert_eq!(report.symbols.n_classes, 1);
        assert_eq!(report.symbols.n_attrs, 0);
    }

    /// A config-less file whose path can't form a valid module name falls back to
    /// `__unknown__`. Its symbol FQNs must carry the `__unknown__.` prefix, or
    /// `--public-only` filtering (which rebuilds that prefix from the report name)
    /// drops every symbol and reports "0 of N typable".
    #[test]
    fn test_unknown_module_public_only_keeps_symbols() {
        let code = "def bar() -> int:\n    return 1\n";
        let (state, handle_fn) = TestEnv::one_with_path("__unknown__", "__unknown__.py", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("__unknown__");
        assert_eq!(handle.module(), ModuleName::unknown());

        let transaction = state.transaction();
        let symbols = ModuleSymbols::collect(&transaction, &handle, false).unwrap();
        // The bug lives in symbol construction: FQNs must be prefixed with the module name.
        assert_eq!(symbols.functions[0].name, "__unknown__.bar");

        let (public_fqns, _) = compute_public_fqns(std::slice::from_ref(&handle), &transaction);
        let mut report = build_module_report(
            "__unknown__".to_owned(),
            "__unknown__.py".to_owned(),
            "__unknown__",
            symbols.line_count(),
            &symbols.functions,
            &symbols.variables,
            &symbols.classes,
            symbols.suppressions,
        );
        filter_module_report_to_public(&mut report, &public_fqns);

        assert!(
            report.names.iter().any(|n| n == "__unknown__.bar"),
            "public symbols in an __unknown__ module must survive --public-only filtering"
        );
    }

    /// The `check`/untyped-error path (`collect_untyped_errors`) is the *other*
    /// `--public-only` consumer: it rebuilds the `__unknown__.` prefix and matches
    /// it against symbol FQNs. With bare FQNs an untyped public symbol in an
    /// `__unknown__` module is silently never flagged, so the fix must reach this
    /// path too — not just the report filter above.
    #[test]
    fn test_unknown_module_public_only_flags_untyped() {
        let code = "def bar(x):\n    return x\n";
        let (state, handle_fn) = TestEnv::one_with_path("__unknown__", "__unknown__.py", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("__unknown__");
        let transaction = state.transaction();
        let symbols = ModuleSymbols::collect(&transaction, &handle, false).unwrap();

        let (public_fqns, _) = compute_public_fqns(std::slice::from_ref(&handle), &transaction);
        let mut errors = Vec::new();
        collect_untyped_errors(
            &mut errors,
            &symbols.module,
            &symbols.functions,
            &symbols.variables,
            false,
            Some(&public_fqns),
        );

        assert!(
            !errors.is_empty(),
            "an untyped public symbol in an __unknown__ module must be flagged under --public-only"
        );
    }

    #[test]
    fn test_compute_public_fqns() {
        let compute = |modules: &[(&str, &str, &str)],
                       handle_names: &[&str]|
         -> (HashSet<String>, HashSet<String>) {
            let mut env = TestEnv::new();
            for &(name, path, source) in modules {
                env.add_with_path(name, path, source);
            }
            let env = env.with_default_require_level(Require::Everything);
            let (state, handle_fn) = env.to_state();
            let transaction = state.transaction();
            let handles: Vec<_> = handle_names.iter().map(|n| handle_fn(n)).collect();
            let (fqns, origins) = compute_public_fqns(&handles, &transaction);
            let origins = origins.iter().map(|h| h.module().to_string()).collect();
            (fqns, origins)
        };

        // Re-export from a private module keeps both the local alias and the
        // traced origin, so reports covering either module stay non-empty.
        let (fqns, origins) = compute(
            &[
                (
                    "pkg",
                    "pkg/__init__.py",
                    "from pkg._internal import Foo\n__all__ = [\"Foo\"]\n",
                ),
                (
                    "pkg._internal",
                    "pkg/_internal.py",
                    "class Foo:\n    x: int = 1\n",
                ),
            ],
            &["pkg", "pkg._internal"],
        );
        assert!(fqns.contains("pkg.Foo"));
        assert!(fqns.contains("pkg._internal.Foo"));
        assert_eq!(origins, HashSet::from(["pkg._internal".to_owned()]));

        // Without __all__, non-underscore local names are exported
        let (fqns, _) = compute(
            &[(
                "pkg",
                "pkg/__init__.py",
                "def foo() -> int: ...\ndef _private(): ...\n",
            )],
            &["pkg"],
        );
        assert!(fqns.contains("pkg.foo"));
        assert!(!fqns.contains("pkg._private"));

        // Regular imports (not `import x as x`) are not re-exports
        let (fqns, origins) = compute(
            &[
                (
                    "pkg",
                    "pkg/__init__.py",
                    "from pkg._internal import helper\ndef local_fn() -> int: ...\n",
                ),
                (
                    "pkg._internal",
                    "pkg/_internal.py",
                    "def helper() -> int: ...\n",
                ),
            ],
            &["pkg", "pkg._internal"],
        );
        assert!(fqns.contains("pkg.local_fn"));
        assert!(!fqns.contains("pkg._internal.helper"));
        assert_eq!(origins, HashSet::new());

        // `import x as x` is an implicit re-export when there is no __all__
        let (fqns, _) = compute(
            &[
                (
                    "pkg",
                    "pkg/__init__.py",
                    "from pkg._internal import helper as helper\n",
                ),
                (
                    "pkg._internal",
                    "pkg/_internal.py",
                    "def helper() -> int: ...\n",
                ),
            ],
            &["pkg", "pkg._internal"],
        );
        assert!(fqns.contains("pkg._internal.helper"));

        // __all__ takes precedence over `import x as x`
        let (fqns, _) = compute(
            &[
                (
                    "pkg",
                    "pkg/__init__.py",
                    concat!(
                        "from pkg._internal import foo as foo\n",
                        "from pkg._internal import bar\n",
                        "baz: int = 1\n",
                        "__all__ = [\"bar\"]\n",
                    ),
                ),
                (
                    "pkg._internal",
                    "pkg/_internal.py",
                    "def foo() -> int: ...\ndef bar() -> int: ...\n",
                ),
            ],
            &["pkg", "pkg._internal"],
        );
        assert!(fqns.contains("pkg._internal.bar"));
        assert!(!fqns.contains("pkg._internal.foo"));
        assert!(!fqns.contains("pkg.baz"));

        // A re-export of a name its origin module doesn't define keeps the
        // local FQN but traces to no origin (github.com/facebook/pyrefly/issues/4025)
        let (fqns, origins) = compute(
            &[
                (
                    "pkg",
                    "pkg/__init__.py",
                    "from pkg._internal import ghost\n__all__ = [\"ghost\"]\n",
                ),
                ("pkg._internal", "pkg/_internal.py", "x: int = 1\n"),
            ],
            &["pkg", "pkg._internal"],
        );
        assert!(fqns.contains("pkg.ghost"));
        assert!(!fqns.contains("pkg._internal.ghost"));
        assert_eq!(origins, HashSet::new());
    }

    /// gh-4034: --project-excludes must not drop symbols of non-excluded public module re-exports.
    #[test]
    fn test_project_excludes_public_reexport() {
        const FILES: [(&str, &str); 2] = [
            (
                "__init__.py",
                "from foo._private import bar\n__all__ = ['bar']",
            ),
            ("_private.py", "def bar(a) -> None: ..."),
        ];

        // Returns foo._private's report, if any (never two).
        let collect_private = |files: &[(&str, &str)], exclude: &str, public_only: bool| {
            let dir = TempDir::new().unwrap();
            let foo = dir.path().join("foo");
            std::fs::create_dir(&foo).unwrap();
            // Without a config, discovered files get unknown module names.
            std::fs::write(dir.path().join("pyrefly.toml"), "search-path = ['.']").unwrap();
            for &(name, source) in files {
                std::fs::write(foo.join(name), source).unwrap();
            }
            let globs = FilteredGlobs::new(
                Globs::new(vec![foo.display().to_string()]).unwrap(),
                Globs::new(vec![foo.join(exclude).display().to_string()]).unwrap(),
                None,
                HiddenDirFilter::Disabled,
            );
            let (reports, _) = collect_module_reports(
                Box::new(globs),
                default_config_finder(None),
                false,
                None,
                public_only,
                None,
                TEST_THREAD_COUNT,
            )
            .unwrap();
            let mut private: Vec<_> = reports
                .into_iter()
                .filter(|r| r.name == "foo._private")
                .collect();
            assert!(private.len() <= 1);
            private.pop()
        };

        // The excluded module's report keeps exactly its publicly re-exported symbol.
        let private = collect_private(&FILES, FILES[1].0, true).unwrap();
        let names: Vec<&str> = private
            .symbol_reports
            .iter()
            .map(SymbolReport::name)
            .collect();
        assert_eq!(names, vec!["foo._private.bar"]);
        assert_eq!((private.slots.n_typable, private.slots.n_typed), (2, 1));

        // Without a public re-export the excluded module stays excluded.
        let plain_import = [("__init__.py", "from foo._private import bar"), FILES[1]];
        assert!(collect_private(&plain_import, FILES[1].0, true).is_none());

        // Outside --public-only, excludes are respected unconditionally.
        assert!(collect_private(&FILES, FILES[1].0, false).is_none());

        // An excluded .pyi twin resolves as the origin (stubs win) but must not report twice.
        let twins = [FILES[0], FILES[1], ("_private.pyi", FILES[1].1)];
        let private = collect_private(&twins, twins[2].0, true).unwrap();
        assert!(private.path.ends_with(FILES[1].0));
    }

    /// Dataclass and NamedTuple fields are IMPLICIT; methods on those classes count normally.
    #[test]
    fn test_report_schema_classes_methods() {
        let report = build_module_report_for_test("schema_classes_methods.py");
        compare_snapshot("schema_classes_methods.expected.json", &report);
    }

    /// IntEnum, StrEnum, Flag, IntFlag members are IMPLICIT (0 typable).
    #[test]
    fn test_report_schema_classes_enums() {
        let report = build_module_report_for_test("schema_classes_enums.py");
        compare_snapshot("schema_classes_enums.expected.json", &report);
    }

    /// TypedDict and dataclass subclasses: fields of all classes are IMPLICIT.
    /// Inherited fields appear only under the defining class.
    #[test]
    fn test_report_schema_classes_inherited() {
        let report = build_module_report_for_test("schema_classes_inherited.py");
        compare_snapshot("schema_classes_inherited.expected.json", &report);
    }

    /// Pydantic BaseModel fields are IMPLICIT (0 typable).
    #[test]
    fn test_report_schema_classes_pydantic() {
        let path = std::env::var("PYDANTIC_TEST_PATH").expect("PYDANTIC_TEST_PATH must be set");
        let env = TestEnv::new_with_site_package_paths(&[&path]);
        let report = build_module_report_for_test_with_env("schema_classes_pydantic.py", env);
        compare_snapshot("schema_classes_pydantic.expected.json", &report);
    }

    /// attrs @define and @attr.s(auto_attribs=True) fields are IMPLICIT (0 typable).
    #[test]
    fn test_report_schema_classes_attrs() {
        let path = std::env::var("ATTRS_TEST_PATH").expect("ATTRS_TEST_PATH must be set");
        let env = TestEnv::new_with_site_package_paths(&[&path]);
        let report = build_module_report_for_test_with_env("schema_classes_attrs.py", env);
        compare_snapshot("schema_classes_attrs.expected.json", &report);
    }
}
