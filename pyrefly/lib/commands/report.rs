/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use pyrefly_build::handle::Handle;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_graph::index::Idx;
use pyrefly_python::dunder;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::Tool;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::PropertyRole;
use pyrefly_types::class::ClassDefIndex;
use pyrefly_types::types::Type;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::includes::Includes;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_python_ast::Expr;
use ruff_python_ast::Parameters;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::Answers;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingClassField;
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
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::export::exports::ExportLocation;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;

/// `(major, minor)` version for the report JSON schema.
const REPORT_SCHEMA_VERSION: (u32, u32) = (0, 1);

/// Slot-level annotation counts for a symbol.
///
/// A "slot" is a single annotation site: a function return type, a function
/// parameter, or a module-level variable. Each slot falls into exactly one of
/// three buckets: typed (concrete annotation with no `Any`), any (annotation
/// that resolves to or contains `Any`), or untyped (no annotation at all).
/// `n_typable` is always the sum of the other three.
#[derive(Debug, Serialize, Default, Clone, Copy)]
struct SlotCounts {
    /// Total number of annotation sites.
    n_typable: usize,
    /// Sites with a concrete annotation containing no `Any`.
    n_typed: usize,
    /// Sites annotated but whose resolved type contains `Any`.
    n_any: usize,
    /// Sites with no annotation at all.
    n_untyped: usize,
}

impl SlotCounts {
    fn merge(self, other: SlotCounts) -> SlotCounts {
        SlotCounts {
            n_typable: self.n_typable + other.n_typable,
            n_typed: self.n_typed + other.n_typed,
            n_any: self.n_any + other.n_any,
            n_untyped: self.n_untyped + other.n_untyped,
        }
    }

    fn typed() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 1,
            n_any: 0,
            n_untyped: 0,
        }
    }

    fn any() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 0,
            n_any: 1,
            n_untyped: 0,
        }
    }

    fn untyped() -> SlotCounts {
        SlotCounts {
            n_typable: 1,
            n_typed: 0,
            n_any: 0,
            n_untyped: 1,
        }
    }

    /// Coverage: (n_typed + n_any) / n_typable. Treats Any-annotated slots as covered.
    fn coverage(&self) -> f64 {
        if self.n_typable == 0 {
            100.0
        } else {
            ((self.n_typed + self.n_any) as f64 / self.n_typable as f64) * 100.0
        }
    }

    /// Strict coverage: n_typed / n_typable. Only concrete types count.
    fn strict_coverage(&self) -> f64 {
        if self.n_typable == 0 {
            100.0
        } else {
            (self.n_typed as f64 / self.n_typable as f64) * 100.0
        }
    }
}

/// Annotation quality for a single slot, ordered worst-to-best so that
/// `min()` gives "worst-wins" semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SlotRank {
    Untyped,
    Any,
    Typed,
    Skip,
}

impl SlotRank {
    /// Map (has_annotation, is_type_known) to a rank.
    fn classify(has_annotation: bool, is_type_known: bool) -> Self {
        match (has_annotation, is_type_known) {
            (false, _) => SlotRank::Untyped,
            (true, true) => SlotRank::Typed,
            (true, false) => SlotRank::Any,
        }
    }
}

impl From<&Parameter> for SlotRank {
    fn from(param: &Parameter) -> Self {
        SlotRank::classify(param.annotation.is_some(), param.is_type_known)
    }
}

impl From<SlotRank> for SlotCounts {
    fn from(rank: SlotRank) -> Self {
        match rank {
            SlotRank::Typed => SlotCounts::typed(),
            SlotRank::Any => SlotCounts::any(),
            SlotRank::Untyped => SlotCounts::untyped(),
            SlotRank::Skip => SlotCounts::default(),
        }
    }
}

/// Parameter dedup key for overload merging.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParamKey {
    Positional(usize),
    Named(String),
    VarPositional,
    VarKeyword,
}

#[derive(Debug, Copy, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct Location {
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, Serialize)]
/// Information about a single function parameter.
struct Parameter {
    name: String,
    annotation: Option<String>,
    /// Whether the resolved type contains no `Any`.
    is_type_known: bool,
    /// Overload merge key (`None` for self/cls and implicit params).
    #[serde(skip_serializing)]
    merge_key: Option<ParamKey>,
    location: Location,
}

/// Renamed from `Suppression` to avoid collision with `pyrefly_python::ignore::Suppression`.
#[derive(Debug, Serialize)]
struct ReportSuppression {
    /// The suppression tool (e.g. pyrefly, mypy, pyre).
    kind: Tool,
    codes: Vec<String>,
    location: Location,
}

#[derive(Debug, Serialize)]
struct Function {
    name: String,
    return_annotation: Option<String>,
    /// Whether the return type contains no `Any`.
    is_return_type_known: bool,
    parameters: Vec<Parameter>,
    is_type_known: bool,
    /// Property role if this function is a property accessor, `None` otherwise.
    #[serde(skip)]
    property_role: Option<PropertyRole>,
    /// Number of non-self/cls, non-implicit params (for entity counting).
    n_params: usize,
    slots: SlotCounts,
    location: Location,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct IncompleteAttribute {
    name: String,
    declared_in: String,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ReportClass {
    name: String,
    incomplete_attributes: Vec<IncompleteAttribute>,
    location: Location,
}

#[derive(Debug, Serialize)]
struct Variable {
    name: String,
    annotation: Option<String>,
    slots: SlotCounts,
    location: Location,
}

/// Per-symbol report with kind discriminator matching typestats' model.
#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
enum SymbolReport {
    #[serde(rename = "attr")]
    Attr {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
    #[serde(rename = "class")]
    Class {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
    #[serde(rename = "property")]
    Property {
        name: String,
        #[serde(flatten)]
        slots: SlotCounts,
        location: Location,
    },
}

#[derive(Debug, Serialize)]
struct ModuleReport {
    /// Fully-qualified module name (e.g. "mypackage.submodule").
    name: String,
    /// Names of symbols defined in this module.
    names: Vec<String>,
    line_count: usize,
    symbol_reports: Vec<SymbolReport>,
    type_ignores: Vec<ReportSuppression>,
    #[serde(flatten)]
    slots: SlotCounts,
    coverage: f64,
    strict_coverage: f64,
    n_functions: usize,
    n_methods: usize,
    n_function_params: usize,
    n_method_params: usize,
    n_classes: usize,
    n_attrs: usize,
    n_properties: usize,
    n_type_ignores: usize,
}

#[derive(Debug, Serialize)]
struct ReportSummary {
    n_modules: usize,
    #[serde(flatten)]
    slots: SlotCounts,
    coverage: f64,
    strict_coverage: f64,
    n_functions: usize,
    n_methods: usize,
    n_function_params: usize,
    n_method_params: usize,
    n_classes: usize,
    n_attrs: usize,
    n_properties: usize,
    n_type_ignores: usize,
}

#[derive(Debug, Serialize)]
struct FullReport {
    /// `"{major}.{minor}"` version for this report format.
    schema_version: String,
    module_reports: Vec<ModuleReport>,
    summary: ReportSummary,
}

/// Generate reports from pyrefly type checking results.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ReportArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// When enabled, `.py` files are skipped if a corresponding `.pyi`
    /// file is also present in the set of files to check.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    prefer_stubs: bool,

    /// Override the module name in the report output. When set, all modules
    /// use this name instead of the name derived from the file path. Useful
    /// when reporting on a single module whose canonical package name differs
    /// from its filesystem layout.
    #[clap(long)]
    module: Option<String>,
}

impl ReportArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override, wrapper)?;
        Self::run_inner(
            files_to_check,
            config_finder,
            self.prefer_stubs,
            self.module,
            thread_count,
        )
    }

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

    /// Parse type-ignore suppressions from source using the multi-tool parser from `ignore.rs`.
    fn parse_suppressions(module: &Module) -> Vec<ReportSuppression> {
        let source = module.lined_buffer().contents();
        let ignore = Ignore::new(source);
        let mut suppressions = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (_line_number, supps) in ignore.iter() {
            for supp in supps {
                let comment_line_num = supp.comment_line().get() as usize;
                let column = comment_line_num
                    .checked_sub(1)
                    .and_then(|idx| lines.get(idx))
                    .and_then(|line| line.find('#'))
                    .map(|c| c + 1);
                let Some(column) = column else {
                    continue;
                };

                suppressions.push(ReportSuppression {
                    kind: supp.tool(),
                    codes: supp.error_codes().to_vec(),
                    location: Location {
                        line: comment_line_num,
                        column,
                    },
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

    /// Build a class's qualified name from its nesting context.
    /// Returns e.g. `"Outer.Inner"` for a nested class or `"MyClass"` for a top-level one.
    fn class_qualified_name(
        module: &Module,
        parent: &NestingContext,
        class_name: impl std::fmt::Display,
    ) -> String {
        let parent_path = module.display(parent).to_string();
        if parent_path.is_empty() {
            class_name.to_string()
        } else {
            format!("{parent_path}.{class_name}")
        }
    }

    /// Merge overloads with the same qualified name into one entry,
    /// keeping the worst annotation quality per deduplicated slot.
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
                let short_name = func.name.rsplit('.').next().unwrap_or(&func.name);
                let has_annotation = func.return_annotation.is_some();
                let ret = if !has_annotation && Self::is_implicit_dunder_return(short_name) {
                    SlotRank::Skip
                } else {
                    SlotRank::classify(has_annotation, func.is_return_type_known)
                };
                return_rank = return_rank.min(ret);

                for param in &func.parameters {
                    if let Some(key) = &param.merge_key {
                        let entry = param_slots.entry(key.clone()).or_insert(SlotRank::Skip);
                        *entry = (*entry).min(param.into());
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

    /// Returns true if the name is public: does not start with `_`, or is a dunder (`__x__`).
    /// Matches typestats `is_public_name`.
    fn is_public_name(name: &str) -> bool {
        !name.starts_with('_') || name.ends_with("__")
    }

    /// Module-level dunders that typestats always excludes from the report.
    const EXCLUDED_MODULE_DUNDERS: &'static [&'static str] =
        &["__all__", "__dir__", "__doc__", "__getattr__"];

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

    /// Returns the ClassBinding if the class owning this field is a schema class (dataclass, enum,
    /// TypedDict, or NamedTuple). Fields of schema classes are IMPLICIT — they have
    /// 0 typable slots because the class definition itself governs their types.
    fn get_cls_binding_if_schema_class_field<'a>(
        bindings: &'a Bindings,
        answers: &Answers,
        field: &BindingClassField,
    ) -> Option<&'a ClassBinding> {
        let cls_binding = match bindings.get(field.class_idx) {
            BindingClass::ClassDef(cls) => cls,
            BindingClass::FunctionalClassDef(..) => return None,
        };
        let metadata_key = KeyClassMetadata(cls_binding.def_index);
        let metadata = answers.get_idx(bindings.key_to_idx(&metadata_key))?;
        if metadata.dataclass_metadata().is_some()
            || metadata.is_enum()
            || metadata.is_typed_dict()
            || metadata.named_tuple_metadata().is_some()
        {
            Some(cls_binding)
        } else {
            None
        }
    }

    fn parse_variables(
        module: &Module,
        bindings: &Bindings,
        answers: &Answers,
        exports: &SmallMap<Name, ExportLocation>,
        functions: &[Function],
        classes: &[ReportClass],
    ) -> Vec<Variable> {
        let module_prefix = if module.name() != ModuleName::unknown() {
            format!("{}.", module.name())
        } else {
            String::new()
        };
        // Collect names already reported as functions or classes so we can skip them.
        let reported_names: SmallSet<&str> = functions
            .iter()
            .map(|f| f.name.as_str())
            .chain(classes.iter().map(|c| c.name.as_str()))
            .collect();

        let mut variables = Vec::new();
        for idx in bindings.keys::<KeyExport>() {
            let KeyExport(name) = bindings.idx_to_key(idx);
            // Skip non-public module-level names and excluded dunders.
            let name_str = name.as_str();
            if !Self::is_public_name(name_str) || Self::EXCLUDED_MODULE_DUNDERS.contains(&name_str)
            {
                continue;
            }
            let qualified_name = format!("{module_prefix}{name}");
            if reported_names.contains(qualified_name.as_str()) {
                continue;
            }
            let binding = bindings.get(idx);
            let location = match exports.get(name) {
                Some(ExportLocation::ThisModule(export)) => {
                    Self::range_to_location(module, export.location)
                }
                _ => continue,
            };
            match binding {
                BindingExport::AnnotatedForward(annot_idx, key_idx) => {
                    // IMPLICIT: type aliases are type-level constructs with 0 slots.
                    if matches!(
                        bindings.get(*key_idx),
                        Binding::TypeAlias(_) | Binding::TypeAliasRef(_)
                    ) {
                        variables.push(Variable {
                            name: qualified_name,
                            annotation: None,
                            slots: SlotCounts::default(),
                            location,
                        });
                        continue;
                    }
                    let annotation_text = match bindings.get(*annot_idx) {
                        BindingAnnotation::AnnotateExpr(_, expr, _) => {
                            Some(module.code_at(expr.range()).to_owned())
                        }
                        _ => None,
                    };
                    let is_type_known = annotation_text.is_some()
                        && answers
                            .get_idx(*annot_idx)
                            .and_then(|awt| awt.annotation.ty.as_ref().map(Self::is_type_known))
                            .unwrap_or(false);
                    let slots = SlotRank::classify(annotation_text.is_some(), is_type_known).into();
                    variables.push(Variable {
                        name: qualified_name,
                        annotation: annotation_text,
                        slots,
                        location,
                    });
                }
                BindingExport::Forward(idx) | BindingExport::PromoteForward(idx) => {
                    match bindings.get(*idx) {
                        // Skip injected implicit globals
                        Binding::Global(_) => {}
                        // IMPLICIT: special type forms and type aliases have 0 slots
                        Binding::TypeVar(_)
                        | Binding::ParamSpec(_)
                        | Binding::TypeVarTuple(_)
                        | Binding::TypeAlias(_)
                        | Binding::TypeAliasRef(_) => {
                            variables.push(Variable {
                                name: qualified_name,
                                annotation: None,
                                slots: SlotCounts::default(),
                                location,
                            });
                        }
                        // Functions and classes are handled by parse_functions/parse_classes;
                        // skip them here even when excluded (e.g. @type_check_only).
                        Binding::Function(..) | Binding::ClassDef(..) => {}
                        // IMPLICIT: non-call assignments have 0 slots;
                        // call assignments are untyped (1 slot)
                        Binding::NameAssign(na) => {
                            let slots = if matches!(na.expr.as_ref(), Expr::Call(_)) {
                                SlotCounts::untyped()
                            } else {
                                SlotCounts::default()
                            };
                            variables.push(Variable {
                                name: qualified_name,
                                annotation: None,
                                slots,
                                location,
                            });
                        }
                        _ => {
                            variables.push(Variable {
                                name: qualified_name,
                                annotation: None,
                                slots: SlotCounts::untyped(),
                                location,
                            });
                        }
                    }
                }
            }
        }
        variables.sort_by(|a, b| a.location.cmp(&b.location));
        variables
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
        tco_classes: &SmallSet<Idx<KeyClass>>,
    ) -> Vec<Variable> {
        let mut attrs = Vec::new();
        let module_prefix = if module.name() != ModuleName::unknown() {
            format!("{}.", module.name())
        } else {
            String::new()
        };

        for field_idx in bindings.keys::<KeyClassField>() {
            let field = bindings.get(field_idx);

            // Skip private class attrs (single-underscore prefix).
            if !Self::is_public_name(field.name.as_str()) {
                continue;
            }

            // Skip class-body dunder attrs with implicit types (__slots__, __doc__, etc.)
            if Self::is_implicit_dunder_attr(field.name.as_str()) {
                continue;
            }

            // Skip attrs of @type_check_only classes.
            if tco_classes.contains(&field.class_idx) {
                continue;
            }

            // Only count instance attrs from recognized methods (__init__, etc.)
            // or a schema class body field. We handle recognized-method fields with full
            // slot classification, and schema body fields as IMPLICIT (0 typable).
            let annotation_idx = match &field.definition {
                ClassFieldDefinition::DefinedInMethod {
                    annotation, method, ..
                } => {
                    if !method.recognized_attribute_defining_method {
                        continue;
                    }
                    *annotation
                }
                ClassFieldDefinition::DeclaredByAnnotation {
                    annotation,
                    initialized_in_recognized_method,
                } => {
                    if !initialized_in_recognized_method {
                        // Check if this is a schema class body field
                        if let Some(cls_binding) =
                            Self::get_cls_binding_if_schema_class_field(bindings, answers, field)
                        {
                            if Self::has_function_ancestor(&cls_binding.parent) {
                                continue;
                            }
                            let class_name = Self::class_qualified_name(
                                module,
                                &cls_binding.parent,
                                &cls_binding.def.name,
                            );
                            let qualified_name =
                                format!("{}{}.{}", module_prefix, class_name, field.name);
                            let location = Self::range_to_location(module, field.range);
                            attrs.push(Variable {
                                name: qualified_name,
                                annotation: None,
                                slots: SlotCounts::default(),
                                location,
                            });
                        }
                        continue;
                    }
                    Some(*annotation)
                }
                ClassFieldDefinition::AssignedInBody { .. } => {
                    // Check if this is an enum member or other schema class body assignment
                    if let Some(cls_binding) =
                        Self::get_cls_binding_if_schema_class_field(bindings, answers, field)
                    {
                        if Self::has_function_ancestor(&cls_binding.parent) {
                            continue;
                        }
                        let class_name = Self::class_qualified_name(
                            module,
                            &cls_binding.parent,
                            &cls_binding.def.name,
                        );
                        let qualified_name =
                            format!("{}{}.{}", module_prefix, class_name, field.name);
                        let location = Self::range_to_location(module, field.range);
                        attrs.push(Variable {
                            name: qualified_name,
                            annotation: None,
                            slots: SlotCounts::default(),
                            location,
                        });
                    }
                    continue;
                }
                _ => continue,
            };

            let cls_binding = match bindings.get(field.class_idx) {
                BindingClass::ClassDef(cls) => cls,
                BindingClass::FunctionalClassDef(..) => continue,
            };
            if Self::has_function_ancestor(&cls_binding.parent) {
                continue;
            }
            let class_name =
                Self::class_qualified_name(module, &cls_binding.parent, &cls_binding.def.name);
            let qualified_name = format!("{}{}.{}", module_prefix, class_name, field.name);
            let location = Self::range_to_location(module, field.range);

            let annotation_text = annotation_idx.and_then(|idx| match bindings.get(idx) {
                BindingAnnotation::AnnotateExpr(_, expr, _) => {
                    Some(module.code_at(expr.range()).to_owned())
                }
                _ => None,
            });
            let is_type_known = annotation_text.is_some()
                && annotation_idx
                    .and_then(|idx| {
                        answers
                            .get_idx(idx)
                            .and_then(|awt| awt.annotation.ty.as_ref().map(Self::is_type_known))
                    })
                    .unwrap_or(false);
            let slots = SlotRank::classify(annotation_text.is_some(), is_type_known).into();

            attrs.push(Variable {
                name: qualified_name,
                annotation: annotation_text,
                slots,
                location,
            });
        }
        attrs.sort_by(|a, b| a.location.cmp(&b.location));
        attrs
    }

    fn parse_functions(
        module: &Module,
        bindings: &Bindings,
        answers: &Answers,
        exports: &SmallMap<Name, ExportLocation>,
        tco_classes: &SmallSet<Idx<KeyClass>>,
    ) -> Vec<Function> {
        let mut functions = Vec::new();
        let module_prefix = if module.name() != ModuleName::unknown() {
            format!("{}.", module.name())
        } else {
            String::new()
        };

        for idx in bindings.keys::<Key>() {
            if let Key::Definition(id) = bindings.idx_to_key(idx)
                && let Binding::Function(x, _pred, _class_meta) = bindings.get(idx)
            {
                let decorated = bindings.get(*x);
                let fun = bindings.get(decorated.undecorated_idx);
                // Skip @type_check_only decorated functions.
                if Self::has_type_check_only_decorator(&fun.decorators, bindings) {
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
                let location = Self::range_to_location(module, fun.def.range);
                let func_name = if let Some(class_key) = fun.class_key {
                    match bindings.get(class_key) {
                        BindingClass::ClassDef(cls) => {
                            // Skip methods of classes nested inside functions
                            if Self::has_function_ancestor(&cls.parent) {
                                continue;
                            }
                            // Skip private class methods (single-underscore prefix).
                            if !Self::is_public_name(fun.def.name.as_str()) {
                                continue;
                            }
                            let class_qname =
                                Self::class_qualified_name(module, &cls.parent, &cls.def.name);
                            format!("{module_prefix}{class_qname}.{}", fun.def.name)
                        }
                        BindingClass::FunctionalClassDef(..) => {
                            continue;
                        }
                    }
                } else {
                    // Skip functions not present in the module's exports
                    // (e.g. functions nested inside other functions).
                    if !exports.contains_key(&fun.def.name.id) {
                        continue;
                    }
                    // Skip non-public module-level functions.
                    if !Self::is_public_name(fun.def.name.as_str()) {
                        continue;
                    }
                    format!("{}{}", module_prefix, fun.def.name)
                };

                // Get return annotation text and check if return type is known
                let return_key = Key::ReturnType(*id);
                let return_idx = bindings.key_to_idx(&return_key);
                let return_annotation = if let Binding::ReturnType(ret) = bindings.get(return_idx) {
                    match &ret.kind {
                        ReturnTypeKind::ShouldValidateAnnotation { range, .. }
                        | ReturnTypeKind::ShouldTrustAnnotation { range, .. } => {
                            Some(module.code_at(*range).to_owned())
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                let is_return_type_known = return_annotation.is_some()
                    && answers
                        .get_type_at(return_idx)
                        .is_some_and(|t| Self::is_type_known(&t));

                let mut parameters = Vec::new();
                let implicit_receiver =
                    Self::has_implicit_receiver(fun, answers, decorated.undecorated_idx);
                let all_params = Self::params_with_keys(&fun.def.parameters, implicit_receiver);
                let mut all_params_type_known = true;

                let property_role = answers
                    .get_type_at(idx)
                    .and_then(|t| t.property_metadata().map(|m| m.role.clone()));
                let is_property_deleter =
                    matches!(property_role, Some(PropertyRole::DeleterDecorator));

                // Implicit dunder returns (e.g. __init__ → None) are always
                // excluded from coverage, even when explicitly annotated.
                //
                // Property setters/deleters have a trivial `-> None` return that
                // is not a meaningful typable, so skip it like implicit returns.
                let skip_return = is_property_deleter
                    || matches!(
                        property_role,
                        Some(PropertyRole::Setter | PropertyRole::SetterDecorator)
                    )
                    || (fun.class_key.is_some()
                        && Self::is_implicit_dunder_return(fun.def.name.as_str()));
                let return_slot = if skip_return {
                    SlotCounts::default()
                } else {
                    SlotRank::classify(return_annotation.is_some(), is_return_type_known).into()
                };
                let mut func_slots = return_slot;
                let mut n_params = 0usize;
                let mut non_self_index = 0usize;

                for (merge_key, param) in &all_params {
                    let param_name = param.name.as_str();
                    let is_self = merge_key.is_none();
                    let param_annotation = param
                        .annotation
                        .as_ref()
                        .map(|ann| module.code_at(ann.range()).to_owned());

                    let is_param_type_known = if is_self {
                        true
                    } else if param.annotation.is_some() {
                        let annot_key =
                            KeyAnnotation::Annotation(ShortIdentifier::new(&param.name));
                        let annot_idx = bindings.key_to_idx(&annot_key);
                        answers
                            .get_idx(annot_idx)
                            .and_then(|awt| awt.annotation.ty.as_ref().map(Self::is_type_known))
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    // Implicit dunder params are always excluded, even when annotated.
                    let is_implicit_param = !is_self
                        && fun.class_key.is_some()
                        && Self::is_implicit_dunder_param(fun.def.name.as_str(), non_self_index);

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
                            SlotRank::classify(param_annotation.is_some(), is_param_type_known)
                                .into(),
                        );
                        n_params += 1;
                    }

                    parameters.push(Parameter {
                        name: param_name.to_owned(),
                        annotation: param_annotation,
                        is_type_known: is_param_type_known,
                        merge_key: effective_key,
                        location: Self::range_to_location(module, param.range),
                    });
                }

                let is_fully_annotated = return_annotation.is_some()
                    && parameters
                        .iter()
                        .all(|p| p.merge_key.is_none() || p.annotation.is_some());
                let is_type_known =
                    is_fully_annotated && is_return_type_known && all_params_type_known;

                functions.push(Function {
                    name: func_name,
                    return_annotation,
                    is_return_type_known,
                    parameters,
                    is_type_known,
                    property_role,
                    n_params,
                    slots: func_slots,
                    location,
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
                if Self::has_function_ancestor(&cls.parent) {
                    continue;
                }
                let class_prefix = format!(
                    "{module_prefix}{}",
                    Self::class_qualified_name(module, &cls.parent, &cls.def.name)
                );
                let target_qualified = format!("{}.{}", class_prefix, target_name);
                if let Some(target_func) = functions.iter().find(|f| f.name == target_qualified) {
                    let alias_name = format!("{}.{}", class_prefix, field.name);
                    let location = Self::range_to_location(module, field.range);
                    functions.push(Function {
                        name: alias_name,
                        slots: target_func.slots,
                        location,
                        return_annotation: target_func.return_annotation.clone(),
                        is_return_type_known: target_func.is_return_type_known,
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

    /// Only the first parameter (`self`/`cls`) is allowed to be unannotated.
    fn is_function_completely_annotated(
        bindings: &Bindings,
        answers: &Answers,
        undecorated_idx: Idx<KeyUndecoratedFunction>,
    ) -> bool {
        let fun = bindings.get(undecorated_idx);
        let return_key = Key::ReturnType(ShortIdentifier::new(&fun.def.name));
        let return_idx = bindings.key_to_idx(&return_key);
        let has_return_annotation = if let Binding::ReturnType(ret) = bindings.get(return_idx) {
            matches!(
                &ret.kind,
                ReturnTypeKind::ShouldValidateAnnotation { .. }
                    | ReturnTypeKind::ShouldTrustAnnotation { .. }
            )
        } else {
            false
        };

        if !has_return_annotation {
            return false;
        }

        let implicit_receiver = Self::has_implicit_receiver(fun, answers, undecorated_idx);
        Self::params_with_keys(&fun.def.parameters, implicit_receiver)
            .iter()
            .all(|(key, param)| key.is_none() || param.annotation.is_some())
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
            "__getattr__" | "__delattr__" | "__setattr__" | "__format__" | "__buffer__"
            | "__release_buffer__" => non_self_param_pos == 0,
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

    /// Check whether a decorator expression matches `type_check_only`.
    /// Handles `@type_check_only`, `@typing.type_check_only`, and call forms.
    fn is_type_check_only_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Name(name) => name.id.as_str() == "type_check_only",
            Expr::Attribute(attr) => attr.attr.as_str() == "type_check_only",
            Expr::Call(call) => Self::is_type_check_only_expr(&call.func),
            _ => false,
        }
    }

    /// Check if any decorator in the list is `@type_check_only`.
    fn has_type_check_only_decorator(
        decorators: &[Idx<KeyDecorator>],
        bindings: &Bindings,
    ) -> bool {
        decorators.iter().any(|&dec_idx| {
            let decorator = bindings.get(dec_idx);
            Self::is_type_check_only_expr(&decorator.expr)
        })
    }

    /// Collect all class keys that have the `@type_check_only` decorator.
    fn collect_type_check_only_classes(bindings: &Bindings) -> SmallSet<Idx<KeyClass>> {
        let mut tco_classes = SmallSet::new();
        for idx in bindings.keys::<Key>() {
            if let Binding::ClassDef(class_key, decorators) = bindings.get(idx)
                && Self::has_type_check_only_decorator(decorators, bindings)
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

    /// Calculate the aggregate summary by summing per-module entity counts.
    fn calculate_summary(module_reports: &[ModuleReport]) -> ReportSummary {
        let n_modules = module_reports.len();
        let mut total_slots = SlotCounts::default();
        let mut n_functions = 0usize;
        let mut n_methods = 0usize;
        let mut n_function_params = 0usize;
        let mut n_method_params = 0usize;
        let mut n_classes = 0usize;
        let mut n_attrs = 0usize;
        let mut n_properties = 0usize;
        let mut n_type_ignores = 0usize;

        for module in module_reports {
            total_slots = total_slots.merge(module.slots);
            n_functions += module.n_functions;
            n_methods += module.n_methods;
            n_function_params += module.n_function_params;
            n_method_params += module.n_method_params;
            n_classes += module.n_classes;
            n_attrs += module.n_attrs;
            n_properties += module.n_properties;
            n_type_ignores += module.n_type_ignores;
        }

        ReportSummary {
            n_modules,
            slots: total_slots,
            coverage: total_slots.coverage(),
            strict_coverage: total_slots.strict_coverage(),
            n_functions,
            n_methods,
            n_function_params,
            n_method_params,
            n_classes,
            n_attrs,
            n_properties,
            n_type_ignores,
        }
    }

    fn parse_classes(
        module: &Module,
        bindings: &Bindings,
        answers: &Answers,
        transaction: &Transaction,
        handle: &Handle,
        tco_classes: &SmallSet<Idx<KeyClass>>,
    ) -> Vec<ReportClass> {
        let mut classes = Vec::new();
        let module_prefix = if module.name() != ModuleName::unknown() {
            format!("{}.", module.name())
        } else {
            String::new()
        };
        for class_idx in bindings.keys::<KeyClass>() {
            // Skip @type_check_only classes.
            if tco_classes.contains(&class_idx) {
                continue;
            }
            let binding_class = bindings.get(class_idx);
            let cls_binding = match binding_class {
                BindingClass::ClassDef(cls) => cls,
                BindingClass::FunctionalClassDef(..) => continue,
            };
            // Skip classes nested inside functions, since they are not public symbols.
            if Self::has_function_ancestor(&cls_binding.parent) {
                continue;
            }
            let class_type = match answers.get_idx(class_idx) {
                Some(result) => match &result.0 {
                    Some(cls) => cls.clone(),
                    None => continue,
                },
                None => continue,
            };
            let class_name = format!(
                "{module_prefix}{}",
                Self::class_qualified_name(module, &cls_binding.parent, &cls_binding.def.name)
            );
            let mro = answers
                .get_idx(bindings.key_to_idx(&KeyClassMro(ClassDefIndex(class_type.index().0))))
                .unwrap_or_else(|| Arc::new(ClassMro::Cyclic));
            // Check methods defined directly on this class
            let mut incomplete_attributes = Vec::new();
            for idx in bindings.keys::<Key>() {
                if let Key::Definition(_id) = bindings.idx_to_key(idx)
                    && let Binding::Function(x, _pred, _class_meta) = bindings.get(idx)
                {
                    let decorated = bindings.get(*x);
                    let undecorated_idx = decorated.undecorated_idx;
                    let fun = bindings.get(undecorated_idx);
                    if let Some(func_class_key) = fun.class_key {
                        if func_class_key != class_idx {
                            continue;
                        }
                        let method_name = fun.def.name.to_string();
                        if !Self::is_function_completely_annotated(
                            bindings,
                            answers,
                            undecorated_idx,
                        ) {
                            incomplete_attributes.push(IncompleteAttribute {
                                name: method_name.clone(),
                                declared_in: class_name.clone(),
                            });
                        }
                    }
                }
            }
            // Check inherited methods
            if let ClassMro::Resolved(ancestors) = &*mro {
                for ancestor_class_type in ancestors {
                    let ancestor_class = ancestor_class_type.class_object();
                    let ancestor_name = {
                        let ancestor_module = ancestor_class.module();
                        let ancestor_module_prefix =
                            if ancestor_module.name() != ModuleName::unknown() {
                                format!("{}.", ancestor_module.name())
                            } else {
                                String::new()
                            };
                        let ancestor_parent_path = ancestor_module
                            .display(ancestor_class.qname().parent())
                            .to_string();
                        if ancestor_parent_path.is_empty() {
                            format!("{}{}", ancestor_module_prefix, ancestor_class.name())
                        } else {
                            format!(
                                "{}{}.{}",
                                ancestor_module_prefix,
                                ancestor_parent_path,
                                ancestor_class.name()
                            )
                        }
                    };
                    // Skip methods inherited from builtins
                    if ancestor_class.module_name().as_str() == "builtins" {
                        continue;
                    }
                    let Some(ancestor_class_fields) =
                        transaction.get_class_fields(handle, ancestor_class)
                    else {
                        continue;
                    };
                    for field_name in ancestor_class_fields.names() {
                        let field_name_str = field_name.to_string();
                        // Skip if we already have this attribute listed (it has been overridden
                        // by the current class or another class in the MRO)
                        if incomplete_attributes
                            .iter()
                            .any(|a| a.name == field_name_str)
                        {
                            continue;
                        }
                        if !ancestor_class_fields.is_field_annotated(field_name) {
                            incomplete_attributes.push(IncompleteAttribute {
                                name: field_name_str,
                                declared_in: ancestor_name.clone(),
                            });
                        }
                    }
                }
            }
            let location = Self::range_to_location(module, cls_binding.def.range);
            incomplete_attributes.sort();
            classes.push(ReportClass {
                name: class_name,
                incomplete_attributes,
                location,
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

    /// When a `.pyi` stub only covers a subset of a `.py` file's public
    /// symbols, add the uncovered symbols from the `.py` so that completeness
    /// metrics reflect the full module interface.
    fn merge_uncovered_py_symbols(
        stub_functions: &mut Vec<Function>,
        stub_variables: &mut Vec<Variable>,
        stub_classes: &mut Vec<ReportClass>,
        py_functions: Vec<Function>,
        py_variables: Vec<Variable>,
        py_classes: Vec<ReportClass>,
    ) {
        let stub_func_names: SmallSet<String> =
            stub_functions.iter().map(|f| f.name.clone()).collect();
        for py_func in py_functions {
            if !stub_func_names.contains(&py_func.name) {
                stub_functions.push(py_func);
            }
        }

        let stub_var_names: SmallSet<String> =
            stub_variables.iter().map(|v| v.name.clone()).collect();
        for py_var in py_variables {
            if !stub_var_names.contains(&py_var.name) {
                stub_variables.push(py_var);
            }
        }

        let stub_class_names: SmallSet<String> =
            stub_classes.iter().map(|c| c.name.clone()).collect();
        for py_class in py_classes {
            if !stub_class_names.contains(&py_class.name) {
                stub_classes.push(py_class);
            }
        }
    }

    fn build_module_report(
        name: String,
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

        // Compute per-module entity counts. Use the derived (file-based)
        // module name for prefix matching, since symbol names are built from
        // the derived name and only rewritten to the override name later.
        let module_prefix = format!("{}.", derived_name);
        let mut n_functions = 0usize;
        let mut n_methods = 0usize;
        let mut n_function_params = 0usize;
        let mut n_method_params = 0usize;
        let mut n_classes = 0usize;
        let mut n_attrs = 0usize;
        let mut n_properties = 0usize;
        // Count functions/methods using the `Function` list directly so we
        // can use `n_params` (accurate even for implicit-return dunders).
        for func in functions.iter().filter(|f| f.property_role.is_none()) {
            if Self::is_method(&func.name, &module_prefix) {
                n_methods += 1;
                n_method_params += func.n_params;
            } else {
                n_functions += 1;
                n_function_params += func.n_params;
            }
        }
        for sym in &symbol_reports {
            match sym {
                SymbolReport::Property { .. } => n_properties += 1,
                SymbolReport::Attr { .. } => n_attrs += 1,
                SymbolReport::Class { .. } => n_classes += 1,
                SymbolReport::Function { .. } => {}
            }
        }
        let n_type_ignores = suppressions.len();

        ModuleReport {
            name,
            names,
            line_count,
            symbol_reports,
            type_ignores: suppressions,
            coverage: total_slots.coverage(),
            strict_coverage: total_slots.strict_coverage(),
            slots: total_slots,
            n_functions,
            n_methods,
            n_function_params,
            n_method_params,
            n_classes,
            n_attrs,
            n_properties,
            n_type_ignores,
        }
    }

    fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        prefer_stubs: bool,
        module_name_override: Option<String>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder, thread_count);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Exports, None),
            true,
        );

        let transaction = forgetter.as_mut();
        let (handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());

        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("Failed to query sourcedb."));
        }

        let mut module_reports: Vec<ModuleReport> = Vec::new();
        transaction.run(handles.as_slice(), Require::Everything, None);

        let shadowed = if prefer_stubs {
            Self::py_paths_shadowed_by_pyi(&handles)
        } else {
            SmallSet::new()
        };

        // When prefer_stubs is true, build a mapping from .pyi paths to their
        // corresponding .py handles.
        let pyi_to_py: HashMap<PathBuf, &Handle> = if prefer_stubs {
            let py_by_path: HashMap<PathBuf, &Handle> = handles
                .iter()
                .filter(|h| !h.path().is_interface())
                .map(|h| (h.path().as_path().to_path_buf(), h))
                .collect();
            handles
                .iter()
                .filter(|h| h.path().is_interface())
                .filter_map(|h| {
                    let py_path = h.path().as_path().with_extension("py");
                    py_by_path
                        .get(&py_path)
                        .map(|&py_h| (h.path().as_path().to_path_buf(), py_h))
                })
                .collect()
        } else {
            HashMap::new()
        };
        for handle in &handles {
            if shadowed.contains(handle.path().as_path()) {
                continue;
            }

            if let Some(bindings) = transaction.get_bindings(handle)
                && let Some(module) = transaction.get_module_info(handle)
                && let Some(answers) = transaction.get_answers(handle)
            {
                let line_count = module.lined_buffer().line_index().line_count();
                let exports = transaction.get_exports(handle);
                let tco_classes = Self::collect_type_check_only_classes(&bindings);
                let mut functions =
                    Self::parse_functions(&module, &bindings, &answers, &exports, &tco_classes);
                Self::merge_overloads(&mut functions);
                let mut classes = Self::parse_classes(
                    &module,
                    &bindings,
                    &answers,
                    transaction,
                    handle,
                    &tco_classes,
                );
                let mut variables = Self::parse_variables(
                    &module, &bindings, &answers, &exports, &functions, &classes,
                );
                variables.extend(Self::parse_instance_attrs(
                    &module,
                    &bindings,
                    &answers,
                    &tco_classes,
                ));
                let suppressions = Self::parse_suppressions(&module);

                // When a .pyi stub shadows a .py file, include uncovered .py symbols.
                if let Some(py_handle) = pyi_to_py.get(&handle.path().as_path().to_path_buf())
                    && let Some(py_bindings) = transaction.get_bindings(py_handle)
                    && let Some(py_module) = transaction.get_module_info(py_handle)
                    && let Some(py_answers) = transaction.get_answers(py_handle)
                {
                    let py_exports = transaction.get_exports(py_handle);
                    let py_tco_classes = Self::collect_type_check_only_classes(&py_bindings);
                    let mut py_functions = Self::parse_functions(
                        &py_module,
                        &py_bindings,
                        &py_answers,
                        &py_exports,
                        &py_tco_classes,
                    );
                    Self::merge_overloads(&mut py_functions);
                    let py_classes = Self::parse_classes(
                        &py_module,
                        &py_bindings,
                        &py_answers,
                        transaction,
                        py_handle,
                        &py_tco_classes,
                    );
                    let mut py_variables = Self::parse_variables(
                        &py_module,
                        &py_bindings,
                        &py_answers,
                        &py_exports,
                        &py_functions,
                        &py_classes,
                    );
                    py_variables.extend(Self::parse_instance_attrs(
                        &py_module,
                        &py_bindings,
                        &py_answers,
                        &py_tco_classes,
                    ));
                    Self::merge_uncovered_py_symbols(
                        &mut functions,
                        &mut variables,
                        &mut classes,
                        py_functions,
                        py_variables,
                        py_classes,
                    );
                }

                let derived_name = handle.module().to_string();
                let name = module_name_override.clone().unwrap_or(derived_name.clone());
                let mut module_report = Self::build_module_report(
                    name.clone(),
                    &derived_name,
                    line_count,
                    &functions,
                    &variables,
                    &classes,
                    suppressions,
                );
                // When --module overrides the name, rewrite symbol prefixes to match.
                if module_name_override.is_some() && name != derived_name {
                    let old_prefix = format!("{}.", derived_name);
                    let new_prefix = format!("{}.", name);
                    for n in &mut module_report.names {
                        if let Some(rest) = n.strip_prefix(&old_prefix) {
                            *n = format!("{new_prefix}{rest}");
                        }
                    }
                    for sym in &mut module_report.symbol_reports {
                        let sym_name = match sym {
                            SymbolReport::Attr { name, .. }
                            | SymbolReport::Function { name, .. }
                            | SymbolReport::Class { name, .. }
                            | SymbolReport::Property { name, .. } => name,
                        };
                        if let Some(rest) = sym_name.strip_prefix(&old_prefix) {
                            *sym_name = format!("{new_prefix}{rest}");
                        }
                    }
                }
                module_reports.push(module_report);
            }
        }

        let summary = Self::calculate_summary(&module_reports);
        let full_report = FullReport {
            schema_version: format!("{}.{}", REPORT_SCHEMA_VERSION.0, REPORT_SCHEMA_VERSION.1),
            module_reports,
            summary,
        };
        let json = serde_json::to_string_pretty(&full_report)?;
        println!("{}", json);

        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;

    use super::*;
    use crate::state::require::Require;
    use crate::test::util::TestEnv;

    /// Load a checked-in test file from the REPORT_TEST_PATH directory.
    /// Normalizes `\r\n` to `\n` so snapshots pass on Windows.
    fn load_test_file(name: &str) -> String {
        let path = std::env::var("REPORT_TEST_PATH").expect("REPORT_TEST_PATH env var must be set");
        std::fs::read_to_string(PathBuf::from(path).join(name))
            .unwrap_or_else(|e| panic!("failed to read test file {name}: {e}"))
            .replace("\r\n", "\n")
    }

    /// Compare serialized JSON output against a checked-in expected file.
    /// When `REPORT_TEST_WRITE_PATH` is set, writes the actual output to that
    /// directory instead of comparing (use this to update snapshots).
    fn compare_snapshot<T: serde::Serialize>(name: &str, actual: &T) {
        let actual_json = serde_json::to_string_pretty(actual).unwrap();

        if let Ok(write_path) = std::env::var("REPORT_TEST_WRITE_PATH") {
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
             REPORT_TEST_WRITE_PATH set to the test_files directory."
        );
    }

    /// Build a `ModuleReport` from a checked-in Python test file,
    /// mirroring the production pipeline in `run_inner`.
    fn build_module_report_for_test(py_file: &str) -> ModuleReport {
        let code = load_test_file(py_file);
        let (state, handle_fn) = TestEnv::one("test", &code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let line_count = module.lined_buffer().line_index().line_count();
        let tco_classes = ReportArgs::collect_type_check_only_classes(&bindings);
        let mut functions =
            ReportArgs::parse_functions(&module, &bindings, &answers, &exports, &tco_classes);
        ReportArgs::merge_overloads(&mut functions);
        let classes = ReportArgs::parse_classes(
            &module,
            &bindings,
            &answers,
            &transaction,
            &handle,
            &tco_classes,
        );
        let mut variables = ReportArgs::parse_variables(
            &module, &bindings, &answers, &exports, &functions, &classes,
        );
        variables.extend(ReportArgs::parse_instance_attrs(
            &module,
            &bindings,
            &answers,
            &tco_classes,
        ));
        let suppressions = ReportArgs::parse_suppressions(&module);

        ReportArgs::build_module_report(
            "test".to_owned(),
            "test",
            line_count,
            &functions,
            &variables,
            &classes,
            suppressions,
        )
    }

    /// Build a `ModuleReport` with a module name override, mirroring
    /// the `run_inner` --module flag logic.
    fn build_module_report_with_override(py_file: &str, override_name: &str) -> ModuleReport {
        let mut report = build_module_report_for_test(py_file);
        let derived_name = "test";
        let old_prefix = format!("{}.", derived_name);
        let new_prefix = format!("{}.", override_name);
        report.name = override_name.to_owned();
        for n in &mut report.names {
            if let Some(rest) = n.strip_prefix(&old_prefix) {
                *n = format!("{new_prefix}{rest}");
            }
        }
        for sym in &mut report.symbol_reports {
            let sym_name = match sym {
                SymbolReport::Attr { name, .. }
                | SymbolReport::Function { name, .. }
                | SymbolReport::Class { name, .. }
                | SymbolReport::Property { name, .. } => name,
            };
            if let Some(rest) = sym_name.strip_prefix(&old_prefix) {
                *sym_name = format!("{new_prefix}{rest}");
            }
        }
        report
    }

    /// Build a `ModuleReport` that merges a `.pyi` stub with its `.py` source,
    /// mirroring the production pipeline in `run_inner` when `prefer_stubs` is
    /// true and both files exist for the same module.
    fn build_stub_module_report(pyi_file: &str, py_file: &str) -> ModuleReport {
        // Parse the stub
        let pyi_code = load_test_file(pyi_file);
        let (pyi_state, pyi_handle_fn) = TestEnv::one_with_path("test", "test.pyi", &pyi_code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let pyi_handle = pyi_handle_fn("test");
        let pyi_txn = pyi_state.transaction();

        let module = pyi_txn.get_module_info(&pyi_handle).unwrap();
        let bindings = pyi_txn.get_bindings(&pyi_handle).unwrap();
        let answers = pyi_txn.get_answers(&pyi_handle).unwrap();
        let exports = pyi_txn.get_exports(&pyi_handle);

        let line_count = module.lined_buffer().line_index().line_count();
        let tco_classes = ReportArgs::collect_type_check_only_classes(&bindings);
        let mut functions =
            ReportArgs::parse_functions(&module, &bindings, &answers, &exports, &tco_classes);
        ReportArgs::merge_overloads(&mut functions);
        let mut classes = ReportArgs::parse_classes(
            &module,
            &bindings,
            &answers,
            &pyi_txn,
            &pyi_handle,
            &tco_classes,
        );
        let mut variables = ReportArgs::parse_variables(
            &module, &bindings, &answers, &exports, &functions, &classes,
        );
        variables.extend(ReportArgs::parse_instance_attrs(
            &module,
            &bindings,
            &answers,
            &tco_classes,
        ));
        let suppressions = ReportArgs::parse_suppressions(&module);

        // Parse the .py source
        let py_code = load_test_file(py_file);
        let (py_state, py_handle_fn) = TestEnv::one("test", &py_code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let py_handle = py_handle_fn("test");
        let py_txn = py_state.transaction();

        let py_module = py_txn.get_module_info(&py_handle).unwrap();
        let py_bindings = py_txn.get_bindings(&py_handle).unwrap();
        let py_answers = py_txn.get_answers(&py_handle).unwrap();
        let py_exports = py_txn.get_exports(&py_handle);

        let py_tco_classes = ReportArgs::collect_type_check_only_classes(&py_bindings);
        let mut py_functions = ReportArgs::parse_functions(
            &py_module,
            &py_bindings,
            &py_answers,
            &py_exports,
            &py_tco_classes,
        );
        ReportArgs::merge_overloads(&mut py_functions);
        let py_classes = ReportArgs::parse_classes(
            &py_module,
            &py_bindings,
            &py_answers,
            &py_txn,
            &py_handle,
            &py_tco_classes,
        );
        let mut py_variables = ReportArgs::parse_variables(
            &py_module,
            &py_bindings,
            &py_answers,
            &py_exports,
            &py_functions,
            &py_classes,
        );
        py_variables.extend(ReportArgs::parse_instance_attrs(
            &py_module,
            &py_bindings,
            &py_answers,
            &py_tco_classes,
        ));

        // Merge uncovered symbols from .py into the stub report
        ReportArgs::merge_uncovered_py_symbols(
            &mut functions,
            &mut variables,
            &mut classes,
            py_functions,
            py_variables,
            py_classes,
        );

        ReportArgs::build_module_report(
            "test".to_owned(),
            "test",
            line_count,
            &functions,
            &variables,
            &classes,
            suppressions,
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
        let shadowed = ReportArgs::py_paths_shadowed_by_pyi(&handles);

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
                is_return_type_known: has_return,
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

    /// --module name override: entity counts (n_functions vs n_methods) must
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

    #[test]
    fn test_full_report_has_schema_version() {
        let report = build_module_report_for_test("functions.py");
        let full = FullReport {
            schema_version: format!("{}.{}", REPORT_SCHEMA_VERSION.0, REPORT_SCHEMA_VERSION.1),
            module_reports: vec![report],
            summary: ReportArgs::calculate_summary(&[]),
        };
        let json: serde_json::Value = serde_json::to_value(&full).unwrap();
        let version = json["schema_version"].as_str().unwrap();
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(parts.len(), 2, "schema_version must be \"major.minor\"");
        assert!(parts[0].parse::<u32>().is_ok());
        assert!(parts[1].parse::<u32>().is_ok());
    }

    /// Type aliases (explicit `TypeAlias`, bare assignments, PEP 695, TypeAliasType)
    /// are type-level constructs and should have 0 typable slots.
    #[test]
    fn test_report_type_aliases() {
        let report = build_module_report_for_test("type_aliases.py");
        compare_snapshot("type_aliases.expected.json", &report);
    }
}
