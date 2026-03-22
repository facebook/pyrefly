/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use pyrefly_build::handle::Handle;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::nesting_context::NestingContext;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::class::ClassDefIndex;
use pyrefly_types::types::Type;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::includes::Includes;
use regex::Regex;
use ruff_python_ast::Parameters;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::Answers;
use crate::alt::types::class_metadata::ClassMro;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::BindingClass;
use crate::binding::binding::BindingExport;
use crate::binding::binding::FunctionDefData;
use crate::binding::binding::Key;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyClassMro;
use crate::binding::binding::KeyExport;
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

/// Location of a code element (start of its declaration)
#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct Location {
    line: usize,
    column: usize,
}

/// Parameter information
#[derive(Debug, Serialize)]
struct Parameter {
    name: String,
    annotation: Option<String>,
    /// Whether the parameter's resolved type is fully known (contains no `Any`).
    is_type_known: bool,
    location: Location,
}

/// Suppression information
#[derive(Debug, Serialize)]
struct Suppression {
    kind: String,
    codes: Vec<String>,
    location: Location,
}

/// Function information
#[derive(Debug, Serialize)]
struct Function {
    name: String,
    return_annotation: Option<String>,
    /// Whether the return type is fully known (contains no `Any`).
    is_return_type_known: bool,
    parameters: Vec<Parameter>,
    /// Whether the function is completely type-known (return + all params known).
    /// This is only true for functions that are also fully annotated.
    is_type_known: bool,
    location: Location,
}

/// An incomplete attribute within a class (method with missing annotations)
#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct IncompleteAttribute {
    name: String,
    declared_in: String,
}

/// Information about a class with incomplete annotations
#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ReportClass {
    name: String,
    incomplete_attributes: Vec<IncompleteAttribute>,
    location: Location,
}

/// A top-level exported variable.
#[derive(Debug, Serialize)]
struct Variable {
    name: String,
    annotation: Option<String>,
    location: Location,
}

/// File report
#[derive(Debug, Serialize)]
struct FileReport {
    variables: Vec<Variable>,
    line_count: usize,
    functions: Vec<Function>,
    classes: Vec<ReportClass>,
    suppressions: Vec<Suppression>,
    /// Percentage of functions that are fully annotated (0.0 to 100.0).
    /// A function is fully annotated if it has return and parameter annotations present.
    annotation_completeness: f64,
    /// Percentage of fully-annotated functions whose resolved types contain no `Any` (0.0 to 100.0).
    type_completeness: f64,
}

/// Summary statistics for the entire report.
#[derive(Debug, Serialize)]
struct ReportSummary {
    /// Total number of files analyzed.
    total_files: usize,
    /// Total number of functions across all files.
    total_functions: usize,
    /// Number of functions that are fully annotated (have annotation text).
    fully_annotated_functions: usize,
    /// Number of fully-annotated functions whose resolved types contain no `Any`.
    type_complete_functions: usize,
    /// Aggregate annotation completeness score across all files (0.0 to 100.0).
    aggregate_annotation_completeness: f64,
    /// Aggregate type completeness across all files (0.0 to 100.0).
    /// Denominator is fully_annotated_functions, not total_functions.
    aggregate_type_completeness: f64,
}

/// Full report including per-file reports and aggregate summary.
#[derive(Debug, Serialize)]
struct FullReport {
    files: HashMap<String, FileReport>,
    summary: ReportSummary,
}

/// Generate reports from pyrefly type checking results.
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Clone, Parser)]
pub struct ReportArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    /// Configuration override options
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// When enabled, `.py` files are skipped if a corresponding `.pyi`
    /// file is also present in the set of files to check.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    prefer_stubs: bool,
}

impl ReportArgs {
    pub fn run(
        self,
        wrapper: Option<ConfigConfigurerWrapper>,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override, wrapper)?;
        Self::run_inner(files_to_check, config_finder, self.prefer_stubs)
    }

    /// Helper to extract all parameters from Parameters struct
    fn extract_parameters(params: &Parameters) -> Vec<&ruff_python_ast::Parameter> {
        let mut all_params = Vec::new();
        all_params.extend(params.posonlyargs.iter().map(|p| &p.parameter));
        all_params.extend(params.args.iter().map(|p| &p.parameter));
        if let Some(vararg) = &params.vararg {
            all_params.push(vararg);
        }
        all_params.extend(params.kwonlyargs.iter().map(|p| &p.parameter));
        if let Some(kwarg) = &params.kwarg {
            all_params.push(kwarg);
        }
        all_params
    }

    fn range_to_location(module: &Module, range: TextRange) -> Location {
        let pos = module.lined_buffer().display_pos(range.start(), None);
        Location {
            line: pos.line_within_file().get() as usize,
            column: pos.column().get() as usize,
        }
    }

    /// Helper to parse suppression comments from source code
    fn parse_suppressions(module: &Module) -> Vec<Suppression> {
        let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[([^\]]*)\]").unwrap();
        let source = module.lined_buffer().contents();
        let lines: Vec<&str> = source.lines().collect();
        let mut suppressions = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            if let Some(caps) = regex.captures(line) {
                let codes: Vec<String> = caps
                    .get(1)
                    .map(|m| {
                        m.as_str()
                            .split(',')
                            .map(|s| s.trim().to_owned())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();

                // Find the position of the comment in the line
                if let Some(comment_start) = line.find('#') {
                    let line_number = line_idx + 1; // 1-indexed
                    let start_col = comment_start + 1; // 1-indexed column

                    suppressions.push(Suppression {
                        kind: "ignore".to_owned(),
                        codes,
                        location: Location {
                            line: line_number,
                            column: start_col,
                        },
                    });
                }
            }
        }

        suppressions
    }

    /// Check if any ancestor in a `NestingContext` chain is a function.
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

    fn parse_variables(
        module: &Module,
        bindings: &Bindings,
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
        let reported_names: HashSet<&str> = functions
            .iter()
            .map(|f| f.name.as_str())
            .chain(classes.iter().map(|c| c.name.as_str()))
            .collect();

        let mut variables = Vec::new();
        for idx in bindings.keys::<KeyExport>() {
            let KeyExport(name) = bindings.idx_to_key(idx);
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
                BindingExport::AnnotatedForward(annot_idx, _) => {
                    let annotation_text = match bindings.get(*annot_idx) {
                        BindingAnnotation::AnnotateExpr(_, expr, _) => {
                            Some(module.code_at(expr.range()).to_owned())
                        }
                        _ => None,
                    };
                    variables.push(Variable {
                        name: qualified_name,
                        annotation: annotation_text,
                        location,
                    });
                }
                BindingExport::Forward(idx) => match bindings.get(*idx) {
                    // Skip injected implicit globals
                    Binding::Global(_) => {}
                    _ => {
                        variables.push(Variable {
                            name: qualified_name,
                            annotation: None,
                            location,
                        });
                    }
                },
            }
        }
        variables.sort_by(|a, b| a.location.cmp(&b.location));
        variables
    }

    fn parse_functions(
        module: &Module,
        bindings: &Bindings,
        answers: &Answers,
        exports: &SmallMap<Name, ExportLocation>,
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
                let fun = bindings.get(bindings.get(*x).undecorated_idx);
                let location = Self::range_to_location(module, fun.def.range);
                let func_name = if let Some(class_key) = fun.class_key {
                    match bindings.get(class_key) {
                        BindingClass::ClassDef(cls) => {
                            // Skip methods of classes nested inside functions
                            if Self::has_function_ancestor(&cls.parent) {
                                continue;
                            }
                            // Build full qualified name using nesting context
                            let parent_path = module.display(&cls.parent).to_string();
                            if parent_path.is_empty() {
                                format!("{}{}.{}", module_prefix, cls.def.name, fun.def.name)
                            } else {
                                format!(
                                    "{}{}.{}.{}",
                                    module_prefix, parent_path, cls.def.name, fun.def.name
                                )
                            }
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

                // Return type is known only if an annotation is present and the resolved
                // type contains no Any.
                let is_return_type_known = return_annotation.is_some()
                    && answers
                        .get_type_at(return_idx)
                        .is_some_and(|t| Self::is_type_fully_known(&t));

                // Get parameters with their annotation and type-known status
                let mut parameters = Vec::new();
                let all_params = Self::extract_parameters(&fun.def.parameters);
                let mut all_params_type_known = true;

                for (i, param) in all_params.iter().enumerate() {
                    let param_name = param.name.as_str();
                    let param_annotation = param
                        .annotation
                        .as_ref()
                        .map(|ann| module.code_at(ann.range()).to_owned());

                    // First parameter named self/cls is always considered type-known
                    let is_param_type_known =
                        if i == 0 && (param_name == "self" || param_name == "cls") {
                            true
                        } else if let Some(ann) = &param.annotation {
                            answers
                                .get_type_trace(ann.range())
                                .is_some_and(|t| Self::is_type_fully_known(&t))
                        } else {
                            false // No annotation means not type-known
                        };

                    if !is_param_type_known
                        && !(i == 0 && (param_name == "self" || param_name == "cls"))
                    {
                        all_params_type_known = false;
                    }

                    parameters.push(Parameter {
                        name: param_name.to_owned(),
                        annotation: param_annotation,
                        is_type_known: is_param_type_known,
                        location: Self::range_to_location(module, param.range),
                    });
                }

                // A function is type-known only if it is fully annotated AND
                // its return type and all param types contain no Any.
                let is_fully_annotated = return_annotation.is_some()
                    && parameters.iter().enumerate().all(|(i, p)| {
                        (i == 0 && (p.name == "self" || p.name == "cls")) || p.annotation.is_some()
                    });
                let is_type_known =
                    is_fully_annotated && is_return_type_known && all_params_type_known;

                functions.push(Function {
                    name: func_name,
                    return_annotation,
                    is_return_type_known,
                    parameters,
                    is_type_known,
                    location,
                });
            }
        }
        functions
    }

    /// Check if a function is completely annotated (has return annotation and all params annotated).
    /// Only the first parameter is allowed to be unannotated if it is named `self` or `cls`.
    fn is_function_completely_annotated(bindings: &Bindings, func_def: &FunctionDefData) -> bool {
        // Check return annotation
        let return_key = Key::ReturnType(ShortIdentifier::new(&func_def.name));
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

        // Check all parameters. Only the first parameter named self/cls may be unannotated.
        let all_params = Self::extract_parameters(&func_def.parameters);
        for (i, param) in all_params.iter().enumerate() {
            if i == 0 && (param.name.as_str() == "self" || param.name.as_str() == "cls") {
                continue;
            }
            if param.annotation.is_none() {
                return false;
            }
        }

        true
    }

    /// Check if a function is fully annotated based on its parsed representation.
    /// A function is fully annotated if it has a return annotation and all parameters
    /// have annotations. Only the first parameter named `self`/`cls` is exempt.
    fn is_fully_annotated(function: &Function) -> bool {
        if function.return_annotation.is_none() {
            return false;
        }
        function.parameters.iter().enumerate().all(|(i, p)| {
            (i == 0 && (p.name == "self" || p.name == "cls")) || p.annotation.is_some()
        })
    }

    /// Returns true if the type contains no `Any` anywhere in its structure.
    fn is_type_fully_known(ty: &Type) -> bool {
        !ty.any(|t| t.is_any())
    }

    /// Calculate the aggregate summary for all file reports.
    fn calculate_summary(files: &HashMap<String, FileReport>) -> ReportSummary {
        let total_files = files.len();
        let all_functions: Vec<&Function> = files.values().flat_map(|f| &f.functions).collect();
        let total_functions = all_functions.len();
        let fully_annotated_functions = all_functions
            .iter()
            .filter(|f| Self::is_fully_annotated(f))
            .count();
        // Type-complete functions are a subset of fully-annotated functions.
        let type_complete_functions = all_functions.iter().filter(|f| f.is_type_known).count();

        let aggregate_annotation_completeness = if total_functions == 0 {
            100.0
        } else {
            (fully_annotated_functions as f64 / total_functions as f64) * 100.0
        };

        let aggregate_type_completeness = if fully_annotated_functions == 0 {
            100.0
        } else {
            (type_complete_functions as f64 / fully_annotated_functions as f64) * 100.0
        };

        ReportSummary {
            total_files,
            total_functions,
            fully_annotated_functions,
            type_complete_functions,
            aggregate_annotation_completeness,
            aggregate_type_completeness,
        }
    }

    fn parse_classes(
        module: &Module,
        bindings: &Bindings,
        answers: &Answers,
        transaction: &Transaction,
        handle: &Handle,
    ) -> Vec<ReportClass> {
        let mut classes = Vec::new();
        let module_prefix = if module.name() != ModuleName::unknown() {
            format!("{}.", module.name())
        } else {
            String::new()
        };
        for class_idx in bindings.keys::<KeyClass>() {
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
            let class_name = {
                let parent_path = module.display(&cls_binding.parent).to_string();
                if parent_path.is_empty() {
                    format!("{}{}", module_prefix, cls_binding.def.name)
                } else {
                    format!("{}{}.{}", module_prefix, parent_path, cls_binding.def.name)
                }
            };
            let mro = answers
                .get_idx(bindings.key_to_idx(&KeyClassMro(ClassDefIndex(class_type.index().0))))
                .unwrap_or_else(|| Arc::new(ClassMro::Cyclic));
            // Check methods defined directly on this class
            let mut incomplete_attributes = Vec::new();
            for idx in bindings.keys::<Key>() {
                if let Key::Definition(_id) = bindings.idx_to_key(idx)
                    && let Binding::Function(x, _pred, _class_meta) = bindings.get(idx)
                {
                    let fun = bindings.get(bindings.get(*x).undecorated_idx);
                    if let Some(func_class_key) = fun.class_key {
                        if func_class_key != class_idx {
                            continue;
                        }
                        let method_name = fun.def.name.to_string();
                        if !Self::is_function_completely_annotated(bindings, &fun.def) {
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
                        // Skip if we already have this attribute listed (it has been overridden by the current class or another class in the MRO)
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
    fn py_paths_shadowed_by_pyi(handles: &[Handle]) -> HashSet<PathBuf> {
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
        let stub_func_names: HashSet<String> =
            stub_functions.iter().map(|f| f.name.clone()).collect();
        for py_func in py_functions {
            if !stub_func_names.contains(&py_func.name) {
                stub_functions.push(py_func);
            }
        }

        let stub_var_names: HashSet<String> =
            stub_variables.iter().map(|v| v.name.clone()).collect();
        for py_var in py_variables {
            if !stub_var_names.contains(&py_var.name) {
                stub_variables.push(py_var);
            }
        }

        let stub_class_names: HashSet<String> =
            stub_classes.iter().map(|c| c.name.clone()).collect();
        for py_class in py_classes {
            if !stub_class_names.contains(&py_class.name) {
                stub_classes.push(py_class);
            }
        }
    }

    fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        prefer_stubs: bool,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder);
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

        let mut report: HashMap<String, FileReport> = HashMap::new();
        transaction.run(handles.as_slice(), Require::Everything, None);

        let shadowed = if prefer_stubs {
            Self::py_paths_shadowed_by_pyi(&handles)
        } else {
            HashSet::new()
        };

        // When prefer_stubs is true, build a mapping from .pyi paths to their
        // corresponding .py handles. This allows us to cross-reference exports
        // and include uncovered .py symbols in the completeness calculations.
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
                let mut functions = Self::parse_functions(&module, &bindings, &answers, &exports);
                let mut classes =
                    Self::parse_classes(&module, &bindings, &answers, transaction, handle);
                let mut variables =
                    Self::parse_variables(&module, &bindings, &exports, &functions, &classes);
                let suppressions = Self::parse_suppressions(&module);

                // When a .pyi stub shadows a .py file, the stub may only cover
                // a subset of the .py's public symbols. Include uncovered .py
                // symbols so that the completeness denominator reflects the full
                // module interface.
                if let Some(py_handle) = pyi_to_py.get(&handle.path().as_path().to_path_buf())
                    && let Some(py_bindings) = transaction.get_bindings(py_handle)
                    && let Some(py_module) = transaction.get_module_info(py_handle)
                    && let Some(py_answers) = transaction.get_answers(py_handle)
                {
                    let py_exports = transaction.get_exports(py_handle);
                    let py_functions =
                        Self::parse_functions(&py_module, &py_bindings, &py_answers, &py_exports);
                    let py_classes = Self::parse_classes(
                        &py_module,
                        &py_bindings,
                        &py_answers,
                        transaction,
                        py_handle,
                    );
                    let py_variables = Self::parse_variables(
                        &py_module,
                        &py_bindings,
                        &py_exports,
                        &py_functions,
                        &py_classes,
                    );
                    Self::merge_uncovered_py_symbols(
                        &mut functions,
                        &mut variables,
                        &mut classes,
                        py_functions,
                        py_variables,
                        py_classes,
                    );
                }

                let annotated_count = functions
                    .iter()
                    .filter(|f| Self::is_fully_annotated(f))
                    .count();
                let annotation_completeness = if functions.is_empty() {
                    100.0
                } else {
                    (annotated_count as f64 / functions.len() as f64) * 100.0
                };
                let type_completeness = if annotated_count == 0 {
                    100.0
                } else {
                    let type_complete = functions.iter().filter(|f| f.is_type_known).count();
                    (type_complete as f64 / annotated_count as f64) * 100.0
                };

                report.insert(
                    handle.path().as_path().display().to_string(),
                    FileReport {
                        variables,
                        line_count,
                        functions,
                        classes,
                        suppressions,
                        annotation_completeness,
                        type_completeness,
                    },
                );
            }
        }

        // Output JSON
        let summary = Self::calculate_summary(&report);
        let full_report = FullReport {
            files: report,
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

    /// Build a full `FileReport` from a checked-in Python test file,
    /// mirroring the production pipeline in `run_inner`.
    fn build_file_report(py_file: &str) -> FileReport {
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
        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);
        let classes =
            ReportArgs::parse_classes(&module, &bindings, &answers, &transaction, &handle);
        let variables =
            ReportArgs::parse_variables(&module, &bindings, &exports, &functions, &classes);
        let suppressions = ReportArgs::parse_suppressions(&module);

        let annotated_count = functions
            .iter()
            .filter(|f| ReportArgs::is_fully_annotated(f))
            .count();
        let annotation_completeness = if functions.is_empty() {
            100.0
        } else {
            (annotated_count as f64 / functions.len() as f64) * 100.0
        };
        let type_completeness = if annotated_count == 0 {
            100.0
        } else {
            let type_complete = functions.iter().filter(|f| f.is_type_known).count();
            (type_complete as f64 / annotated_count as f64) * 100.0
        };

        FileReport {
            variables,
            line_count,
            functions,
            classes,
            suppressions,
            annotation_completeness,
            type_completeness,
        }
    }

    /// Build a `FileReport` that merges a `.pyi` stub with its `.py` source,
    /// mirroring the production pipeline in `run_inner` when `prefer_stubs` is
    /// true and both files exist for the same module.
    fn build_stub_report(pyi_file: &str, py_file: &str) -> FileReport {
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
        let mut functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);
        let mut classes =
            ReportArgs::parse_classes(&module, &bindings, &answers, &pyi_txn, &pyi_handle);
        let mut variables =
            ReportArgs::parse_variables(&module, &bindings, &exports, &functions, &classes);
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

        let py_functions =
            ReportArgs::parse_functions(&py_module, &py_bindings, &py_answers, &py_exports);
        let py_classes =
            ReportArgs::parse_classes(&py_module, &py_bindings, &py_answers, &py_txn, &py_handle);
        let py_variables = ReportArgs::parse_variables(
            &py_module,
            &py_bindings,
            &py_exports,
            &py_functions,
            &py_classes,
        );

        // Merge uncovered symbols from .py into the stub report
        ReportArgs::merge_uncovered_py_symbols(
            &mut functions,
            &mut variables,
            &mut classes,
            py_functions,
            py_variables,
            py_classes,
        );

        let annotated_count = functions
            .iter()
            .filter(|f| ReportArgs::is_fully_annotated(f))
            .count();
        let annotation_completeness = if functions.is_empty() {
            100.0
        } else {
            (annotated_count as f64 / functions.len() as f64) * 100.0
        };
        let type_completeness = if annotated_count == 0 {
            100.0
        } else {
            let type_complete = functions.iter().filter(|f| f.is_type_known).count();
            (type_complete as f64 / annotated_count as f64) * 100.0
        };

        FileReport {
            variables,
            line_count,
            functions,
            classes,
            suppressions,
            annotation_completeness,
            type_completeness,
        }
    }

    #[test]
    fn test_report_suppressions() {
        let report = build_file_report("suppressions.py");
        compare_snapshot("suppressions.expected.json", &report);
    }

    #[test]
    fn test_report_variables() {
        let report = build_file_report("variables.py");
        compare_snapshot("variables.expected.json", &report);
    }

    #[test]
    fn test_report_functions() {
        let report = build_file_report("functions.py");
        compare_snapshot("functions.expected.json", &report);
    }

    #[test]
    fn test_report_incomplete_methods() {
        let report = build_file_report("incomplete_methods.py");
        compare_snapshot("incomplete_methods.expected.json", &report);
    }

    #[test]
    fn test_report_nested_classes() {
        let report = build_file_report("nested_classes.py");
        compare_snapshot("nested_classes.expected.json", &report);
    }

    #[test]
    fn test_report_inheritance() {
        let report = build_file_report("inheritance.py");
        compare_snapshot("inheritance.expected.json", &report);
    }

    #[test]
    fn test_report_nested_exclusions() {
        let report = build_file_report("nested_exclusions.py");
        compare_snapshot("nested_exclusions.expected.json", &report);
    }

    /// When a .pyi stub only covers a subset of the .py file's exports,
    /// the uncovered symbols appear as unannotated and reduce completeness.
    #[test]
    fn test_report_partial_stub() {
        let report = build_stub_report("partial_stub.pyi", "partial_stub.py");
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
                    .map(|(param_name, annotated)| Parameter {
                        name: param_name.to_owned(),
                        annotation: if annotated {
                            Some("str".to_owned())
                        } else {
                            None
                        },
                        is_type_known: annotated,
                        location: Location { line: 1, column: 1 },
                    })
                    .collect(),
                is_type_known: false, // Not relevant for annotation-only tests
                location: Location { line: 1, column: 1 },
            }
        }

        // Fully annotated function
        assert!(ReportArgs::is_fully_annotated(&make_function(
            "foo",
            true,
            vec![("x", true)]
        )));

        // self as first param is exempt
        assert!(ReportArgs::is_fully_annotated(&make_function(
            "bar",
            true,
            vec![("self", false), ("y", true)]
        )));

        // cls as first param is exempt
        assert!(ReportArgs::is_fully_annotated(&make_function(
            "cls_method",
            true,
            vec![("cls", false)]
        )));

        // Missing return annotation
        assert!(!ReportArgs::is_fully_annotated(&make_function(
            "no_return",
            false,
            vec![]
        )));

        // Missing parameter annotation
        assert!(!ReportArgs::is_fully_annotated(&make_function(
            "missing_param",
            true,
            vec![("x", false)]
        )));

        // "self" as a non-first parameter should NOT be exempt
        assert!(!ReportArgs::is_fully_annotated(&make_function(
            "bad_self",
            true,
            vec![("x", true), ("self", false)]
        )));

        // "cls" as a non-first parameter should NOT be exempt
        assert!(!ReportArgs::is_fully_annotated(&make_function(
            "bad_cls",
            true,
            vec![("x", true), ("cls", false)]
        )));

        // No functions means 100% complete
        let empty: Vec<Function> = vec![];
        let summary = ReportArgs::calculate_summary(&HashMap::new());
        assert_eq!(summary.aggregate_annotation_completeness, 100.0);
        assert_eq!(summary.total_functions, 0);
        assert!(empty.is_empty()); // suppress unused warning
    }

    #[test]
    fn test_calculate_summary() {
        /// Helper to create a Function for summary testing.
        fn make_function(
            name: &str,
            has_return: bool,
            is_type_known: bool,
            params: Vec<(&str, bool)>,
        ) -> Function {
            Function {
                name: name.to_owned(),
                return_annotation: if has_return {
                    Some("int".to_owned())
                } else {
                    None
                },
                is_return_type_known: is_type_known,
                parameters: params
                    .into_iter()
                    .map(|(param_name, annotated)| Parameter {
                        name: param_name.to_owned(),
                        annotation: if annotated {
                            Some("str".to_owned())
                        } else {
                            None
                        },
                        is_type_known: annotated,
                        location: Location { line: 1, column: 1 },
                    })
                    .collect(),
                is_type_known,
                location: Location { line: 1, column: 1 },
            }
        }

        // Single file with mixed annotation status
        let mut single_file: HashMap<String, FileReport> = HashMap::new();
        single_file.insert(
            "file1.py".to_owned(),
            FileReport {
                variables: vec![],
                line_count: 10,
                functions: vec![
                    make_function("foo", true, true, vec![("x", true)]),
                    make_function("bar", false, false, vec![("y", true)]),
                ],
                classes: vec![],
                suppressions: vec![],
                annotation_completeness: 50.0,
                type_completeness: 100.0,
            },
        );
        let summary = ReportArgs::calculate_summary(&single_file);
        assert_eq!(summary.total_files, 1);
        assert_eq!(summary.total_functions, 2);
        assert_eq!(summary.fully_annotated_functions, 1);
        assert_eq!(summary.type_complete_functions, 1);
        assert_eq!(summary.aggregate_annotation_completeness, 50.0);
        assert_eq!(summary.aggregate_type_completeness, 100.0);

        // Multiple files — aggregate is weighted by function count, not averaged per file
        let mut multi_file: HashMap<String, FileReport> = HashMap::new();
        multi_file.insert(
            "file1.py".to_owned(),
            FileReport {
                variables: vec![],
                line_count: 10,
                functions: vec![
                    make_function("foo", true, true, vec![("x", true)]),
                    make_function("bar", true, true, vec![("y", true)]),
                ],
                classes: vec![],
                suppressions: vec![],
                annotation_completeness: 100.0,
                type_completeness: 100.0,
            },
        );
        multi_file.insert(
            "file2.py".to_owned(),
            FileReport {
                variables: vec![],
                line_count: 20,
                functions: vec![
                    make_function("baz", true, true, vec![("z", true)]),
                    make_function("qux", false, false, vec![("w", false)]),
                ],
                classes: vec![],
                suppressions: vec![],
                annotation_completeness: 50.0,
                type_completeness: 100.0,
            },
        );
        let summary = ReportArgs::calculate_summary(&multi_file);
        assert_eq!(summary.total_files, 2);
        assert_eq!(summary.total_functions, 4);
        assert_eq!(summary.fully_annotated_functions, 3);
        assert_eq!(summary.type_complete_functions, 3);
        assert_eq!(summary.aggregate_annotation_completeness, 75.0);
        assert_eq!(summary.aggregate_type_completeness, 100.0);

        // self/cls exemption only applies to first parameter
        let mut with_self: HashMap<String, FileReport> = HashMap::new();
        with_self.insert(
            "file.py".to_owned(),
            FileReport {
                variables: vec![],
                line_count: 5,
                functions: vec![make_function(
                    "method",
                    true,
                    true,
                    vec![("self", false), ("x", true)],
                )],
                classes: vec![],
                suppressions: vec![],
                annotation_completeness: 100.0,
                type_completeness: 100.0,
            },
        );
        let summary = ReportArgs::calculate_summary(&with_self);
        assert_eq!(summary.fully_annotated_functions, 1);
        assert_eq!(summary.type_complete_functions, 1);
        assert_eq!(summary.aggregate_annotation_completeness, 100.0);
        assert_eq!(summary.aggregate_type_completeness, 100.0);

        // Annotated but not type-complete (contains Any)
        let mut with_any: HashMap<String, FileReport> = HashMap::new();
        with_any.insert(
            "file.py".to_owned(),
            FileReport {
                variables: vec![],
                line_count: 5,
                functions: vec![
                    // Annotated and type-complete
                    make_function("good", true, true, vec![("x", true)]),
                    // Annotated but return type contains Any (not type-complete)
                    make_function("has_any", true, false, vec![("x", true)]),
                ],
                classes: vec![],
                suppressions: vec![],
                annotation_completeness: 100.0,
                type_completeness: 50.0,
            },
        );
        let summary = ReportArgs::calculate_summary(&with_any);
        assert_eq!(summary.fully_annotated_functions, 2);
        assert_eq!(summary.type_complete_functions, 1);
        assert_eq!(summary.aggregate_annotation_completeness, 100.0);
        // Type completeness denominator is fully_annotated_functions (2), not total_functions
        assert_eq!(summary.aggregate_type_completeness, 50.0);
    }
}
