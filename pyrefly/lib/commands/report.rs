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
use dupe::Dupe;
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
            let BindingExport(binding) = bindings.get(idx);
            let location = match exports.get(name) {
                Some(ExportLocation::ThisModule(export)) => {
                    Self::range_to_location(module, export.location)
                }
                _ => continue,
            };
            match binding {
                Binding::AnnotatedType(annot_idx, _inner) => {
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
                Binding::Forward(idx) => match bindings.get(*idx) {
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
                _ => {}
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
        bindings: Bindings,
        answers: Arc<Answers>,
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
                        if !Self::is_function_completely_annotated(&bindings, &fun.def) {
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
                    for field_name in ancestor_class.fields() {
                        let field_name_str = field_name.to_string();
                        // Skip if we already have this attribute listed (it has been overridden by the current class or another class in the MRO)
                        if incomplete_attributes
                            .iter()
                            .any(|a| a.name == field_name_str)
                        {
                            continue;
                        }
                        if !ancestor_class.is_field_annotated(field_name) {
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
        transaction.run(handles.as_slice(), Require::Everything);

        let shadowed = if prefer_stubs {
            Self::py_paths_shadowed_by_pyi(&handles)
        } else {
            HashSet::new()
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
                let functions = Self::parse_functions(&module, &bindings, &answers, &exports);
                let classes = Self::parse_classes(&module, bindings.dupe(), answers.dupe());
                let variables =
                    Self::parse_variables(&module, &bindings, &exports, &functions, &classes);
                let suppressions = Self::parse_suppressions(&module);

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
    use std::sync::Arc;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;

    use super::*;
    use crate::state::require::Require;
    use crate::test::util::TestEnv;

    /// Helper to create a module from source code for testing
    fn create_test_module(code: &str) -> Module {
        Module::new(
            ModuleName::from_str("test"),
            ModulePath::memory(PathBuf::from("test.py")),
            Arc::new(code.to_owned()),
        )
    }

    #[test]
    fn test_parse_suppressions() {
        let code = r#"
x = 1  # pyrefly: ignore[error-code]
y = 2
z = 3  # pyrefly: ignore[code1, code2]
"#;
        let module = create_test_module(code);
        let suppressions = ReportArgs::parse_suppressions(&module);

        assert_eq!(suppressions.len(), 2);

        // Suppression with single error code
        assert_eq!(suppressions[0].kind, "ignore");
        assert_eq!(suppressions[0].codes, vec!["error-code"]);
        assert_eq!(suppressions[0].location.line, 2);

        // Suppression with multiple error codes
        assert_eq!(suppressions[1].kind, "ignore");
        assert_eq!(suppressions[1].codes, vec!["code1", "code2"]);
        assert_eq!(suppressions[1].location.line, 4);
    }

    #[test]
    fn test_parse_variables() {
        let code = r#"
from typing import Callable, TypeVar
from typing import List as MyList

T = TypeVar("T")
x = 42
y: Callable[[int], int] = lambda n: n
z: str = "hello"

def some_func() -> None:
    pass

class SomeClass:
    my_field = 42
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);
        let classes = ReportArgs::parse_classes(&module, bindings.dupe(), answers);
        let variables =
            ReportArgs::parse_variables(&module, &bindings, &exports, &functions, &classes);
        // T (line 5), x (line 6), y (line 7), z (line 8)
        assert_eq!(variables.len(), 4, "should have 4 variables: T, x, y, z");

        // T (TypeVar) on line 5
        assert_eq!(variables[0].name, "test.T");
        assert_eq!(variables[0].annotation, None);
        assert_eq!(variables[0].location.line, 5, "T should be on line 5");

        // x has no annotation on line 6
        assert_eq!(variables[1].name, "test.x");
        assert_eq!(variables[1].annotation, None);
        assert_eq!(variables[1].location.line, 6, "x should be on line 6");

        // y has a Callable[[int], int] annotation on line 7
        assert_eq!(variables[2].name, "test.y");
        assert_eq!(
            variables[2].annotation,
            Some("Callable[[int], int]".to_owned())
        );
        assert_eq!(variables[2].location.line, 7, "y should be on line 7");

        // z has a str annotation on line 8
        assert_eq!(variables[3].name, "test.z");
        assert_eq!(variables[3].annotation, Some("str".to_owned()));
        assert_eq!(variables[3].location.line, 8, "z should be on line 8");

        // Functions and classes should NOT appear as variables
        assert!(
            !variables.iter().any(|v| v.name.contains("some_func")),
            "functions should not be reported as variables"
        );
        assert!(
            !variables.iter().any(|v| v.name.contains("SomeClass")),
            "classes should not be reported as variables"
        );
        assert!(
            !variables.iter().any(|v| v.name.contains("Callable")),
            "we should not report re-exported symbols as variables"
        );
        assert!(
            !variables.iter().any(|v| v.name.contains("MyList")),
            "we should not report re-exported symbols as variables"
        );
        assert!(
            !variables.iter().any(|v| v.name.contains("my_field")),
            "we should not report class fields as variables"
        );
    }

    #[test]
    fn test_parse_functions() {
        let code = r#"
def foo(x: int, y: str) -> bool:
    return True
def foo_unannotated(x, y):
    return True
class C:
    def bar(self, x: int, y: str) -> bool:
        return True
    class Inner:
        def baz(self, x: int, y: str) -> bool:
            return True
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);

        assert_eq!(functions.len(), 4);

        // functions[0]: foo - fully annotated top-level function
        let foo = &functions[0];
        assert_eq!(foo.name, "test.foo");
        assert_eq!(foo.return_annotation, Some("bool".to_owned()));
        assert_eq!(foo.parameters.len(), 2);
        assert_eq!(foo.parameters[0].name, "x");
        assert_eq!(foo.parameters[0].annotation, Some("int".to_owned()));
        assert_eq!(foo.parameters[1].name, "y");
        assert_eq!(foo.parameters[1].annotation, Some("str".to_owned()));

        // functions[1]: foo_unannotated - no annotations
        let foo_unannotated = &functions[1];
        assert_eq!(foo_unannotated.name, "test.foo_unannotated");
        assert_eq!(foo_unannotated.return_annotation, None);
        assert_eq!(foo_unannotated.parameters.len(), 2);
        assert_eq!(foo_unannotated.parameters[0].name, "x");
        assert_eq!(foo_unannotated.parameters[0].annotation, None);
        assert_eq!(foo_unannotated.parameters[1].name, "y");
        assert_eq!(foo_unannotated.parameters[1].annotation, None);

        // functions[2]: C.bar - method in class C
        let c_bar = &functions[2];
        assert_eq!(c_bar.name, "test.C.bar");
        assert_eq!(c_bar.return_annotation, Some("bool".to_owned()));
        assert_eq!(c_bar.parameters.len(), 3);
        assert_eq!(c_bar.parameters[0].name, "self");
        assert_eq!(c_bar.parameters[0].annotation, None);
        assert_eq!(c_bar.parameters[1].name, "x");
        assert_eq!(c_bar.parameters[1].annotation, Some("int".to_owned()));
        assert_eq!(c_bar.parameters[2].name, "y");
        assert_eq!(c_bar.parameters[2].annotation, Some("str".to_owned()));

        // functions[3]: C.Inner.baz - method in nested class Inner
        let inner_baz = &functions[3];
        assert_eq!(inner_baz.name, "test.C.Inner.baz");
        assert_eq!(inner_baz.return_annotation, Some("bool".to_owned()));
        assert_eq!(inner_baz.parameters.len(), 3);
        assert_eq!(inner_baz.parameters[0].name, "self");
        assert_eq!(inner_baz.parameters[0].annotation, None);
        assert_eq!(inner_baz.parameters[1].name, "x");
        assert_eq!(inner_baz.parameters[1].annotation, Some("int".to_owned()));
        assert_eq!(inner_baz.parameters[2].name, "y");
        assert_eq!(inner_baz.parameters[2].annotation, Some("str".to_owned()));
    }

    #[test]
    fn test_unknown_module() {
        let code = r#"
def foo():
    pass
"#;
        let module = Module::new(
            ModuleName::unknown(),
            ModulePath::memory(PathBuf::from("test.py")),
            Arc::new(code.to_owned()),
        );
        let (state, handle_fn) = TestEnv::one("__unknown__", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("__unknown__");
        let transaction = state.transaction();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);

        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0].name, "foo");
    }

    #[test]
    fn test_parse_classes_with_incomplete_methods() {
        let code = r#"
class Complete:
    def method(self, x: int) -> bool:
        return True

class Incomplete:
    def method_unannotated(self, x):
        pass
    def method_partial(self, x: int):
        pass
    def method_complete(self, x: int) -> bool:
        return True
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();

        let classes = ReportArgs::parse_classes(&module, bindings, answers);

        assert_eq!(classes.len(), 2);

        assert_eq!(classes[0].name, "test.Complete");
        assert_eq!(classes[0].incomplete_attributes.len(), 0);

        assert_eq!(classes[1].name, "test.Incomplete");
        assert_eq!(classes[1].incomplete_attributes.len(), 2);
        assert_eq!(
            classes[1].incomplete_attributes[0].name.as_str(),
            "method_partial"
        );
        assert_eq!(
            classes[1].incomplete_attributes[1].name.as_str(),
            "method_unannotated"
        );
        for attr in &classes[1].incomplete_attributes {
            assert_eq!(attr.declared_in, "test.Incomplete");
        }
    }

    #[test]
    fn test_parse_classes_nested() {
        let code = r#"
class Outer:
    def method(self, x: int) -> bool:
        return True

    class Inner:
        def inner_method(self, x):
            pass
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();

        let classes = ReportArgs::parse_classes(&module, bindings, answers);
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].name, "test.Outer");
        assert_eq!(classes[0].incomplete_attributes.len(), 0);
        assert_eq!(classes[1].name, "test.Outer.Inner");
        assert_eq!(classes[1].incomplete_attributes.len(), 1);
        assert_eq!(classes[1].incomplete_attributes[0].name, "inner_method");
    }

    #[test]
    fn test_parse_classes_inheritance() {
        let code = r#"
class Base:
    def base_method(self, x):
        pass

class Child(Base):
    def child_method(self, x: int) -> bool:
        return True
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();

        let classes = ReportArgs::parse_classes(&module, bindings, answers);

        // Both Base and Child should be reported
        // Base has base_method incomplete
        // Child inherits base_method from Base
        assert!(!classes.is_empty());

        // Find Base class
        let base = classes.iter().find(|c| c.name == "test.Base");
        assert!(base.is_some(), "Base class should be reported");
        let base = base.unwrap();
        assert_eq!(base.incomplete_attributes.len(), 1);
        assert_eq!(base.incomplete_attributes[0].name, "base_method");
        assert_eq!(base.incomplete_attributes[0].declared_in, "test.Base");
    }

    #[test]
    fn test_parse_functions_excludes_nested_in_functions() {
        let code = r#"
def outer() -> None:
    def inner() -> None:
        pass
    def inner2(x: int) -> str:
        return str(x)

def top_level(x: int) -> bool:
    return True
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);

        // Only top-level functions should be reported; inner and inner2 are nested
        // inside outer and are not public symbols.
        let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        assert!(
            names.contains(&"test.outer"),
            "top-level function 'outer' should be reported, got: {names:?}"
        );
        assert!(
            names.contains(&"test.top_level"),
            "top-level function 'top_level' should be reported, got: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("inner")),
            "nested functions should not be reported, got: {names:?}"
        );
        assert_eq!(
            functions.len(),
            2,
            "only 2 top-level functions expected, got: {names:?}"
        );
    }

    #[test]
    fn test_parse_functions_excludes_methods_of_classes_nested_in_functions() {
        let code = r#"
def outer() -> None:
    class LocalClass:
        def method(self) -> None:
            pass

class TopLevel:
    def method(self) -> None:
        pass
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();
        let exports = transaction.get_exports(&handle);

        let functions = ReportArgs::parse_functions(&module, &bindings, &answers, &exports);
        // LocalClass.method is inside a function and is not a public symbol.
        let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        assert!(
            names.contains(&"test.outer"),
            "top-level function 'outer' should be reported, got: {names:?}"
        );
        assert!(
            names.contains(&"test.TopLevel.method"),
            "method of top-level class should be reported, got: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("LocalClass")),
            "methods of classes nested in functions should not be reported, got: {names:?}"
        );
    }

    #[test]
    fn test_parse_classes_excludes_nested_in_functions() {
        let code = r#"
def outer() -> None:
    class LocalClass:
        def method(self) -> None:
            pass

class TopLevel:
    def method(self) -> None:
        pass
"#;
        let (state, handle_fn) = TestEnv::one("test", code)
            .with_default_require_level(Require::Everything)
            .to_state();
        let handle = handle_fn("test");
        let transaction = state.transaction();

        let module = transaction.get_module_info(&handle).unwrap();
        let bindings = transaction.get_bindings(&handle).unwrap();
        let answers = transaction.get_answers(&handle).unwrap();

        let classes = ReportArgs::parse_classes(&module, bindings, answers);

        // Only TopLevel should be reported; LocalClass is nested inside a function
        // and is not a public symbol.
        let names: Vec<&str> = classes.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.contains(&"test.TopLevel"),
            "top-level class should be reported, got: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("LocalClass")),
            "classes nested in functions should not be reported, got: {names:?}"
        );
        assert_eq!(
            classes.len(),
            1,
            "only 1 top-level class expected, got: {names:?}"
        );
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
