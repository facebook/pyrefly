/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod ast_visitor;
pub mod call_graph;
pub mod captured_variable;
pub mod class;
pub mod collect;
pub mod context;
pub mod function;
pub mod global_variable;
pub mod is_test_module;
pub mod location;
pub mod module;
pub mod module_index;
pub mod override_graph;
pub mod scope;
pub mod step_logger;
pub mod type_of_expression;
pub mod types;

use core::panic;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Not;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use serde::Serialize;

use crate::error::error::Error as TypeError;
use crate::module::bundled::BundledStub;
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;
use crate::report::pysa::call_graph::CallGraph;
use crate::report::pysa::call_graph::ExpressionIdentifier;
use crate::report::pysa::call_graph::export_call_graphs;
use crate::report::pysa::captured_variable::ModuleCapturedVariables;
use crate::report::pysa::captured_variable::collect_captured_variables_for_module;
use crate::report::pysa::captured_variable::export_captured_variables_for_module;
use crate::report::pysa::class::ClassDefinition;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::export_all_classes;
use crate::report::pysa::collect::CollectNoDuplicateKeys;
use crate::report::pysa::context::ModuleAnswersContext;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::context::PysaResolver;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::ModuleFunctionDefinitions;
use crate::report::pysa::function::export_all_functions;
use crate::report::pysa::function::export_function_definitions;
use crate::report::pysa::global_variable::GlobalVariable;
use crate::report::pysa::global_variable::ModuleGlobalVariables;
use crate::report::pysa::global_variable::collect_global_variables_for_module;
use crate::report::pysa::global_variable::export_global_variables;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module_index::PysaModuleIndex;
use crate::report::pysa::override_graph::ModuleReversedOverrideGraph;
use crate::report::pysa::override_graph::create_reversed_override_graph_for_module;
use crate::report::pysa::step_logger::StepLogger;
use crate::report::pysa::type_of_expression::export_type_of_expressions;
use crate::report::pysa::types::PysaType;
use crate::state::state::Transaction;

#[derive(Debug, Clone, Serialize)]
struct PysaProjectModule {
    module_id: ModuleId,
    module_name: ModuleName,        // e.g, `foo.bar`
    source_path: ModulePathDetails, // Path to the source code
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_source_path: Option<PathBuf>, // Path relative to a root or search path
    info_filename: Option<PathBuf>, // Filename for info files
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_test: bool, // Uses a set of heuristics to determine if the module is a test file.
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_interface: bool, // Is this a .pyi file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_init: bool, // Is this a __init__.py(i) file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_internal: bool, // Is this a module from the project (as opposed to a dependency)?
}

/// Format of the index file `pyrefly.pysa.json`
#[derive(Debug, Clone, Serialize)]
struct PysaProjectFile {
    format_version: u32,
    modules: HashMap<ModuleId, PysaProjectModule>,
    builtin_module_id: ModuleId,
    object_class_id: ClassId,
    dict_class_id: ClassId,
    typing_module_id: ModuleId,
    typing_mapping_class_id: ClassId,
}

/// Format of the file `definitions/my.module:id.json` containing all definitions
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleDefinitions {
    format_version: u32,
    module_id: ModuleId,
    module_name: ModuleName,
    source_path: ModulePathDetails,
    function_definitions: ModuleFunctionDefinitions<FunctionDefinition>,
    class_definitions: HashMap<PysaLocation, ClassDefinition>,
    global_variables: HashMap<Name, GlobalVariable>,
}

/// Format of the file `type_of_expressions/my.module:id.json` containing type of expressions
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleTypeOfExpressions {
    format_version: u32,
    module_id: ModuleId,
    module_name: ModuleName,
    source_path: ModulePathDetails,
    type_of_expression: HashMap<PysaLocation, PysaType>,
}

/// Format of the file `call_graphs/my.module:id.json` containing module call graphs
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleCallGraphs {
    format_version: u32,
    module_id: ModuleId,
    module_name: ModuleName,
    source_path: ModulePathDetails,
    call_graphs: HashMap<FunctionId, CallGraph<ExpressionIdentifier, FunctionRef>>,
}

/// Per-module intermediate information required by Pysa for its report step.
/// Stored as `Arc<PysaSolutions>` inside pyrefly `Solutions` when pysa reporting is enabled.
pub struct PysaSolutions {
    pub module_id: ModuleId,
    pub module_index: PysaModuleIndex,
    pub function_base_definitions: ModuleFunctionDefinitions<FunctionBaseDefinition>,
    pub global_variables: ModuleGlobalVariables,
    pub is_test_module: bool,
}

impl std::fmt::Debug for PysaSolutions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PysaSolutions")
            .field("module_id", &self.module_id)
            .field("is_test_module", &self.is_test_module)
            .finish_non_exhaustive()
    }
}

impl PysaSolutions {
    /// Build per-module intermediate information required by Pysa for its report step.
    /// This only depends on non-cross-module information, currently represented as `ModuleAnswersContext`.
    pub fn build(context: &ModuleAnswersContext) -> Arc<Self> {
        let module_index = PysaModuleIndex::build(context);
        let global_variables = collect_global_variables_for_module(context);
        let function_base_definitions = export_all_functions(context);
        let is_test_module = is_test_module::is_test_module(context);

        Arc::new(Self {
            module_id: context.module_id,
            module_index,
            function_base_definitions,
            global_variables,
            is_test_module,
        })
    }
}

/// Marker stored in `Transaction` to indicate that Pysa reporting is in progress.
pub struct PysaReporter {
    pub module_ids: ModuleIds,
    pub pysa_directory: PathBuf,
    pub definitions_directory: PathBuf,
    pub type_of_expressions_directory: PathBuf,
    pub call_graphs_directory: PathBuf,
}

impl PysaReporter {
    /// Create a new PysaReporter, setting up report directories.
    pub fn new(pysa_directory: &Path, handles: &[Handle]) -> anyhow::Result<Box<Self>> {
        tracing::info!("Writing pysa results to `{}`", pysa_directory.display());

        pyrefly_util::fs_anyhow::create_dir_all(pysa_directory)?;
        let definitions_directory = pysa_directory.join("definitions");
        let type_of_expressions_directory = pysa_directory.join("type_of_expressions");
        let call_graphs_directory = pysa_directory.join("call_graphs");
        pyrefly_util::fs_anyhow::create_dir_all(&definitions_directory)?;
        pyrefly_util::fs_anyhow::create_dir_all(&type_of_expressions_directory)?;
        pyrefly_util::fs_anyhow::create_dir_all(&call_graphs_directory)?;

        let module_ids = ModuleIds::new(handles);

        Ok(Box::new(Self {
            module_ids,
            pysa_directory: pysa_directory.to_path_buf(),
            definitions_directory,
            type_of_expressions_directory,
            call_graphs_directory,
        }))
    }

    /// Write JSON files about the current module/handle.
    ///
    /// This can perform cross-module lookups using the `transaction` (wrapped in `PysaResolver`).
    pub fn report_module(&self, handle: &Handle, transaction: &Transaction) {
        let info_filename = match handle.path().details() {
            ModulePathDetails::Namespace(_) => None,
            _ => Some(PathBuf::from(format!(
                "{}:{}.json",
                String::from_iter(
                    handle
                        .module()
                        .to_string()
                        .chars()
                        .filter(|c| c.is_ascii())
                        .take(220)
                ),
                self.module_ids.get_from_handle(handle).to_int()
            ))),
        };

        if let Some(info_filename) = &info_filename {
            let resolver = PysaResolver::new(transaction, &self.module_ids, handle.dupe());
            let context = ModuleContext {
                answers_context: ModuleAnswersContext::create(
                    handle.dupe(),
                    transaction,
                    &self.module_ids,
                ),
                resolver: &resolver,
            };

            let captured_variables = collect_captured_variables_for_module(&context);
            let reversed_override_graph = create_reversed_override_graph_for_module(&context);

            let module_definitions =
                export_module_definitions(&context, &captured_variables, &reversed_override_graph);
            let writer = BufWriter::new(
                File::create(self.definitions_directory.join(info_filename))
                    .expect("Failed to create definitions file"),
            );
            serde_json::to_writer(writer, &module_definitions)
                .expect("Failed to write definitions file");

            let module_type_of_expressions = export_module_type_of_expressions(&context);
            let writer = BufWriter::new(
                File::create(self.type_of_expressions_directory.join(info_filename))
                    .expect("Failed to create type_of_expressions file"),
            );
            serde_json::to_writer(writer, &module_type_of_expressions)
                .expect("Failed to write type_of_expressions file");

            let module_call_graphs = export_module_call_graphs(&context, &captured_variables);
            let writer = BufWriter::new(
                File::create(self.call_graphs_directory.join(info_filename))
                    .expect("Failed to create call_graphs file"),
            );
            serde_json::to_writer(writer, &module_call_graphs)
                .expect("Failed to write call_graphs file");
        }
    }
}

pub fn export_module_definitions(
    context: &ModuleContext,
    captured_variables: &ModuleCapturedVariables<FunctionRef>,
    reversed_override_graph: &ModuleReversedOverrideGraph,
) -> PysaModuleDefinitions {
    let global_variables_exported = export_global_variables(
        &context.resolver.current_module_solutions().global_variables,
        context,
    );
    let class_definitions = export_all_classes(context);
    let captured_variables = export_captured_variables_for_module(captured_variables);
    let function_definitions =
        export_function_definitions(&captured_variables, reversed_override_graph, context);
    PysaModuleDefinitions {
        format_version: 1,
        module_id: context.answers_context.module_id,
        module_name: context.answers_context.module_info.name(),
        source_path: context.answers_context.module_info.path().details().clone(),
        function_definitions,
        class_definitions,
        global_variables: global_variables_exported,
    }
}

pub fn export_module_type_of_expressions(context: &ModuleContext) -> PysaModuleTypeOfExpressions {
    let type_of_expression = export_type_of_expressions(context);
    PysaModuleTypeOfExpressions {
        format_version: 1,
        module_id: context.answers_context.module_id,
        module_name: context.answers_context.module_info.name(),
        source_path: context.answers_context.module_info.path().details().clone(),
        type_of_expression,
    }
}

pub fn export_module_call_graphs(
    context: &ModuleContext,
    captured_variables: &ModuleCapturedVariables<FunctionRef>,
) -> PysaModuleCallGraphs {
    let call_graphs = export_call_graphs(context, captured_variables)
        .into_iter()
        .map(|(function_ref, call_graph)| (function_ref.function_id, call_graph))
        .collect_no_duplicate_keys()
        .expect("Found multiple call graphs for the same function");
    PysaModuleCallGraphs {
        format_version: 1,
        module_id: context.answers_context.module_id,
        module_name: context.answers_context.module_info.name(),
        source_path: context.answers_context.module_info.path().details().clone(),
        call_graphs,
    }
}

fn build_module_mapping(
    handles: &Vec<Handle>,
    project_handles: &[Handle],
    module_ids: &ModuleIds,
    transaction: &Transaction,
) -> HashMap<ModuleId, PysaProjectModule> {
    let step = StepLogger::start("Building module list", "Built module list");

    // Set of handles from the "project-includes", i.e only handles that are typed checked.
    let project_handles: HashSet<&Handle> = project_handles.iter().collect();

    let mut project_modules = HashMap::new();
    for handle in handles {
        let module_id = module_ids.get_from_handle(handle);

        // Path where we will store the information on the module.
        let info_filename = match handle.path().details() {
            ModulePathDetails::Namespace(_) => {
                // Indicates a directory that contains a `__init__.py` file.
                None
            }
            _ => {
                Some(PathBuf::from(format!(
                    "{}:{}.json",
                    // Filename must be less than 255 bytes
                    String::from_iter(
                        handle
                            .module()
                            .to_string()
                            .chars()
                            .filter(|c| c.is_ascii())
                            .take(220)
                    ),
                    module_id.to_int()
                )))
            }
        };

        let module_name = handle.module();
        let module_path = handle.path();
        let relative_source_path = match module_path.details() {
            ModulePathDetails::FileSystem(path) | ModulePathDetails::Namespace(path) => module_path
                .root_of(module_name)
                .and_then(|root| path.as_path().strip_prefix(root).ok())
                .map(|path| path.to_path_buf()),
            ModulePathDetails::Memory(_) => None,
            ModulePathDetails::BundledTypeshed(relative_path)
            | ModulePathDetails::BundledTypeshedThirdParty(relative_path)
            | ModulePathDetails::BundledThirdParty(relative_path) => {
                Some(relative_path.to_path_buf())
            }
        };

        assert!(
            project_modules
                .insert(
                    module_id,
                    PysaProjectModule {
                        module_id,
                        module_name,
                        source_path: module_path.details().clone(),
                        relative_source_path,
                        info_filename: info_filename.clone(),
                        is_test: transaction
                            .get_solutions(handle)
                            .expect("missing solutions")
                            .pysa_solutions()
                            .expect("missing pysa solutions")
                            .is_test_module,
                        is_interface: handle.path().is_interface(),
                        is_init: handle.path().is_init(),
                        is_internal: project_handles.contains(handle),
                    }
                )
                .is_none(),
            "Found multiple handles with the same module id"
        );
    }

    step.finish();
    project_modules
}

fn write_bundle_stubs(bundle: &impl BundledStub, directory: &Path) -> anyhow::Result<()> {
    for module in bundle.modules() {
        let module_path = bundle.find(module).unwrap();
        let relative_path = match module_path.details() {
            ModulePathDetails::BundledTypeshed(path) => &**path,
            ModulePathDetails::BundledTypeshedThirdParty(path) => &**path,
            _ => panic!("unexpected module path for typeshed module"),
        };
        let content = bundle.load(relative_path).unwrap();
        let target_path = directory.join(relative_path);
        fs_anyhow::create_dir_all(target_path.parent().unwrap())?;
        fs_anyhow::write(&target_path, content.as_bytes())?;
    }

    Ok(())
}

// Dump all typeshed files, so we can parse them.
fn write_typeshed_files(results_directory: &Path) -> anyhow::Result<()> {
    let step = StepLogger::start("Exporting typeshed files", "Exported typeshed files");

    let typeshed = typeshed()?;
    write_bundle_stubs(typeshed, &results_directory.join("typeshed"))?;

    if let Ok(typeshed_third_party) = typeshed_third_party() {
        write_bundle_stubs(
            typeshed_third_party,
            &results_directory.join("typeshed_third_party"),
        )?;
    }

    step.finish();
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct PysaTypeError {
    module_id: ModuleId,
    location: PysaLocation,
    kind: pyrefly_config::error_kind::ErrorKind,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct PysaTypeErrorsFile {
    format_version: u32,
    errors: Vec<PysaTypeError>,
}

fn write_errors_file(
    results_directory: &Path,
    errors: &[TypeError],
    module_ids: &ModuleIds,
) -> anyhow::Result<()> {
    let step = StepLogger::start("Exporting type errors", "Exported type errors");

    let writer = BufWriter::new(File::create(results_directory.join("errors.json"))?);
    serde_json::to_writer(
        writer,
        &PysaTypeErrorsFile {
            format_version: 1,
            errors: errors
                .iter()
                .map(|error| PysaTypeError {
                    module_id: module_ids.get_from_module(error.module()),
                    location: PysaLocation::from_text_range(error.range(), error.module()),
                    kind: error.error_kind(),
                    message: error.msg(),
                })
                .collect::<Vec<_>>(),
        },
    )?;

    step.finish();
    Ok(())
}

/// Write the project-level pysa files after inline extraction.
///
/// Per-module JSON files (definitions, type_of_expressions, call_graphs) are
/// already written by `PysaReporter::report_module` during type checking.
/// This function writes the remaining project-level files:
/// module mapping, typeshed files, errors, and `pyrefly.pysa.json`.
pub fn write_project_file(
    pysa_reporter: &PysaReporter,
    transaction: &Transaction,
    project_handles: &[Handle],
    errors: &[TypeError],
) -> anyhow::Result<()> {
    let results_directory = &pysa_reporter.pysa_directory;

    write_typeshed_files(results_directory)?;
    write_errors_file(results_directory, errors, &pysa_reporter.module_ids)?;

    let step = StepLogger::start(
        &format!(
            "Writing `{}`",
            results_directory.join("pyrefly.pysa.json").display()
        ),
        &format!(
            "Wrote `{}`",
            results_directory.join("pyrefly.pysa.json").display()
        ),
    );

    let handles = transaction.handles();
    let project_modules = build_module_mapping(
        &handles,
        project_handles,
        &pysa_reporter.module_ids,
        transaction,
    );

    let builtin_module = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "builtins")
        .exactly_one()
        .expect("expected exactly one builtins module");
    let typing_module = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "typing")
        .exactly_one()
        .expect("expected exactly one typing module");
    let object_class_id = ClassId::from_class(
        transaction
            .get_stdlib(builtin_module)
            .object()
            .class_object(),
    );
    let dict_class_id = ClassId::from_class(transaction.get_stdlib(builtin_module).dict_object());
    let typing_mapping_class_id =
        ClassId::from_class(transaction.get_stdlib(typing_module).mapping_object());

    let writer = BufWriter::new(File::create(results_directory.join("pyrefly.pysa.json"))?);
    serde_json::to_writer(
        writer,
        &PysaProjectFile {
            format_version: 1,
            modules: project_modules,
            builtin_module_id: pysa_reporter.module_ids.get_from_handle(builtin_module),
            object_class_id,
            dict_class_id,
            typing_module_id: pysa_reporter.module_ids.get_from_handle(typing_module),
            typing_mapping_class_id,
        },
    )?;

    step.finish();
    Ok(())
}
