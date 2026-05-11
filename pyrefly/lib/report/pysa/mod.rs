/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod ast_visitor;
pub mod call_graph;
pub mod capnp_writer;
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
#[allow(clippy::all)]
pub mod pysa_report_capnp;
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
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_util::fs_anyhow;
use pyrefly_util::interned_path::InternedPath;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use serde::Serialize;

use crate::error::error::Error as TypeError;
use crate::module::bundled::BundledStub;
use crate::module::third_party::get_bundled_third_party;
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
use crate::report::pysa::class::ClassRef;
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

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum PysaFormat {
    Capnp,
    Json,
}

#[derive(Debug, Clone, Serialize)]
pub struct PysaProjectModule {
    pub module_id: ModuleId,
    pub module_name: ModuleName,        // e.g, `foo.bar`
    pub source_path: ModulePathDetails, // Path to the source code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_source_path: Option<PathBuf>, // Path relative to a root or search path
    pub info_filename: Option<PathBuf>, // Filename for info files
    pub python_version: PythonVersion,
    pub platform: PythonPlatform,
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_test: bool, // Uses a set of heuristics to determine if the module is a test file.
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_interface: bool, // Is this a .pyi file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_init: bool, // Is this a __init__.py(i) file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    pub is_internal: bool, // Is this a module from the project (as opposed to a dependency)?
}

/// Format of the index file `pyrefly.pysa.json`
#[derive(Debug, Clone, Serialize)]
pub struct PysaProjectFile {
    pub format_version: u32,
    pub modules: HashMap<ModuleId, PysaProjectModule>,
    pub builtin_module_ids: Vec<ModuleId>,
    pub object_class_refs: Vec<ClassRef>,
    pub dict_class_refs: Vec<ClassRef>,
    pub typing_module_ids: Vec<ModuleId>,
    pub typing_mapping_class_refs: Vec<ClassRef>,
}

/// Format of the file `definitions/my.module:id.json` containing all definitions
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleDefinitions {
    pub format_version: u32,
    pub module_id: ModuleId,
    pub module_name: ModuleName,
    pub source_path: ModulePathDetails,
    pub function_definitions: ModuleFunctionDefinitions<FunctionDefinition>,
    pub class_definitions: HashMap<ClassId, ClassDefinition>,
    pub global_variables: HashMap<Name, GlobalVariable>,
}

/// Type identifier within a function's deduplicated type table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct LocalTypeId(pub u32);

/// Per-function type-of-expression data, with deduplicated types.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionTypeOfExpressions {
    /// Deduplicated type table. `LocalTypeId(n)` refers to `type_table[n]`.
    pub type_table: Vec<PysaType>,
    /// Map from expression location to its LocalTypeId in the type table.
    pub locations: HashMap<PysaLocation, LocalTypeId>,
}

/// Format of the file `type_of_expressions/my.module:id.json` containing type of expressions
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleTypeOfExpressions {
    pub format_version: u32,
    pub module_id: ModuleId,
    pub module_name: ModuleName,
    pub source_path: ModulePathDetails,
    pub functions: HashMap<FunctionId, FunctionTypeOfExpressions>,
}

/// Format of the file `call_graphs/my.module:id.json` containing module call graphs
#[derive(Debug, Clone, Serialize)]
pub struct PysaModuleCallGraphs {
    pub format_version: u32,
    pub module_id: ModuleId,
    pub module_name: ModuleName,
    pub source_path: ModulePathDetails,
    pub call_graphs: HashMap<FunctionId, CallGraph<ExpressionIdentifier, FunctionRef>>,
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
    pub format: PysaFormat,
}

impl PysaReporter {
    /// Create a new PysaReporter, setting up report directories.
    pub fn new(
        pysa_directory: &Path,
        handles: &[Handle],
        format: PysaFormat,
    ) -> anyhow::Result<Box<Self>> {
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
            format,
        }))
    }

    fn file_extension(&self) -> &str {
        match self.format {
            PysaFormat::Json => "json",
            PysaFormat::Capnp => "capnp.bin",
        }
    }

    /// Write output files about the current module/handle.
    ///
    /// This can perform cross-module lookups using the `transaction` (wrapped in `PysaResolver`).
    pub fn report_module(&self, handle: &Handle, transaction: &Transaction) {
        let info_filename = match handle.path().details() {
            ModulePathDetails::Namespace(_) => None,
            _ => Some(PathBuf::from(format!(
                "{}:{}.{}",
                String::from_iter(
                    handle
                        .module()
                        .to_string()
                        .chars()
                        .filter(|c| c.is_ascii())
                        .take(220)
                ),
                self.module_ids.get_from_handle(handle).to_int(),
                self.file_extension()
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
            match self.format {
                PysaFormat::Json => serde_json::to_writer(writer, &module_definitions)
                    .expect("Failed to write definitions file"),
                PysaFormat::Capnp => capnp_writer::write_definitions(writer, &module_definitions)
                    .expect("Failed to write definitions file"),
            }

            let module_type_of_expressions = export_module_type_of_expressions(&context);
            let writer = BufWriter::new(
                File::create(self.type_of_expressions_directory.join(info_filename))
                    .expect("Failed to create type_of_expressions file"),
            );
            match self.format {
                PysaFormat::Json => serde_json::to_writer(writer, &module_type_of_expressions)
                    .expect("Failed to write type_of_expressions file"),
                PysaFormat::Capnp => {
                    capnp_writer::write_type_of_expressions(writer, &module_type_of_expressions)
                        .expect("Failed to write type_of_expressions file")
                }
            }

            let module_call_graphs = export_module_call_graphs(&context, &captured_variables);
            let writer = BufWriter::new(
                File::create(self.call_graphs_directory.join(info_filename))
                    .expect("Failed to create call_graphs file"),
            );
            match self.format {
                PysaFormat::Json => serde_json::to_writer(writer, &module_call_graphs)
                    .expect("Failed to write call_graphs file"),
                PysaFormat::Capnp => capnp_writer::write_call_graphs(writer, &module_call_graphs)
                    .expect("Failed to write call_graphs file"),
            }
        }
    }
}

/// Make relative paths in `ModulePathDetails` absolute using the current directory.
/// Manifest paths from buck are relative to the project root (because pyrefly
/// might run in RE). Pysa output needs absolute paths.
fn absolutize_source_path(details: &ModulePathDetails) -> ModulePathDetails {
    match details {
        ModulePathDetails::FileSystem(p) if p.as_path().is_relative() => {
            let absolute = std::env::current_dir()
                .expect("current_dir() failed: cannot absolutize relative source path")
                .join(p.as_path());
            ModulePathDetails::FileSystem(InternedPath::new(absolute))
        }
        ModulePathDetails::Namespace(p) if p.as_path().is_relative() => {
            let absolute = std::env::current_dir()
                .expect("current_dir() failed: cannot absolutize relative source path")
                .join(p.as_path());
            ModulePathDetails::Namespace(InternedPath::new(absolute))
        }
        other => other.clone(),
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
        source_path: absolutize_source_path(context.answers_context.module_info.path().details()),
        function_definitions,
        class_definitions,
        global_variables: global_variables_exported,
    }
}

pub fn export_module_type_of_expressions(context: &ModuleContext) -> PysaModuleTypeOfExpressions {
    let functions = export_type_of_expressions(context);
    PysaModuleTypeOfExpressions {
        format_version: 1,
        module_id: context.answers_context.module_id,
        module_name: context.answers_context.module_info.name(),
        source_path: absolutize_source_path(context.answers_context.module_info.path().details()),
        functions,
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
        source_path: absolutize_source_path(context.answers_context.module_info.path().details()),
        call_graphs,
    }
}

fn build_module_mapping(
    handles: &Vec<Handle>,
    project_handles: &[Handle],
    module_ids: &ModuleIds,
    transaction: &Transaction,
    file_extension: &str,
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
                    "{}:{}.{}",
                    // Filename must be less than 255 bytes
                    String::from_iter(
                        handle
                            .module()
                            .to_string()
                            .chars()
                            .filter(|c| c.is_ascii())
                            .take(220)
                    ),
                    module_id.to_int(),
                    file_extension
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
            handle.sys_info().type_checking(),
            "Expected type_checking to be true for handle"
        );
        assert!(
            project_modules
                .insert(
                    module_id,
                    PysaProjectModule {
                        module_id,
                        module_name,
                        source_path: absolutize_source_path(module_path.details()),
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
                        python_version: handle.sys_info().version(),
                        platform: handle.sys_info().platform().clone(),
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
            ModulePathDetails::BundledThirdParty(path) => &**path,
            _ => panic!("unexpected module path for bundled module"),
        };
        let content = bundle.load(relative_path).unwrap();
        let target_path = directory.join(relative_path);
        fs_anyhow::create_dir_all(target_path.parent().unwrap())?;
        fs_anyhow::write(&target_path, content.as_bytes())?;
    }

    Ok(())
}

// Dump all bundled stub files, so we can parse them.
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

    if let Ok(bundled_third_party) = get_bundled_third_party() {
        write_bundle_stubs(bundled_third_party, &results_directory.join("third_party"))?;
    }

    step.finish();
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct PysaTypeError {
    pub module_name: ModuleName,
    pub module_path: ModulePathDetails,
    pub location: PysaLocation,
    pub kind: pyrefly_config::error_kind::ErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PysaTypeErrorsFile {
    pub format_version: u32,
    pub errors: Vec<PysaTypeError>,
}

fn write_errors_file(
    results_directory: &Path,
    errors: &[TypeError],
    format: PysaFormat,
) -> anyhow::Result<()> {
    let step = StepLogger::start("Exporting type errors", "Exported type errors");

    let errors = PysaTypeErrorsFile {
        format_version: 1,
        errors: errors
            .iter()
            .map(|error| PysaTypeError {
                module_name: error.module().name(),
                module_path: error.module().path().details().clone(),
                location: PysaLocation::from_text_range(error.range(), error.module()),
                kind: error.error_kind(),
                message: error.msg(),
            })
            .collect::<Vec<_>>(),
    };

    match format {
        PysaFormat::Json => {
            let writer = BufWriter::new(File::create(results_directory.join("errors.json"))?);
            serde_json::to_writer(writer, &errors)?;
        }
        PysaFormat::Capnp => {
            let writer = BufWriter::new(File::create(results_directory.join("errors.capnp.bin"))?);
            capnp_writer::write_errors(writer, &errors)?;
        }
    }

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

    let format = pysa_reporter.format;
    let file_extension = pysa_reporter.file_extension();

    write_typeshed_files(results_directory)?;
    write_errors_file(results_directory, errors, format)?;

    let project_filename = format!("pyrefly.pysa.{file_extension}");
    let project_filepath = results_directory.join(&project_filename);
    let step = StepLogger::start(
        &format!("Writing `{}`", project_filepath.display(),),
        &format!("Wrote `{}`", project_filepath.display(),),
    );

    let handles = transaction.handles();
    let project_modules = build_module_mapping(
        &handles,
        project_handles,
        &pysa_reporter.module_ids,
        transaction,
        file_extension,
    );

    let builtin_modules = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "builtins")
        .collect::<Vec<_>>();
    let typing_modules = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "typing")
        .collect::<Vec<_>>();
    let builtin_module_ids = builtin_modules
        .iter()
        .map(|handle| pysa_reporter.module_ids.get_from_handle(handle))
        .collect::<Vec<_>>();
    let object_class_refs = builtin_modules
        .iter()
        .map(|handle| {
            let stdlib = transaction.get_stdlib(handle);
            let class = stdlib.object().class_object();
            ClassRef {
                module_id: pysa_reporter.module_ids.get_from_handle(handle),
                class_id: ClassId::from_class(class),
                class: class.clone(),
            }
        })
        .collect::<Vec<_>>();
    let dict_class_refs = builtin_modules
        .iter()
        .map(|handle| {
            let stdlib = transaction.get_stdlib(handle);
            let class = stdlib.dict_object();
            ClassRef {
                module_id: pysa_reporter.module_ids.get_from_handle(handle),
                class_id: ClassId::from_class(class),
                class: class.clone(),
            }
        })
        .collect::<Vec<_>>();
    let typing_module_ids = typing_modules
        .iter()
        .map(|handle| pysa_reporter.module_ids.get_from_handle(handle))
        .collect::<Vec<_>>();
    let typing_mapping_class_refs = typing_modules
        .iter()
        .map(|handle| {
            let stdlib = transaction.get_stdlib(handle);
            let class = stdlib.mapping_object();
            ClassRef {
                module_id: pysa_reporter.module_ids.get_from_handle(handle),
                class_id: ClassId::from_class(class),
                class: class.clone(),
            }
        })
        .collect::<Vec<_>>();

    let project_file = PysaProjectFile {
        format_version: 1,
        modules: project_modules,
        builtin_module_ids,
        object_class_refs,
        dict_class_refs,
        typing_module_ids,
        typing_mapping_class_refs,
    };

    match format {
        PysaFormat::Json => {
            let writer = BufWriter::new(File::create(project_filepath)?);
            serde_json::to_writer(writer, &project_file)?;
        }
        PysaFormat::Capnp => {
            let writer = BufWriter::new(File::create(project_filepath)?);
            capnp_writer::write_project_file(writer, &project_file)?;
        }
    }

    step.finish();
    Ok(())
}
