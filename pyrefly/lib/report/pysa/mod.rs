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
pub mod override_graph;
pub mod scope;
pub mod slow_fun_monitor;
pub mod step_logger;
pub mod type_of_expression;
pub mod types;

use core::panic;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Not;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use itertools::Itertools;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use ruff_python_ast::name::Name;
use serde::Serialize;

use crate::error::error::Error as TypeError;
use crate::module::bundled::BundledStub;
use crate::module::typeshed::typeshed;
use crate::report::pysa::call_graph::CallGraph;
use crate::report::pysa::call_graph::ExpressionIdentifier;
use crate::report::pysa::call_graph::export_call_graphs;
use crate::report::pysa::captured_variable::WholeProgramCapturedVariables;
use crate::report::pysa::captured_variable::collect_captured_variables;
use crate::report::pysa::captured_variable::export_captured_variables_for_module;
use crate::report::pysa::class::ClassDefinition;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::export_all_classes;
use crate::report::pysa::collect::CollectNoDuplicateKeys;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::ModuleFunctionDefinitions;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::function::collect_function_base_definitions;
use crate::report::pysa::function::export_function_definitions;
use crate::report::pysa::global_variable::GlobalVariable;
use crate::report::pysa::global_variable::WholeProgramGlobalVariables;
use crate::report::pysa::global_variable::collect_global_variables;
use crate::report::pysa::global_variable::export_global_variables;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleId;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::module::ModuleKey;
use crate::report::pysa::override_graph::OverrideGraph;
use crate::report::pysa::override_graph::build_reversed_override_graph;
use crate::report::pysa::slow_fun_monitor::slow_fun_monitor_scope;
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
    #[serde(skip_serializing)]
    handle: Handle,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_test: bool, // Uses a set of heuristics to determine if the module is a test file.
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_interface: bool, // Is this a .pyi file?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_init: bool, // Is this a __init__.py(i) file?
}

/// Format of the index file `pyrefly.pysa.json`
#[derive(Debug, Clone, Serialize)]
struct PysaProjectFile {
    format_version: u32,
    modules: HashMap<ModuleId, PysaProjectModule>,
    builtin_module_id: ModuleId,
    object_class_id: ClassId,
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

pub fn export_module_definitions(
    context: &ModuleContext,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    global_variables: &WholeProgramGlobalVariables,
    captured_variables: &WholeProgramCapturedVariables,
) -> PysaModuleDefinitions {
    let global_variables_exported = export_global_variables(global_variables, context);
    let class_definitions = export_all_classes(function_base_definitions, context);
    let captured_variables = export_captured_variables_for_module(captured_variables, context);
    let function_definitions =
        export_function_definitions(function_base_definitions, &captured_variables, context);
    PysaModuleDefinitions {
        format_version: 1,
        module_id: context.module_id,
        module_name: context.module_info.name(),
        source_path: context.module_info.path().details().clone(),
        function_definitions,
        class_definitions,
        global_variables: global_variables_exported,
    }
}

pub fn export_module_type_of_expressions(context: &ModuleContext) -> PysaModuleTypeOfExpressions {
    let type_of_expression = export_type_of_expressions(context);
    PysaModuleTypeOfExpressions {
        format_version: 1,
        module_id: context.module_id,
        module_name: context.module_info.name(),
        source_path: context.module_info.path().details().clone(),
        type_of_expression,
    }
}

pub fn export_module_call_graphs(
    context: &ModuleContext,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    override_graph: &OverrideGraph,
    global_variables: &WholeProgramGlobalVariables,
    captured_variables: &WholeProgramCapturedVariables,
) -> PysaModuleCallGraphs {
    let call_graphs = export_call_graphs(
        context,
        function_base_definitions,
        override_graph,
        global_variables,
        captured_variables,
    )
    .into_iter()
    .map(|(function_ref, call_graph)| (function_ref.function_id, call_graph))
    .collect_no_duplicate_keys()
    .expect("Found multiple call graphs for the same function");
    PysaModuleCallGraphs {
        format_version: 1,
        module_id: context.module_id,
        module_name: context.module_info.name(),
        source_path: context.module_info.path().details().clone(),
        call_graphs,
    }
}

fn build_module_mapping(
    handles: &Vec<Handle>,
    module_ids: &ModuleIds,
) -> HashMap<ModuleId, PysaProjectModule> {
    let step = StepLogger::start("Building module list", "Built module list");

    let mut project_modules = HashMap::new();
    for handle in handles {
        let module_id = module_ids.get(ModuleKey::from_handle(handle)).unwrap();

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
                        handle: handle.clone(),
                        is_test: false,
                        is_interface: handle.path().is_interface(),
                        is_init: handle.path().is_init(),
                    }
                )
                .is_none(),
            "Found multiple handles with the same module id"
        );
    }

    step.finish();
    project_modules
}

fn make_module_work_list(
    project_modules: &HashMap<ModuleId, PysaProjectModule>,
) -> Vec<(Handle, ModuleId, PathBuf)> {
    project_modules
        .iter()
        .filter_map(|(module_id, module)| {
            module
                .info_filename
                .as_ref()
                .map(|info_filename| (module.handle.clone(), *module_id, info_filename.clone()))
        })
        .collect::<Vec<_>>()
}

fn write_module_definitions_files(
    module_work_list: &Vec<(Handle, ModuleId, PathBuf)>,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    global_variables: &WholeProgramGlobalVariables,
    captured_variables: &WholeProgramCapturedVariables,
    transaction: &Transaction,
    module_ids: &ModuleIds,
    definitions_directory: &Path,
) -> anyhow::Result<()> {
    let step = StepLogger::start(
        "Exporting module definitions",
        "Exported module definitions",
    );

    ThreadPool::new().install(|| -> anyhow::Result<()> {
        slow_fun_monitor_scope(|slow_function_monitor| {
            module_work_list.par_iter().try_for_each(
                |(handle, _, info_filename)| -> anyhow::Result<()> {
                    let context =
                        ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                    let module_definitions = slow_function_monitor.monitor_function(
                        || {
                            export_module_definitions(
                                &context,
                                function_base_definitions,
                                global_variables,
                                captured_variables,
                            )
                        },
                        format!(
                            "Exporting module definitions for `{}`",
                            handle.module().as_str()
                        ),
                        /* max_time_in_seconds */ 4,
                    );
                    let writer =
                        BufWriter::new(File::create(definitions_directory.join(info_filename))?);
                    serde_json::to_writer(writer, &module_definitions)?;
                    Ok(())
                },
            )
        })
    })?;

    step.finish();
    Ok(())
}

fn write_module_type_of_expressions_files(
    module_work_list: &Vec<(Handle, ModuleId, PathBuf)>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
    type_of_expressions_directory: &Path,
) -> anyhow::Result<()> {
    let step = StepLogger::start(
        "Exporting type of expressions of modules",
        "Exported type of expressions of modules",
    );

    ThreadPool::new().install(|| -> anyhow::Result<()> {
        slow_fun_monitor_scope(|slow_function_monitor| {
            module_work_list.par_iter().try_for_each(
                |(handle, _, info_filename)| -> anyhow::Result<()> {
                    let context =
                        ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                    let module_type_of_expressions = slow_function_monitor.monitor_function(
                        || export_module_type_of_expressions(&context),
                        format!(
                            "Exporting type of expressions for `{}`",
                            handle.module().as_str()
                        ),
                        /* max_time_in_seconds */ 4,
                    );
                    let writer = BufWriter::new(File::create(
                        type_of_expressions_directory.join(info_filename),
                    )?);
                    serde_json::to_writer(writer, &module_type_of_expressions)?;
                    Ok(())
                },
            )
        })
    })?;

    step.finish();
    Ok(())
}

fn write_module_call_graph_files(
    module_work_list: &Vec<(Handle, ModuleId, PathBuf)>,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    override_graph: &OverrideGraph,
    global_variables: &WholeProgramGlobalVariables,
    captured_variables: &WholeProgramCapturedVariables,
    transaction: &Transaction,
    module_ids: &ModuleIds,
    call_graphs_directory: &Path,
) -> anyhow::Result<()> {
    let step = StepLogger::start(
        "Exporting module call graphs",
        "Exported module call graphs",
    );

    ThreadPool::new().install(|| -> anyhow::Result<()> {
        slow_fun_monitor_scope(|slow_function_monitor| {
            module_work_list.par_iter().try_for_each(
                |(handle, _, info_filename)| -> anyhow::Result<()> {
                    let context =
                        ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                    let module_call_graphs = slow_function_monitor.monitor_function(
                        || {
                            export_module_call_graphs(
                                &context,
                                function_base_definitions,
                                override_graph,
                                global_variables,
                                captured_variables,
                            )
                        },
                        format!("Exporting call graphs for `{}`", handle.module().as_str()),
                        /* max_time_in_seconds */ 4,
                    );
                    let writer =
                        BufWriter::new(File::create(call_graphs_directory.join(info_filename))?);
                    serde_json::to_writer(writer, &module_call_graphs)?;
                    Ok(())
                },
            )
        })
    })?;

    step.finish();
    Ok(())
}

fn add_module_is_test_flags(
    project_modules: HashMap<ModuleId, PysaProjectModule>,
    module_work_list: &Vec<(Handle, ModuleId, PathBuf)>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> anyhow::Result<HashMap<ModuleId, PysaProjectModule>> {
    let step = StepLogger::start("Checking for test modules", "Checked for test modules");

    let project_modules = Arc::new(Mutex::new(project_modules));
    ThreadPool::new().install(|| -> anyhow::Result<()> {
        slow_fun_monitor_scope(|slow_function_monitor| {
            module_work_list.par_iter().try_for_each(
                |(handle, module_id, _)| -> anyhow::Result<()> {
                    let context =
                        ModuleContext::create(handle.clone(), transaction, module_ids).unwrap();
                    slow_function_monitor.monitor_function(
                        || {
                            if is_test_module::is_test_module(&context) {
                                project_modules
                                    .lock()
                                    .unwrap()
                                    .get_mut(module_id)
                                    .unwrap()
                                    .is_test = true;
                            }
                        },
                        format!(
                            "Checking if `{}` is a test module",
                            handle.module().as_str()
                        ),
                        /* max_time_in_seconds */ 4,
                    );
                    Ok(())
                },
            )
        })
    })?;

    step.finish();
    Ok(Arc::into_inner(project_modules)
        .unwrap()
        .into_inner()
        .unwrap())
}

// Dump all typeshed files, so we can parse them.
fn write_typeshed_files(results_directory: &Path) -> anyhow::Result<()> {
    let step = StepLogger::start("Exporting typeshed files", "Exported typeshed files");
    let typeshed = typeshed()?;

    for typeshed_module in typeshed.modules() {
        let module_path = typeshed.find(typeshed_module).unwrap();
        let relative_path = match module_path.details() {
            ModulePathDetails::BundledTypeshed(path) => &**path,
            _ => panic!("unexpected module path for typeshed module"),
        };
        let content = typeshed.load(relative_path).unwrap();
        let target_path = results_directory.join("typeshed").join(relative_path);
        fs_anyhow::create_dir_all(target_path.parent().unwrap())?;
        fs_anyhow::write(&target_path, content.as_bytes())?;
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
                    module_id: module_ids
                        .get(ModuleKey::from_module(error.module()))
                        .unwrap(),
                    location: PysaLocation::new(error.display_range().clone()),
                    kind: error.error_kind(),
                    message: error.msg(),
                })
                .collect::<Vec<_>>(),
        },
    )?;

    step.finish();
    Ok(())
}

pub fn write_results(
    results_directory: &Path,
    transaction: &Transaction,
    errors: &[TypeError],
) -> anyhow::Result<()> {
    let step = StepLogger::start(
        &format!("Writing results to `{}`", results_directory.display()),
        &format!("Wrote results to `{}`", results_directory.display()),
    );

    fs_anyhow::create_dir_all(results_directory)?;
    let definitions_directory = results_directory.join("definitions");
    let type_of_expressions_directory = results_directory.join("type_of_expressions");
    let call_graphs_directory = results_directory.join("call_graphs");
    fs_anyhow::create_dir_all(&definitions_directory)?;
    fs_anyhow::create_dir_all(&type_of_expressions_directory)?;
    fs_anyhow::create_dir_all(&call_graphs_directory)?;

    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);
    let project_modules = build_module_mapping(&handles, &module_ids);
    let module_work_list = make_module_work_list(&project_modules);

    let reversed_override_graph = build_reversed_override_graph(&handles, transaction, &module_ids);
    let function_base_definitions = collect_function_base_definitions(
        &handles,
        transaction,
        &module_ids,
        &reversed_override_graph,
    );
    let global_variables = collect_global_variables(&handles, transaction, &module_ids);
    let captured_variables = collect_captured_variables(&handles, transaction, &module_ids);

    let override_graph =
        OverrideGraph::from_reversed(&reversed_override_graph, &function_base_definitions);

    write_module_definitions_files(
        &module_work_list,
        &function_base_definitions,
        &global_variables,
        &captured_variables,
        transaction,
        &module_ids,
        &definitions_directory,
    )?;

    write_module_type_of_expressions_files(
        &module_work_list,
        transaction,
        &module_ids,
        &type_of_expressions_directory,
    )?;

    write_module_call_graph_files(
        &module_work_list,
        &function_base_definitions,
        &override_graph,
        &global_variables,
        &captured_variables,
        transaction,
        &module_ids,
        &call_graphs_directory,
    )?;

    let project_modules =
        add_module_is_test_flags(project_modules, &module_work_list, transaction, &module_ids)?;

    write_typeshed_files(results_directory)?;

    write_errors_file(results_directory, errors, &module_ids)?;

    let builtin_module = handles
        .iter()
        .filter(|handle| handle.module().as_str() == "builtins")
        .exactly_one()
        .expect("expected exactly one builtins module");
    let object_class_id = ClassId::from_class(
        transaction
            .get_stdlib(builtin_module)
            .object()
            .class_object(),
    );

    let writer = BufWriter::new(File::create(results_directory.join("pyrefly.pysa.json"))?);
    serde_json::to_writer(
        writer,
        &PysaProjectFile {
            format_version: 1,
            modules: project_modules,
            builtin_module_id: module_ids
                .get(ModuleKey::from_handle(builtin_module))
                .unwrap(),
            object_class_id,
        },
    )?;

    step.finish();
    Ok(())
}
