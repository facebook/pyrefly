/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::includes::Includes;
use regex::Regex;
use ruff_python_ast::Parameters;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;

use crate::binding::binding::Binding;
use crate::binding::binding::BindingClass;
use crate::binding::binding::Key;
use crate::binding::binding::ReturnTypeKind;
use crate::binding::bindings::Bindings;
use crate::commands::check::Handles;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::state::require::Require;
use crate::state::state::State;

/// Location information for code elements
#[derive(Debug, Serialize)]
struct Location {
    start: Position,
    end: Position,
}

/// Position with line and column
#[derive(Debug, Serialize)]
struct Position {
    line: usize,
    column: usize,
}

/// Parameter information
#[derive(Debug, Serialize)]
struct Parameter {
    name: String,
    annotation: Option<String>,
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
    parameters: Vec<Parameter>,
    location: Location,
}

/// File report
#[derive(Debug, Serialize)]
struct FileReport {
    line_count: usize,
    functions: Vec<Function>,
    suppressions: Vec<Suppression>,
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
}

impl ReportArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override)?;
        Self::run_inner(files_to_check, config_finder)
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

    /// Helper to convert byte offset to line and column position
    fn offset_to_position(module: &Module, offset: ruff_text_size::TextSize) -> Position {
        let location = module.lined_buffer().line_index().source_location(
            offset,
            module.lined_buffer().contents(),
            ruff_source_file::PositionEncoding::Utf8,
        );
        Position {
            line: location.line.get(),
            column: location.character_offset.get(),
        }
    }

    /// Helper to convert a text range to a Location
    fn range_to_location(module: &Module, range: TextRange) -> Location {
        Location {
            start: Self::offset_to_position(module, range.start()),
            end: Self::offset_to_position(module, range.end()),
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
                    let end_col = line.len();

                    suppressions.push(Suppression {
                        kind: "ignore".to_owned(),
                        codes,
                        location: Location {
                            start: Position {
                                line: line_number,
                                column: start_col,
                            },
                            end: Position {
                                line: line_number,
                                column: end_col,
                            },
                        },
                    });
                }
            }
        }

        suppressions
    }

    fn parse_functions(module: &Module, bindings: Bindings) -> Vec<Function> {
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
                    format!("{}{}", module_prefix, fun.def.name)
                };
                // Get return annotation from ReturnTypeKind
                let return_annotation = {
                    let return_key = Key::ReturnType(*id);
                    let return_idx = bindings.key_to_idx(&return_key);
                    if let Binding::ReturnType(ret) = bindings.get(return_idx) {
                        match &ret.kind {
                            ReturnTypeKind::ShouldValidateAnnotation { range, .. } => {
                                Some(module.code_at(*range).to_owned())
                            }
                            ReturnTypeKind::ShouldTrustAnnotation { .. } => {
                                // For trusted annotations, get from AST
                                fun.def
                                    .returns
                                    .as_ref()
                                    .map(|ann| module.code_at(ann.range()).to_owned())
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                };

                // Get parameters
                let mut parameters = Vec::new();
                let all_params = Self::extract_parameters(&fun.def.parameters);

                for param in all_params {
                    let param_name = param.name.as_str();
                    let param_annotation = param
                        .annotation
                        .as_ref()
                        .map(|ann| module.code_at(ann.range()).to_owned());

                    parameters.push(Parameter {
                        name: param_name.to_owned(),
                        annotation: param_annotation,
                        location: Self::range_to_location(module, param.range),
                    });
                }
                functions.push(Function {
                    name: func_name,
                    return_annotation,
                    parameters,
                    location,
                });
            }
        }
        functions
    }

    fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
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

        for handle in handles {
            transaction.run(&[handle.dupe()], Require::Everything);

            if let Some(bindings) = transaction.get_bindings(&handle)
                && let Some(module) = transaction.get_module_info(&handle)
            {
                let line_count = module.lined_buffer().line_index().line_count();
                let functions = Self::parse_functions(&module, bindings);
                let suppressions = Self::parse_suppressions(&module);

                report.insert(
                    handle.path().as_path().display().to_string(),
                    FileReport {
                        line_count,
                        functions,
                        suppressions,
                    },
                );
            }
        }

        // Output JSON
        let json = serde_json::to_string_pretty(&report)?;
        println!("{}", json);

        Ok(CommandExitStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;

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
        assert_eq!(suppressions[0].location.start.line, 2);

        // Suppression with multiple error codes
        assert_eq!(suppressions[1].kind, "ignore");
        assert_eq!(suppressions[1].codes, vec!["code1", "code2"]);
        assert_eq!(suppressions[1].location.start.line, 4);
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

        let functions = ReportArgs::parse_functions(&module, bindings);

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

        let functions = ReportArgs::parse_functions(&module, bindings);

        assert_eq!(functions.len(), 1);
        assert_eq!(functions[0].name, "foo");
    }
}
