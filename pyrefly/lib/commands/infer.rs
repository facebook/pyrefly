/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use clap::Parser;
use dupe::Dupe;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_types::types::Union;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::includes::Includes;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;
use tracing::error;

use crate::commands::check;
use crate::commands::check::Handles;
use crate::commands::files::FilesArgs;
use crate::commands::files::get_project_config_for_current_dir;
use crate::commands::util::CommandExitStatus;
use crate::config::error_kind::ErrorKind;
use crate::lsp::wasm::inlay_hints::ParameterAnnotation;
use crate::state::ide::insert_import_edit_with_forced_import_format;
use crate::state::lsp::AnnotationKind;
use crate::state::require::Require;
use crate::state::state::State;
use crate::types::class::Class;
use crate::types::simplify::unions_with_literals;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

#[deny(clippy::missing_docs_in_private_items)]
/// Flags for controlling the behavior of the autotype command
#[derive(Debug, Clone, Parser)]
pub struct InferFlags {
    // Default should be false for all of them and then we can override to easily customize
    /// Whether to add type annotations to container types like lists and dictionaries
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    pub containers: Option<bool>,
    /// Whether to add return type annotations to functions
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    pub return_types: Option<bool>,
    /// Whether to add type annotations to function parameters
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    pub parameter_types: Option<bool>,
    /// Whether to automatically add imports for types used in annotations
    #[arg(
        long,
        default_missing_value = "true",
        require_equals = true,
        num_args = 0..=1
    )]
    pub imports: Option<bool>,
}

impl InferFlags {
    pub fn default() -> Self {
        Self {
            containers: Some(false),
            return_types: Some(true),
            parameter_types: Some(true),
            imports: Some(true),
        }
    }

    pub fn containers(&self) -> bool {
        self.containers.unwrap_or(false)
    }

    pub fn return_types(&self) -> bool {
        self.return_types.unwrap_or(true)
    }

    pub fn parameter_types(&self) -> bool {
        self.parameter_types.unwrap_or(true)
    }

    pub fn imports(&self) -> bool {
        self.imports.unwrap_or(true)
    }
}

/// Arguments for the autotype command which automatically adds type annotations to Python code
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct InferArgs {
    /// Which files to check.
    #[command(flatten)]
    files: FilesArgs,

    /// Type checking arguments and configuration
    #[command(flatten)]
    config_override: ConfigOverrideArgs,

    /// Flags controlling the behavior of the autotype command
    #[command(flatten)]
    flags: InferFlags,
}

impl ParameterAnnotation {
    fn to_inlay_hint(self) -> Option<(TextSize, Type, AnnotationKind)> {
        if let Some(ty) = self.ty {
            if ty.is_any() || self.has_annotation {
                return None;
            }
            Some((self.text_size, ty, AnnotationKind::Parameter))
        } else {
            None
        }
    }
}

fn is_container(hint: &Type) -> bool {
    match hint {
        Type::ClassType(c) => c.name().eq("list") || c.name().eq("dict"),
        _ => false,
    }
}

fn format_hints(
    inlay_hints: Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
) -> Vec<(ruff_text_size::TextSize, String)> {
    let mut qualified_hints = Vec::new();
    for (position, hint, kind) in inlay_hints {
        let is_container = is_container(&hint);
        let formatted_hint = hint_to_string(hint, stdlib, enum_members);
        // TODO: Put these behind a flag
        if formatted_hint.contains("Any") {
            continue;
        }
        if formatted_hint.contains("@") {
            continue;
        }
        if formatted_hint.contains("Unknown") {
            continue;
        }
        if formatted_hint.contains("Never") {
            continue;
        }
        if formatted_hint == "None" && kind == AnnotationKind::Parameter {
            continue;
        }
        if !is_container && kind == AnnotationKind::Variable {
            continue;
        }
        match kind {
            AnnotationKind::Parameter => {
                qualified_hints.push((position, format!(": {formatted_hint}")));
            }
            AnnotationKind::Return => {
                qualified_hints.push((position, format!(" -> {formatted_hint}")));
            }
            AnnotationKind::Variable => {
                qualified_hints.push((position, format!(": {formatted_hint}")));
            }
        }
    }
    qualified_hints
}

// Sort the hints by reverse order so we don't have to recalculate positions
fn sort_inlay_hints(
    inlay_hints: Vec<(ruff_text_size::TextSize, String)>,
) -> Vec<(ruff_text_size::TextSize, String)> {
    let mut sorted_inlay_hints = inlay_hints;
    sorted_inlay_hints.sort_by(|(a, _), (b, _)| b.cmp(a));
    sorted_inlay_hints
}

fn hint_to_string(
    hint: Type,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
) -> String {
    let hint = hint.promote_literals(stdlib);
    let hint = hint.explicit_any().clean_var();
    let hint = match hint {
        Type::Union(box Union { members: types, .. }) => {
            unions_with_literals(types, stdlib, enum_members)
        }
        _ => hint,
    };
    hint.to_string()
}

impl InferArgs {
    pub fn run(self) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        let (files_to_check, config_finder) = self.files.resolve(self.config_override)?;
        Self::run_inner(files_to_check, config_finder, self.flags)
    }

    pub fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        flags: InferFlags,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files())?;
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );

        let mut cancellable_transaction = holder.as_ref().cancellable_transaction();
        let transaction = forgetter.as_mut();

        let (handles, _, sourcedb_errors) = handles.all(holder.as_ref().config_finder());
        if !sourcedb_errors.is_empty() {
            for error in sourcedb_errors {
                error.print();
            }
            return Err(anyhow::anyhow!("Failed to query sourcedb."));
        }
        for handle in handles {
            transaction.run(&[handle.dupe()], Require::Everything);
            let stdlib = transaction.get_stdlib(&handle);
            let inferred_types: Option<Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>> =
                transaction.inferred_types(&handle, flags.return_types(), flags.containers());
            let parameter_annotations = if flags.parameter_types() {
                transaction.infer_parameter_annotations(&handle, &mut cancellable_transaction)
            } else {
                Vec::new()
            };
            // Map them to the inferred_types pattern
            let mut parameter_types: Vec<(TextSize, Type, AnnotationKind)> = parameter_annotations
                .into_iter()
                .filter_map(|p| p.to_inlay_hint())
                .collect();
            if let Some(inferred_types) = inferred_types {
                parameter_types.extend(inferred_types);
                let formatted = format_hints(parameter_types, &stdlib, &|cls| {
                    transaction
                        .ad_hoc_solve(&handle, |solver| {
                            let meta = solver.get_metadata_for_class(cls);
                            if meta.is_enum() {
                                Some(solver.get_enum_members(cls).len())
                            } else {
                                None
                            }
                        })
                        .flatten()
                });
                let sorted = sort_inlay_hints(formatted);
                let file_path = handle.path().as_path();
                Self::add_annotations_to_file(file_path, sorted)?;
            }
        }
        // Add imports, if needed
        let check_args = check::CheckArgs::parse_from(["check", "--output-format", "omit-errors"]);
        let current_dir_config =
            get_project_config_for_current_dir(ConfigOverrideArgs::default())?.0;
        let config_finder = ConfigFinder::new_constant(current_dir_config);
        let state = holder.as_ref();
        match check_args.run_once(files_to_check, config_finder) {
            Ok((_, errors)) => {
                for error in errors {
                    match error.error_kind() {
                        ErrorKind::UnknownName => {
                            let module_info = error.module();
                            let module_path = module_info.path().clone();
                            let config = state.config_finder().python_file(
                                pyrefly_python::module_name::ModuleName::unknown(),
                                &module_path,
                            );
                            let handle = config.handle_from_module_path(module_path);
                            if let Some(ast) = transaction.get_ast(&handle) {
                                let error_range = error.range();
                                let unknown_name = module_info.code_at(error_range);
                                let imports: Vec<(TextSize, String, String)> = transaction
                                    .search_exports_exact(unknown_name)
                                    .into_iter()
                                    .map(|(handle_to_import_from, _)| {
                                        insert_import_edit_with_forced_import_format(
                                            &ast,
                                            handle.dupe(),
                                            handle_to_import_from.dupe(),
                                            unknown_name,
                                            true,
                                        )
                                    })
                                    .collect();
                                let path = error.path();
                                Self::add_imports_to_file(path.as_path(), imports)?;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                error!("Failed to run pyrefly check: {}", e);
            }
        };
        Ok(CommandExitStatus::Success)
    }

    fn add_annotations_to_file(
        file_path: &Path,
        sorted: Vec<(TextSize, String)>,
    ) -> anyhow::Result<()> {
        let file_content = fs_anyhow::read_to_string(file_path)?;
        let mut result = file_content;
        for inlay_hint in sorted {
            let (position, hint) = inlay_hint;
            // Convert the TextSize to a byte offset
            let offset = (position).into();
            if offset <= result.len() {
                result.insert_str(offset, &hint);
            }
        }
        fs_anyhow::write(file_path, result)
    }

    fn add_imports_to_file(
        file_path: &Path,
        imports: Vec<(TextSize, String, String)>,
    ) -> anyhow::Result<()> {
        let file_content = fs_anyhow::read_to_string(file_path)?;
        let mut result = file_content;
        for (position, import, _) in imports {
            let offset = (position).into();
            if !result.contains(&import) {
                result.insert_str(offset, &import);
            }
        }
        fs_anyhow::write(file_path, result)
    }
}

#[cfg(test)]
mod test {
    use pretty_assertions::assert_str_eq;
    use pyrefly_util::globs::FilteredGlobs;
    use pyrefly_util::globs::Globs;
    use tempfile;

    use super::*;
    use crate::test::util::TestEnv;

    fn assert_annotations(input: &str, output: &str, flags: Option<InferFlags>) {
        let flags = flags.unwrap_or_else(InferFlags::default);
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input).unwrap();
        let mut t = TestEnv::new();
        t.add(&path.display().to_string(), input);
        let includes =
            Globs::new(vec![format!("{}/**/*", tdir.path().display()).to_owned()]).unwrap();
        let f_globs = Box::new(FilteredGlobs::new(includes, Globs::empty(), None));
        let config_finder = t.config_finder();
        let result = InferArgs::run_inner(f_globs, config_finder, flags);
        assert!(
            result.is_ok(),
            "autotype command failed: {:?}",
            result.err()
        );

        let got_file = fs_anyhow::read_to_string(&path).unwrap();
        assert_str_eq!(
            output,
            got_file,
            "File content after autotype doesn't match expected output"
        );
    }

    fn assert_imports_and_annotations(file_one: &str, file_two: &str, output: &str) {
        let configuration = r#"
        project_includes = [
            "file_one.py",
            "file_two.py",
        ]
        project_excludes = []
        "#;
        let tdir = tempfile::TempDir::with_prefix("pyrefly_infer_test").unwrap();
        let file_one_path = tdir.path().join("file_one.py");
        fs_anyhow::write(&file_one_path, file_one).unwrap();
        let file_two_path = tdir.path().join("file_two.py");
        fs_anyhow::write(&file_two_path, file_two).unwrap();
        let config_path = tdir.path().join("pyrefly.toml");
        fs_anyhow::write(&config_path, configuration).unwrap();
        let mut t = TestEnv::new();
        t.add(&file_one_path.display().to_string(), file_one);
        t.add(&file_two_path.display().to_string(), file_two);
        t.add(&config_path.display().to_string(), configuration);
        let args = InferArgs::parse_from(["infer", "--config", &config_path.display().to_string()]);
        let result = args.run();
        assert!(result.is_ok(), "infer command failed: {:?}", result.err());

        let got_file = fs_anyhow::read_to_string(&file_one_path).unwrap();
        assert_str_eq!(
            output,
            got_file,
            "File content after infer doesn't match expected output"
        );
    }

    #[test]
    fn test_literal() -> anyhow::Result<()> {
        // Test return type annotation for integer literal
        assert_annotations(
            r#"
def foo():
    return 1
"#,
            r#"
def foo() -> int:
    return 1
"#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_literal_string() -> anyhow::Result<()> {
        // Test return type annotation for integer literal
        assert_annotations(
            r#"
def foo():
    return ""
"#,
            r#"
def foo() -> str:
    return ""
"#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_boolean_literal() -> anyhow::Result<()> {
        // Test boolean return type
        assert_annotations(
            r#"
    def is_valid():
        return True
    "#,
            r#"
    def is_valid() -> bool:
        return True
    "#,
            None,
        );
        Ok(())
    }

    #[test]

    fn test_parameter() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(a, b, c):
        return c
    example(1, 2, 3)
    "#,
            r#"
    def example(a: int, b: int, c: int):
        return c
    example(1, 2, 3)
    "#,
            None,
        );
        Ok(())
    }

    #[test]

    fn test_parameter_unions() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(a, b, c):
        return c
    example(1, 2, 3)
    x = 2
    example("a", "b", x)
    "#,
            r#"
    def example(a: int | str, b: int | str, c: int):
        return c
    example(1, 2, 3)
    x = 2
    example("a", "b", x)
    "#,
            None,
        );
        Ok(())
    }

    #[test]

    fn test_default_parameters() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(a, b, c = None):
        return c
    example(1, 2, 3)
    x = 2
    example("a", "b", x)
    "#,
            r#"
    def example(a: int | str, b: int | str, c: int | None = None):
        return c
    example(1, 2, 3)
    x = 2
    example("a", "b", x)
    "#,
            None,
        );
        Ok(())
    }

    #[test]

    fn test_default_parameters_infer_default_type() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(c = 1):
        return c
    example("a")
    "#,
            r#"
    def example(c: int | str = 1):
        return c
    example("a")
    "#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_return_none() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(c):
        c + 1
    "#,
            r#"
    def example(c) -> None:
        c + 1
    "#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_no_none_parameter() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def example(c = None):
        pass
    example(None)
    "#,
            r#"
    def example(c = None) -> None:
        pass
    example(None)
    "#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_default_parameter_no_call_site() -> anyhow::Result<()> {
        assert_annotations(
            r#"
    def foo(a=2) -> None:
        pass
    "#,
            r#"
    def foo(a: int=2) -> None:
        pass
    "#,
            None,
        );
        Ok(())
    }

    #[test]
    fn test_empty_container() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.containers = Some(true);
        assert_annotations(
            r#"
    def foo() -> None:
        x = []
        x.append(1)
    "#,
            r#"
    def foo() -> None:
        x: list[int] = []
        x.append(1)
    "#,
            Some(flags),
        );
        Ok(())
    }

    #[test]
    fn test_empty_dictionary() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.containers = Some(true);
        assert_annotations(
            r#"
    def foo() -> None:
        x = {}
        x["a"] = 1
    "#,
            r#"
    def foo() -> None:
        x: dict[str, int] = {}
        x["a"] = 1
    "#,
            Some(flags),
        );
        Ok(())
    }

    #[test]
    fn test_non_empty_dictionary() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.containers = Some(true);

        assert_annotations(
            r#"
    def foo() -> None:
        x = {"a": 1}
        x["a"] = 1
    "#,
            r#"
    def foo() -> None:
        x = {"a": 1}
        x["a"] = 1
    "#,
            Some(flags),
        );
        Ok(())
    }

    // TEST FLAGS
    #[test]
    fn test_no_parameter_flag() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.parameter_types = Some(false);
        assert_annotations(
            r#"
    def example(c = 1):
        return c
    example("a")
    "#,
            r#"
    def example(c = 1):
        return c
    example("a")
    "#,
            Some(flags),
        );
        Ok(())
    }

    #[test]
    fn test_no_containers_empty_dictionary() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.containers = Some(false);
        assert_annotations(
            r#"
    def foo() -> None:
        x = {}
        x["a"] = 1
    "#,
            r#"
    def foo() -> None:
        x = {}
        x["a"] = 1
    "#,
            Some(flags),
        );
        Ok(())
    }

    #[test]
    fn test_no_return_literal_string() -> anyhow::Result<()> {
        // Test return type annotation for integer literal
        let mut flags = InferFlags::default();
        flags.return_types = Some(false);

        assert_annotations(
            r#"
def foo():
    return ""
"#,
            r#"
def foo():
    return ""
"#,
            Some(flags),
        );
        Ok(())
    }

    #[test]
    fn test_imports() -> anyhow::Result<()> {
        let file_one = r#"
        from file_two import get_a
        def foo():
            return get_a()
        "#;
        let file_two = r#"
        class ExampleA:
            pass
        def get_a():
            return ExampleA()
        "#;
        let output = r#"
        from file_two import ExampleA
from file_two import get_a
        def foo() -> ExampleA:
            return get_a()
        "#;
        assert_imports_and_annotations(file_one, file_two, output);
        Ok(())
    }
    #[test]
    fn test_multiple_imports() -> anyhow::Result<()> {
        let file_one = r#"
        from file_two import get_a, get_b
        def foo():
            return get_a()
        def bar():
            return get_b()
        "#;
        let file_two = r#"
        class ExampleA:
            pass
        class ExampleB:
            pass
        def get_a():
            return ExampleA()
        def get_b():
            return ExampleB()
        "#;
        let output = r#"
        from file_two import ExampleB
from file_two import ExampleA
from file_two import get_a, get_b
        def foo() -> ExampleA:
            return get_a()
        def bar() -> ExampleB:
            return get_b()
        "#;
        assert_imports_and_annotations(file_one, file_two, output);
        Ok(())
    }
}
