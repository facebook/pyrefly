/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;

use clap::Parser;
use pyrefly_config::args::ConfigOverrideArgs;
use pyrefly_config::base::InferReturnTypes;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::qname::QName;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::includes::Includes;
use pyrefly_util::thread_pool::ThreadCount;
use ruff_python_ast::Stmt;
use ruff_python_ast::helpers::is_docstring_stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

use crate::commands::check::Handles;
use crate::commands::config_finder::ConfigConfigurerWrapper;
use crate::commands::files::FilesArgs;
use crate::commands::util::CommandExitStatus;
use crate::lsp::wasm::inlay_hints::ParameterAnnotation;
use crate::state::lsp::AnnotationKind;
use crate::state::require::Require;
use crate::state::state::State;
use crate::types::class::Class;
use crate::types::display::TypeDisplayContext;
use crate::types::heap::TypeHeap;
use crate::types::simplify::unions_with_literals;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

/// Check if a statement is a `from __future__ import ...` statement.
/// New imports must be inserted after `__future__` imports to produce valid Python.
fn is_future_import_stmt(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::ImportFrom(import_from) if import_from.module.as_ref().is_some_and(|m| m.id == "__future__")
    )
}

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
    /// Print what would change and exit (1 if any changes, else 0), without writing any files.
    #[arg(long)]
    pub dry_run: bool,
}

impl InferFlags {
    pub fn default() -> Self {
        Self {
            containers: Some(false),
            return_types: Some(true),
            parameter_types: Some(true),
            imports: Some(true),
            dry_run: false,
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

    pub fn dry_run(&self) -> bool {
        self.dry_run
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

/// Returns the name to import for a given QName. For top-level names, this is
/// the name itself. For nested classes (e.g., `Outer.Inner`), this walks up
/// the parent chain to find the outermost class that should be imported.
fn importable_name_for_qname(qname: &QName) -> String {
    let mut nesting = qname.parent();
    if nesting.is_toplevel() {
        return qname.id().to_string();
    }
    // Walk up the parent chain to find the outermost non-toplevel context
    while let Some(parent) = nesting.parent() {
        if parent.is_toplevel() {
            let short_id = nesting
                .identifier()
                .expect("non-toplevel NestingContext must have an identifier");
            return qname.module().code_at(short_id.range()).to_owned();
        }
        nesting = parent;
    }
    unreachable!("NestingContext chain should always end with Toplevel")
}

/// Formats inlay hints as annotation strings and collects the set of imports
/// needed for the types that pass filtering. Returns the formatted annotations
/// and a set of `(source_module, importable_name)` pairs to import.
fn format_hints(
    inlay_hints: Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
    heap: &TypeHeap,
    current_module_name: ModuleName,
) -> (
    Vec<(ruff_text_size::TextSize, String)>,
    HashSet<(ModuleName, String)>,
) {
    let mut qualified_hints = Vec::new();
    let mut needed_imports = HashSet::new();
    for (position, hint, kind) in inlay_hints {
        let is_container = is_container(&hint);
        let contains_self_type = hint.any(|sub_type| matches!(sub_type, Type::SelfType(_)));
        // Collect QNames before hint_to_string consumes the type. Each QName
        // carries the defining module, so we know exactly where to import from.
        let mut hint_imports = Vec::new();
        hint.universe(&mut |sub_type| {
            if matches!(sub_type, Type::SelfType(_)) {
                return;
            }
            if let Some(qname) = sub_type.qname() {
                let module_name = qname.module_name();
                if module_name != ModuleName::builtins() && module_name != current_module_name {
                    hint_imports.push((module_name, importable_name_for_qname(qname)));
                }
            }
        });
        if contains_self_type {
            hint_imports.push((ModuleName::typing(), "Self".to_owned()));
        }
        let formatted_hint = hint_to_string(hint, stdlib, enum_members, heap);
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
        if formatted_hint.contains("Overload") {
            continue;
        }
        if formatted_hint == "None" && kind == AnnotationKind::Parameter {
            continue;
        }
        if !is_container && kind == AnnotationKind::Variable {
            continue;
        }
        // Only record imports for types that pass all filters above
        needed_imports.extend(hint_imports);
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
    (qualified_hints, needed_imports)
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
    heap: &TypeHeap,
) -> String {
    let hint = hint.promote_implicit_literals(stdlib);
    let hint = hint.explicit_any().clean_var();
    let hint = match hint {
        Type::Union(u) => unions_with_literals(u.members, stdlib, enum_members, heap),
        _ => hint,
    };
    let mut ctx = TypeDisplayContext::new(&[&hint]);
    ctx.render_self_type_as_self();
    ctx.display(&hint).to_string()
}

impl InferArgs {
    pub fn run(
        mut self,
        wrapper: Option<ConfigConfigurerWrapper>,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        self.config_override.validate()?;
        // The infer command must analyze function bodies to produce meaningful
        // return type annotations. Ensure both settings are enabled unless
        // the user explicitly set them via CLI.
        self.config_override
            .set_check_unannotated_defs_if_unset(true);
        self.config_override
            .set_infer_return_types_if_unset(InferReturnTypes::Checked);
        let (files_to_check, config_finder, _) =
            self.files.resolve(self.config_override, wrapper)?;
        Self::run_inner(files_to_check, config_finder, self.flags, thread_count)
    }

    pub fn run_inner(
        files_to_check: Box<dyn Includes>,
        config_finder: ConfigFinder,
        flags: InferFlags,
        thread_count: ThreadCount,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = config_finder.checkpoint(files_to_check.files_iter())?;
        let state = State::new(config_finder, thread_count);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(expanded_file_list);
        // Use Exports as the default require level for dependency modules —
        // only the files being inferred need Everything.
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Exports, None),
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
        // Type-check all handles at once so the work can be parallelised across
        // the thread pool, instead of checking one file at a time sequentially.
        transaction.run(&handles, Require::Everything, None);
        let mut any_changes = false;
        for handle in handles {
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
                let heap = TypeHeap::new();
                let (formatted, needed_imports) = format_hints(
                    parameter_types,
                    &stdlib,
                    &|cls| {
                        transaction
                            .ad_hoc_solve(&handle, "infer_enum_metadata", |solver| {
                                let meta = solver.get_metadata_for_class(cls);
                                if meta.is_enum() {
                                    Some(solver.get_enum_members(cls).len())
                                } else {
                                    None
                                }
                            })
                            .flatten()
                    },
                    &heap,
                    handle.module(),
                );
                let sorted = sort_inlay_hints(formatted);
                let file_path = handle.path().as_path();
                // Build import edits once so dry-run and write share them.
                let imports: Vec<(TextSize, String, String)> = if flags.imports()
                    && !needed_imports.is_empty()
                    && let Some(ast) = transaction.get_ast(&handle)
                {
                    let position = ast
                        .body
                        .iter()
                        .find(|stmt| !is_docstring_stmt(stmt) && !is_future_import_stmt(stmt))
                        .map_or(ast.range.end(), |stmt| stmt.range().start());
                    let mut imports: Vec<(TextSize, String, String)> = needed_imports
                        .into_iter()
                        .map(|(module_name, name)| {
                            let import_text =
                                format!("from {} import {}\n", module_name.as_str(), name);
                            (position, import_text, module_name.as_str().to_owned())
                        })
                        .collect();
                    imports.sort_by(|(_, a, _), (_, b, _)| a.cmp(b));
                    imports
                } else {
                    Vec::new()
                };
                if flags.dry_run() {
                    if !sorted.is_empty() || !imports.is_empty() {
                        any_changes = true;
                        println!(
                            "{}: would add {} annotation(s) and {} import(s)",
                            file_path.display(),
                            sorted.len(),
                            imports.len()
                        );
                    }
                } else {
                    Self::add_annotations_to_file(file_path, sorted)?;
                    if !imports.is_empty() {
                        Self::add_imports_to_file(file_path, imports)?;
                    }
                }
            }
        }
        Ok(if flags.dry_run() && any_changes {
            CommandExitStatus::UserError
        } else {
            CommandExitStatus::Success
        })
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
    use pyrefly_util::globs::HiddenDirFilter;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
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
        let f_globs = Box::new(FilteredGlobs::new(
            includes,
            Globs::empty(),
            None,
            HiddenDirFilter::Disabled,
        ));
        let config_finder = t.config_finder();
        let result = InferArgs::run_inner(f_globs, config_finder, flags, TEST_THREAD_COUNT);
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
        let result = args.run(None, TEST_THREAD_COUNT);
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
    fn test_fully_annotated_function_skipped() -> anyhow::Result<()> {
        // Parameters on a fully annotated function should not be re-inferred.
        assert_annotations(
            r#"
    def example(a: int, b: str) -> None:
        pass
    example(1, "hello")
    "#,
            r#"
    def example(a: int, b: str) -> None:
        pass
    example(1, "hello")
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
    fn test_unannotated_dunder_new_return_is_inserted_as_self() -> anyhow::Result<()> {
        let mut flags = InferFlags::default();
        flags.parameter_types = Some(false);

        assert_annotations(
            r#"
class C:
    def __new__(cls):
        return super().__new__(cls)
"#,
            r#"
from typing import Self
class C:
    def __new__(cls) -> Self:
        return super().__new__(cls)
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

    #[test]
    fn test_best_import_selected_when_multiple_modules_export_same_name() -> anyhow::Result<()> {
        // When multiple modules export the same name, only the best import
        // should be added (shortest public module path).
        let configuration = r#"
        project_includes = [
            "file_one.py",
            "short.py",
            "_private/long_path.py",
        ]
        project_excludes = []
        "#;
        let tdir = tempfile::TempDir::with_prefix("pyrefly_infer_test").unwrap();

        let file_one = r#"
from short import get_a
def foo():
    return get_a()
"#;
        // short.py exports MyClass — public, short path
        let file_short = r#"
class MyClass:
    pass
def get_a():
    return MyClass()
"#;
        // _private/long_path.py also exports MyClass — private, longer path
        let private_dir = tdir.path().join("_private");
        std::fs::create_dir_all(&private_dir).unwrap();
        let file_private = r#"
class MyClass:
    pass
"#;

        let file_one_path = tdir.path().join("file_one.py");
        fs_anyhow::write(&file_one_path, file_one).unwrap();
        let file_short_path = tdir.path().join("short.py");
        fs_anyhow::write(&file_short_path, file_short).unwrap();
        let file_private_path = private_dir.join("long_path.py");
        fs_anyhow::write(&file_private_path, file_private).unwrap();
        let config_path = tdir.path().join("pyrefly.toml");
        fs_anyhow::write(&config_path, configuration).unwrap();

        let mut t = TestEnv::new();
        t.add(&file_one_path.display().to_string(), file_one);
        t.add(&file_short_path.display().to_string(), file_short);
        t.add(&file_private_path.display().to_string(), file_private);
        t.add(&config_path.display().to_string(), configuration);
        let args = InferArgs::parse_from(["infer", "--config", &config_path.display().to_string()]);
        let result = args.run(None, TEST_THREAD_COUNT);
        assert!(result.is_ok(), "infer command failed: {:?}", result.err());

        let got_file = fs_anyhow::read_to_string(&file_one_path).unwrap();
        // Should import from "short" (public, 1 component), not from
        // "_private.long_path" (private, 2 components).
        assert!(
            got_file.contains("from short import MyClass"),
            "Expected import from 'short' module, got:\n{}",
            got_file,
        );
        assert!(
            !got_file.contains("_private"),
            "Should not import from private module, got:\n{}",
            got_file,
        );
        Ok(())
    }

    #[test]
    fn test_imports_after_future_import() -> anyhow::Result<()> {
        // New imports must be inserted after `from __future__ import annotations`,
        // not before it, to produce valid Python.
        let file_one = r#"from __future__ import annotations
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
        let output = r#"from __future__ import annotations
from file_two import ExampleA
from file_two import get_a
def foo() -> ExampleA:
    return get_a()
"#;
        assert_imports_and_annotations(file_one, file_two, output);
        Ok(())
    }

    #[test]
    fn test_imports_after_future_import_with_docstring() -> anyhow::Result<()> {
        // New imports must be inserted after both the docstring and `from __future__ import`.
        let file_one = r#""""Module docstring."""
from __future__ import annotations
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
        let output = r#""""Module docstring."""
from __future__ import annotations
from file_two import ExampleA
from file_two import get_a
def foo() -> ExampleA:
    return get_a()
"#;
        assert_imports_and_annotations(file_one, file_two, output);
        Ok(())
    }

    // -- dry-run tests --

    fn run_dry_run(input: &str) -> (CommandExitStatus, String) {
        let flags = InferFlags {
            dry_run: true,
            ..InferFlags::default()
        };
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input).unwrap();
        let mut t = TestEnv::new();
        t.add(&path.display().to_string(), input);
        let includes =
            Globs::new(vec![format!("{}/**/*", tdir.path().display()).to_owned()]).unwrap();
        let f_globs = Box::new(FilteredGlobs::new(
            includes,
            Globs::empty(),
            None,
            HiddenDirFilter::Disabled,
        ));
        let config_finder = t.config_finder();
        let status = InferArgs::run_inner(f_globs, config_finder, flags, TEST_THREAD_COUNT)
            .expect("run_inner should not error");
        let on_disk = fs_anyhow::read_to_string(&path).unwrap();
        (status, on_disk)
    }

    #[test]
    fn test_dry_run_changes_present_exits_1_and_file_unchanged() -> anyhow::Result<()> {
        // A file that would receive annotations: dry-run must leave it unchanged
        // and return UserError (exit 1).
        let input = r#"
def foo():
    return 1
"#;
        let (status, on_disk) = run_dry_run(input);
        assert_eq!(
            status,
            CommandExitStatus::UserError,
            "dry-run with changes should return UserError"
        );
        assert_str_eq!(input, on_disk, "dry-run must not modify the file");
        Ok(())
    }

    #[test]
    fn test_dry_run_no_changes_exits_0() -> anyhow::Result<()> {
        // A fully-annotated file: dry-run must return Success (exit 0).
        let input = r#"
def foo() -> int:
    return 1
"#;
        let (status, on_disk) = run_dry_run(input);
        assert_eq!(
            status,
            CommandExitStatus::Success,
            "dry-run with no changes should return Success"
        );
        assert_str_eq!(input, on_disk, "dry-run must not modify the file");
        Ok(())
    }

    #[test]
    fn test_non_dry_run_still_writes_regression() -> anyhow::Result<()> {
        // Without --dry-run, infer must still annotate the file (R6 regression guard).
        let input = r#"
def foo():
    return 1
"#;
        let expected = r#"
def foo() -> int:
    return 1
"#;
        assert_annotations(input, expected, None);
        Ok(())
    }

    #[test]
    fn test_dry_run_multiple_files_any_change_exits_1() -> anyhow::Result<()> {
        // Two files: one missing annotations, one fully annotated.
        // Dry-run must exit 1 (any file would change) and leave BOTH files unmodified.
        let unannotated = "def foo():\n    return 1\n";
        let annotated = "def bar() -> int:\n    return 2\n";
        let configuration =
            "project_includes = [\"unannotated.py\", \"annotated.py\"]\nproject_excludes = []\n";
        let tdir = tempfile::TempDir::with_prefix("pyrefly_infer_test_multi").unwrap();
        let unannotated_path = tdir.path().join("unannotated.py");
        let annotated_path = tdir.path().join("annotated.py");
        let config_path = tdir.path().join("pyrefly.toml");
        fs_anyhow::write(&unannotated_path, unannotated).unwrap();
        fs_anyhow::write(&annotated_path, annotated).unwrap();
        fs_anyhow::write(&config_path, configuration).unwrap();

        let mut t = TestEnv::new();
        t.add(&unannotated_path.display().to_string(), unannotated);
        t.add(&annotated_path.display().to_string(), annotated);
        t.add(&config_path.display().to_string(), configuration);

        let args = InferArgs::parse_from([
            "infer",
            "--dry-run",
            "--config",
            &config_path.display().to_string(),
        ]);
        let status = args.run(None, TEST_THREAD_COUNT)?;

        assert_eq!(
            status,
            CommandExitStatus::UserError,
            "dry-run with at least one changed file should return UserError"
        );
        assert_str_eq!(
            unannotated,
            fs_anyhow::read_to_string(&unannotated_path).unwrap(),
            "dry-run must not modify unannotated.py"
        );
        assert_str_eq!(
            annotated,
            fs_anyhow::read_to_string(&annotated_path).unwrap(),
            "dry-run must not modify annotated.py"
        );
        Ok(())
    }
}
