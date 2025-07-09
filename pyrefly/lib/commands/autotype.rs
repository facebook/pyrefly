/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use dupe::Dupe;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::globs::FilteredGlobs;
use ruff_text_size::TextSize;

use crate::commands::check::Handles;
use crate::commands::check::checkpoint;
use crate::commands::run::CommandExitStatus;
use crate::config::finder::ConfigFinder;
use crate::state::lsp::AnnotationKind;
use crate::state::lsp::ParameterAnnotation;
use crate::state::require::Require;
use crate::state::state::State;
use crate::types::simplify::unions_with_literals;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

/// Arguments for the autotype command which automatically adds type annotations to Python code
#[deny(clippy::missing_docs_in_private_items)]
#[derive(Debug, Parser, Clone)]
pub struct Args {}

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

fn format_hints(
    inlay_hints: Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>,
    stdlib: &Stdlib,
) -> Vec<(ruff_text_size::TextSize, String)> {
    let mut qualified_hints = Vec::new();
    for (position, hint, kind) in inlay_hints {
        let formatted_hint = hint_to_string(hint, stdlib);
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
        match kind {
            AnnotationKind::Parameter => {
                qualified_hints.push((position, format!(": {}", formatted_hint)));
            }
            AnnotationKind::Return => {
                qualified_hints.push((position, format!(" -> {}", formatted_hint)));
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

fn hint_to_string(hint: Type, stdlib: &Stdlib) -> String {
    let hint = hint.promote_literals(stdlib);
    let hint = hint.explicit_any().clean_var();
    let hint = match hint {
        Type::Union(types) => unions_with_literals(types, stdlib),
        _ => hint,
    };
    hint.to_string()
}

impl Args {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(
        self,
        files_to_check: FilteredGlobs,
        config_finder: ConfigFinder,
        search_path: Option<Vec<PathBuf>>,
    ) -> anyhow::Result<CommandExitStatus> {
        let expanded_file_list = checkpoint(files_to_check.files(), &config_finder)?;
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);
        let handles = Handles::new(
            expanded_file_list,
            search_path.as_deref().unwrap_or_default(),
            holder.as_ref().config_finder(),
        );
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );

        let mut cancellable_transaction = holder.as_ref().cancellable_transaction();
        let transaction = forgetter.as_mut();

        for (handle, _) in handles.all(Require::Everything) {
            transaction.run(&[(handle.dupe(), Require::Everything)]);
            let stdlib = transaction.get_stdlib(&handle);
            let inferred_types: Option<Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>> =
                transaction.inferred_types(&handle);
            let parameter_annotations =
                transaction.infer_parameter_annotations(&handle, &mut cancellable_transaction);
            // Map them to the inferred_types pattern
            let mut parameter_types: Vec<(TextSize, Type, AnnotationKind)> = parameter_annotations
                .into_iter()
                .filter_map(|p| p.to_inlay_hint())
                .collect();
            if let Some(inferred_types) = inferred_types {
                parameter_types.extend(inferred_types);
                let formatted = format_hints(parameter_types, &stdlib);
                let sorted = sort_inlay_hints(formatted);
                let file_path = handle.path().as_path();
                self.add_annotations_to_file(file_path, sorted)?;
            }
        }
        Ok(CommandExitStatus::Success)
    }

    fn add_annotations_to_file(
        &self,
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
        fs_anyhow::write(file_path, result.as_bytes())
    }
}

#[cfg(test)]
mod test {
    use pretty_assertions::assert_str_eq;
    use pyrefly_util::globs::Globs;
    use tempfile;

    use super::*;
    use crate::test::util::TestEnv;

    fn assert_annotations(input: &str, output: &str) {
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input.as_bytes()).unwrap();
        let mut t = TestEnv::new();
        t.add(&path.display().to_string(), input);
        let includes = Globs::new(vec![format!("{}/**/*", tdir.path().display()).to_owned()]);
        let f_globs = FilteredGlobs::new(includes, Globs::new(vec![]));
        let config_finder = t.config_finder();
        let arg = Args::new();
        let result = arg.run(f_globs, config_finder, None);
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
        );
        Ok(())
    }
}
