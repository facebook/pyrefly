/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Parser;
use dupe::Dupe;
use pyrefly_util::forgetter::Forgetter;
use pyrefly_util::fs_anyhow;
use pyrefly_util::globs::FilteredGlobs;

use crate::commands::check::Handles;
use crate::commands::check::checkpoint;
use crate::commands::run::CommandExitStatus;
use crate::config::finder::ConfigFinder;
use crate::state::lsp::AnnotationKind;
use crate::state::require::Require;
use crate::state::state::State;
use crate::types::stdlib::Stdlib;
use crate::types::types::Type;

#[derive(Debug, Parser, Clone)]
pub struct Args {}

fn format_hints(
    inlay_hints: Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>,
    stdlib: &Stdlib,
) -> Vec<(ruff_text_size::TextSize, String)> {
    let mut qualified_hints = Vec::new();
    for (position, hint, kind) in inlay_hints {
        let formatted_hint = hint_to_string(hint, stdlib);
        // TODO: Put this behind a flag
        if formatted_hint == "Any" {
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
    let hint = hint.promote_literals(stdlib).explicit_any();
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
        let holder = Forgetter::new(State::new(config_finder), false);
        let handles = Handles::new(
            expanded_file_list,
            search_path.as_deref().unwrap_or_default(),
            holder.as_ref().config_finder(),
        );
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );
        let transaction = forgetter.as_mut();
        for (handle, _) in handles.all(Require::Everything) {
            transaction.run(&[(handle.dupe(), Require::Everything)]);
            let stdlib = transaction.get_stdlib(&handle);
            let inferred_types: Option<Vec<(ruff_text_size::TextSize, Type, AnnotationKind)>> =
                transaction.inferred_types(&handle);
            if let Some(i_types) = inferred_types {
                let formatted = format_hints(i_types, &stdlib.clone());
                let sorted = sort_inlay_hints(formatted);

                let file_path = handle.path().as_path();
                let file_content = fs_anyhow::read_to_string(file_path)
                    .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
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
                    .with_context(|| format!("Failed to write to file: {}", file_path.display()))?;
            }
        }
        Ok(CommandExitStatus::Success)
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
    fn test_parameter_annotation() -> anyhow::Result<()> {
        // Test parameter type annotation
        // TODO: Figure out how to get the parameter type inferred here, too
        assert_annotations(
            r#"
def greet(name):
    return "Hello, " + name
"#,
            r#"
def greet(name) -> str:
    return "Hello, " + name
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
    fn test_complex_function() -> anyhow::Result<()> {
        // Test a more complex function with multiple types
        assert_annotations(
            r#"
    def process_data(items, factor):
        result = []
        for item in items:
            result.append(item * factor)
        return result
    "#,
            r#"
    def process_data(items, factor) -> list[Any]:
        result = []
        for item in items:
            result.append(item * factor)
        return result
    "#,
        );
        Ok(())
    }
}
