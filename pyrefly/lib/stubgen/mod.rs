/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod emit;
pub mod extract;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use dupe::Dupe;
    use pyrefly_util::forgetter::Forgetter;
    use pyrefly_util::fs_anyhow;
    use pyrefly_util::globs::FilteredGlobs;
    use pyrefly_util::globs::Globs;
    use pyrefly_util::includes::Includes;

    use super::emit::emit_stub;
    use super::extract::ExtractConfig;
    use super::extract::extract_module_stub;
    use crate::state::require::Require;
    use crate::state::state::State;
    use crate::test::util::TestEnv;

    fn run_stubgen(input: &str) -> String {
        run_stubgen_with_config(
            input,
            &ExtractConfig {
                include_private: false,
                include_docstrings: false,
            },
        )
    }

    fn run_stubgen_with_config(input: &str, config: &ExtractConfig) -> String {
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("input.py");
        fs_anyhow::write(&path, input).unwrap();
        let mut t = TestEnv::new();
        t.add(&path.display().to_string(), input);
        let includes = Globs::new(vec![format!("{}/**/*", tdir.path().display())]).unwrap();
        let f_globs = Box::new(FilteredGlobs::new(includes, Globs::empty(), None));
        let config_finder = t.config_finder();

        let expanded = config_finder.checkpoint(f_globs.files()).unwrap();
        let state = State::new(config_finder);
        let holder = Forgetter::new(state, false);

        let handles_obj = crate::commands::check::Handles::new(expanded);
        let mut forgetter = Forgetter::new(
            holder.as_ref().new_transaction(Require::Everything, None),
            true,
        );
        let transaction = forgetter.as_mut();

        let (handles, _, _) = handles_obj.all(holder.as_ref().config_finder());

        let mut result = String::new();
        for handle in &handles {
            transaction.run(&[handle.dupe()], Require::Everything, None);
            if let Some(stub) = extract_module_stub(transaction, handle, config) {
                result = emit_stub(&stub);
            }
        }
        result
    }

    /// Get the path to the stubgen test fixtures directory.
    fn get_test_dir() -> PathBuf {
        let test_path = std::env::var("STUBGEN_TEST_PATH")
            .expect("STUBGEN_TEST_PATH env var not set: buck should set this automatically");
        let mut dir = std::env::current_dir().expect("Failed to get current directory");
        dir.push(test_path);
        dir
    }

    /// Run a snapshot test for a specific test case directory.
    fn assert_stubgen_snapshot(test_name: &str) {
        let test_dir = get_test_dir().join(test_name);
        let input_path = test_dir.join("input.py");
        let expected_path = test_dir.join("expected.pyi");

        assert!(
            input_path.exists(),
            "Input file does not exist: {}",
            input_path.display()
        );

        let input = fs_anyhow::read_to_string(&input_path).unwrap();
        let actual = run_stubgen(&input);

        if std::env::var("STUBGEN_UPDATE_SNAPSHOTS").is_ok() {
            fs_anyhow::create_dir_all(&test_dir).unwrap();
            let out = test_dir.join("expected.pyi");
            fs_anyhow::write(&out, &actual).unwrap();
            println!("Updated snapshot for {} -> {}", test_name, out.display());
            return;
        }

        assert!(
            expected_path.exists(),
            "Expected file does not exist: {}\nRun with STUBGEN_UPDATE_SNAPSHOTS=1 to generate.",
            expected_path.display()
        );

        let expected = fs_anyhow::read_to_string(&expected_path)
            .unwrap()
            .replace("\r\n", "\n");
        // Strip the AT generated header and trim whitespace so the expected
        // files can keep the header for tooling without affecting comparison.
        let expected = expected
            .strip_prefix(&format!("# @{}generated\n", "")) // Avoid this file from being recognized as generated
            .unwrap_or(&expected)
            .trim()
            .to_owned();
        let actual = actual.replace("\r\n", "\n").trim().to_owned();

        pretty_assertions::assert_str_eq!(
            expected,
            actual,
            "Stub mismatch for {test_name}.\nTo update, run with STUBGEN_UPDATE_SNAPSHOTS=1."
        );
    }

    #[test]
    fn test_stubgen_functions() {
        assert_stubgen_snapshot("functions");
    }

    #[test]
    fn test_stubgen_classes() {
        assert_stubgen_snapshot("classes");
    }

    #[test]
    fn test_stubgen_variables() {
        assert_stubgen_snapshot("variables");
    }

    #[test]
    fn test_stubgen_imports() {
        assert_stubgen_snapshot("imports");
    }

    #[test]
    fn test_stubgen_mixed() {
        assert_stubgen_snapshot("mixed");
    }

    #[test]
    fn test_stubgen_overloads() {
        assert_stubgen_snapshot("overloads");
    }

    #[test]
    fn test_stubgen_typevar() {
        assert_stubgen_snapshot("typevar");
    }

    #[test]
    fn test_stubgen_type_alias_old_style() {
        assert_stubgen_snapshot("type_alias_old_style");
    }

    #[test]
    fn test_stubgen_generics() {
        assert_stubgen_snapshot("generics");
    }

    #[test]
    fn test_stubgen_docstrings() {
        let input = r#"
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

def no_doc(x: int) -> int:
    return x

class MyClass:
    """A class with a docstring."""

    def method(self) -> None:
        """Do something."""
        pass
"#;
        let config = ExtractConfig {
            include_private: false,
            include_docstrings: true,
        };
        let actual = run_stubgen_with_config(input, &config);
        assert!(
            actual.contains(r#""""Say hello.""""#),
            "Function docstring should be emitted:\n{actual}"
        );
        assert!(
            actual.contains(r#""""A class with a docstring.""""#),
            "Class docstring should be emitted:\n{actual}"
        );
        assert!(
            actual.contains(r#""""Do something.""""#),
            "Method docstring should be emitted:\n{actual}"
        );

        // Without docstrings, none should appear.
        let no_doc_config = ExtractConfig {
            include_private: false,
            include_docstrings: false,
        };
        let without = run_stubgen_with_config(input, &no_doc_config);
        assert!(
            !without.contains("Say hello."),
            "Function docstring should not appear with include_docstrings=false:\n{without}"
        );
        assert!(
            !without.contains("A class with a docstring."),
            "Class docstring should not appear with include_docstrings=false:\n{without}"
        );
    }
}
