use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_name::ModuleNameWithKind;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::thread_pool::ThreadCount;

use crate::commands::config_finder::default_config_finder;
use crate::config::error_kind::ErrorKind;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;

pub struct Diagnostic {
    pub kind: ErrorKind,
    pub message: String,
    // Location of the code
    pub line: u32,
    pub column: u32,
}

pub struct Checker {
    state: State,
    snippet_path: ModulePath,
}

impl Checker {
    pub fn new(project_root: &Path) -> Self {
        let state = State::new(default_config_finder(None), ThreadCount::default());
        let snippet_path = ModulePath::memory(project_root.join("__pyrefly_snippet__.py"));
        Self {
            state,
            snippet_path,
        }
    }

    pub fn check(&mut self, code: &str) -> Vec<Diagnostic> {
        let module_name = ModuleName::from_str("__main__");
        let config = self.state.config_finder().python_file(
            ModuleNameWithKind::guaranteed(module_name),
            &self.snippet_path,
        );
        let handle = Handle::new(
            module_name,
            self.snippet_path.clone(),
            config.get_sys_info(),
        );

        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(vec![(
            PathBuf::from(self.snippet_path.as_path()),
            Some(Arc::new(FileContents::from_source(code.to_owned()))),
        )]);
        self.state.run_with_committing_transaction(
            transaction,
            std::slice::from_ref(&handle),
            Require::Everything,
            None,
            None,
        );

        self.state
            .transaction()
            .get_errors([&handle])
            .collect_errors()
            .ordinary
            .into_iter()
            .map(|e| {
                let range = e.display_range();
                Diagnostic {
                    kind: e.error_kind(),
                    message: e.msg(),
                    line: range.start.line_within_file().get(),
                    column: range.start.column().get(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::api::Checker;
    use crate::config::error_kind::ErrorKind;

    #[test]
    fn test_checker_checks_snippet_against_project_tree() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("pyrefly.toml"), "").unwrap();
        fs::write(root.path().join("helper.py"), "def greet() -> int: ...\n").unwrap();

        let mut checker = Checker::new(root.path());

        let errors = checker.check("from helper import greet\nx: str = greet()");
        assert_eq!(
            errors.iter().map(|e| e.kind).collect::<Vec<_>>(),
            vec![ErrorKind::BadAssignment],
        );

        // now this should work
        assert!(
            checker
                .check("from helper import greet\ny: int = greet()")
                .is_empty()
        );
    }
}
