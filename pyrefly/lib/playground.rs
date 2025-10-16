/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::num::NonZeroU32;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use dupe::Dupe;
use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::SourceDatabase;
use pyrefly_build::source_db::Target;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::prelude::VecExt;
use ruff_text_size::TextSize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::config::config::ConfigFile;
use crate::config::error_kind::Severity;
use crate::config::finder::ConfigFinder;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;

#[derive(Debug, Clone)]
struct PlaygroundSourceDatabase {
    module_mappings: SmallMap<ModuleName, ModulePath>,
    sys_info: SysInfo,
}

impl PlaygroundSourceDatabase {
    fn new(module_mappings: SmallMap<ModuleName, ModulePath>, sys_info: SysInfo) -> Self {
        Self {
            module_mappings,
            sys_info,
        }
    }
}

impl SourceDatabase for PlaygroundSourceDatabase {
    fn modules_to_check(&self) -> Vec<Handle> {
        self.module_mappings
            .iter()
            .map(|(module_name, module_path)| {
                Handle::new(*module_name, module_path.dupe(), self.sys_info.dupe())
            })
            .collect()
    }

    fn lookup(
        &self,
        module_name: &ModuleName,
        _: Option<&Path>,
        _: Option<ModuleStyle>,
    ) -> Option<ModulePath> {
        self.module_mappings.get(module_name).cloned()
    }

    fn handle_from_module_path(&self, path: ModulePath) -> Option<Handle> {
        // It should be fine to just iterate through this naively, since there generally
        // shouldn't be too many files open in the web editor.
        let (name, _) = self.module_mappings.iter().find(|(_, p)| *p == &path)?;
        Some(Handle::new(name.dupe(), path, self.sys_info.dupe()))
    }

    fn requery_source_db(&self, _: SmallSet<PathBuf>) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn get_critical_files(&self) -> SmallSet<PathBuf> {
        self.module_mappings
            .values()
            .map(|p| p.as_path().to_path_buf())
            .collect()
    }

    fn get_target(&self, _: Option<&Path>) -> Option<Target> {
        None
    }
}

#[derive(Serialize)]
pub struct Position {
    #[serde(rename(serialize = "column"))]
    pub column: i32,
    #[serde(rename(serialize = "lineNumber"))]
    pub line: i32,
}

impl Position {
    pub fn new(line: i32, column: i32) -> Self {
        Self { line, column }
    }

    fn from_display_pos(position: DisplayPos) -> Self {
        Self {
            line: position.line.get() as i32,
            column: position.column.get() as i32,
        }
    }

    // This should always succeed, but we are being conservative
    fn to_display_pos(&self) -> Option<DisplayPos> {
        Some(DisplayPos {
            line: LineNumber::new(u32::try_from(self.line).ok()?)?,
            column: NonZeroU32::new(u32::try_from(self.column).ok()?)?,
        })
    }
}

#[derive(Serialize)]
pub struct Range {
    #[serde(rename(serialize = "startLineNumber"))]
    pub start_line: i32,
    #[serde(rename(serialize = "startColumn"))]
    pub start_col: i32,
    #[serde(rename(serialize = "endLineNumber"))]
    pub end_line: i32,
    #[serde(rename(serialize = "endColumn"))]
    pub end_col: i32,
}

impl Range {
    fn new(range: DisplayRange) -> Self {
        Self {
            start_line: range.start.line.get() as i32,
            start_col: range.start.column.get() as i32,
            end_line: range.end.line.get() as i32,
            end_col: range.end.column.get() as i32,
        }
    }
}

#[derive(Serialize, Clone)]
pub struct Diagnostic {
    #[serde(rename(serialize = "startLineNumber"))]
    pub start_line: i32,
    #[serde(rename(serialize = "startColumn"))]
    pub start_col: i32,
    #[serde(rename(serialize = "endLineNumber"))]
    pub end_line: i32,
    #[serde(rename(serialize = "endColumn"))]
    pub end_col: i32,
    pub message_header: String,
    pub message_details: String,
    pub kind: String,
    pub severity: i32,
    pub filename: String,
}

#[derive(Serialize)]
pub struct TypeQueryContent {
    language: String,
    value: String,
}

#[derive(Serialize)]
pub struct TypeQueryResult {
    contents: Vec<TypeQueryContent>,
}

#[derive(Serialize)]
pub struct AutoCompletionItem {
    label: String,
    detail: Option<String>,
    kind: Option<CompletionItemKind>,
    #[serde(rename(serialize = "sortText"))]
    sort_text: Option<String>,
}

#[derive(Serialize)]
pub struct InlayHint {
    label: String,
    position: Position,
}

pub struct Playground {
    state: State,
    handles: SmallMap<String, Handle>,
    active_filename: String,
    sys_info: SysInfo,
    config_finder: ConfigFinder,
    /// Diagnostics produced while loading/parsing configuration in the sandbox
    config_diagnostics: Vec<Diagnostic>,
}

impl Playground {
    pub fn new(python_version: Option<&str>) -> Result<Self, String> {
        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        config.interpreters.skip_interpreter_query = true;

        let sys_info = match python_version {
            Some(version_str) => {
                let parsed_version = PythonVersion::from_str(version_str)
                    .map_err(|e| format!("Invalid Python version '{version_str}': {e}"))?;
                config.python_environment.python_version = Some(parsed_version);
                SysInfo::new(parsed_version, PythonPlatform::linux())
            }
            None => SysInfo::default(),
        };

        config.configure();
        let config = ArcId::new(config);
        let config_finder = ConfigFinder::new_constant(config.dupe());

        let state = State::new(config_finder);
        let config_finder_for_self = ConfigFinder::new_constant(config.dupe());

        Ok(Self {
            state,
            handles: SmallMap::new(),
            active_filename: String::new(),
            sys_info,
            config_finder: config_finder_for_self,
            config_diagnostics: Vec::new(),
        })
    }

    pub fn update_sandbox_files(
        &mut self,
        files: SmallMap<String, String>,
        force_update: bool,
    ) -> Option<String> {
        self.config_diagnostics.clear();
        // Parse configuration if present in the in-memory files
        let mut parsed_config: Option<ConfigFile> = None;
        if let Some(cfg_str) = files.get("pyrefly.toml") {
            match toml::from_str::<ConfigFile>(cfg_str) {
                Ok(cfg) => parsed_config = Some(cfg),
                Err(err) => {
                    // Attach a diagnostic to pyrefly.toml on parse/validation failure
                    self.config_diagnostics.push(Diagnostic {
                        start_line: 1,
                        start_col: 1,
                        end_line: 1,
                        end_col: 1,
                        message_header: "TOML parse error".to_owned(),
                        message_details: err.to_string(),
                        kind: "parse-error".to_owned(),
                        // MarkerSeverity.Error (8)
                        severity: 8,
                        filename: "pyrefly.toml".to_owned(),
                    });
                    if !force_update {
                        return None;
                    }
                }
            }
        }

        self.handles.clear();
        // Base configuration: use parsed config if available, otherwise defaults
        let mut config = parsed_config.unwrap_or_default();
        config.python_environment.set_empty_to_default();
        config.interpreters.skip_interpreter_query = true;

        // Determine runtime SysInfo from config (if provided) or previous value
        let desired_version = config
            .python_environment
            .python_version
            .unwrap_or(self.sys_info.version());
        let desired_platform = self.sys_info.platform().clone();
        self.sys_info = SysInfo::new(desired_version, desired_platform.clone());
        config.python_environment.python_version = Some(self.sys_info.version());
        config.python_environment.python_platform = Some(desired_platform);

        // Build source DB from .py and .pyi files only
        let mut file_contents = Vec::new();
        let mut module_mappings = SmallMap::new();
        for (filename, content) in &files {
            if !filename.ends_with(".py") && !filename.ends_with(".pyi") {
                continue;
            }
            let suffix = if filename.ends_with(".pyi") {
                ".pyi"
            } else {
                ".py"
            };
            let module_name =
                ModuleName::from_str(filename.strip_suffix(suffix).unwrap_or(filename));
            let module_path = PathBuf::from(filename.clone());
            let memory_path = ModulePath::memory(module_path.clone());

            module_mappings.insert(module_name, memory_path.dupe());

            let handle = Handle::new(module_name, memory_path, self.sys_info.dupe());
            self.handles.insert(filename.clone(), handle);
            file_contents.push((module_path, Some(Arc::new(content.clone()))));
        }

        let source_db = PlaygroundSourceDatabase::new(module_mappings, self.sys_info.dupe());
        config.source_db = Some(Arc::new(Box::new(source_db)));

        config.configure();
        let config = ArcId::new(config);
        let new_config_finder = ConfigFinder::new_constant(config.dupe());

        self.state = State::new(new_config_finder);
        self.config_finder = ConfigFinder::new_constant(config.dupe());

        if self.handles.contains_key("sandbox.py") {
            self.active_filename = "sandbox.py".to_owned();
        } else if let Some((first_filename, _)) = self.handles.first() {
            self.active_filename = first_filename.clone();
        }

        let mut transaction = self
            .state
            .new_committable_transaction(Require::Exports, None);
        transaction.as_mut().set_memory(file_contents);

        let handles: Vec<Handle> = self.handles.values().map(|handle| handle.dupe()).collect();

        self.state
            .run_with_committing_transaction(transaction, &handles, Require::Everything);
        Some(format!(
            "{}.{}",
            desired_version.major, desired_version.minor
        ))
    }

    pub fn update_single_file(&mut self, filename: String, content: String) {
        if let Some(_handle) = self.handles.get(&filename) {
            let module_path = PathBuf::from(&filename);
            let file_content = vec![(module_path, Some(Arc::new(content)))];

            let mut transaction = self
                .state
                .new_committable_transaction(Require::Exports, None);
            transaction.as_mut().set_memory(file_content);

            let handles: Vec<Handle> = self.handles.values().map(|handle| handle.dupe()).collect();

            self.state
                .run_with_committing_transaction(transaction, &handles, Require::Everything);

            if self.handles.contains_key(&filename) {
                self.active_filename = filename;
            }
        }
    }

    pub fn set_active_file(&mut self, filename: &str) {
        if self.handles.contains_key(filename) {
            self.active_filename = filename.to_owned();
        }
    }

    pub fn get_errors(&self) -> Vec<Diagnostic> {
        let mut all_diagnostics = Vec::new();

        for (filename, handle) in &self.handles {
            let file_errors = self
                .state
                .transaction()
                .get_errors([handle])
                .collect_errors()
                .shown
                .into_map(|e| {
                    let range = e.display_range();
                    Diagnostic {
                        start_line: range.start.line.get() as i32,
                        start_col: range.start.column.get() as i32,
                        end_line: range.end.line.get() as i32,
                        end_col: range.end.column.get() as i32,
                        message_header: e.msg_header().to_owned(),
                        message_details: e.msg_details().unwrap_or("").to_owned(),
                        kind: e.error_kind().to_name().to_owned(),
                        // Severity values defined here: https://microsoft.github.io/monaco-editor/typedoc/enums/MarkerSeverity.html
                        severity: match e.error_kind().default_severity() {
                            Severity::Error => 8,
                            Severity::Warn => 4,
                            Severity::Info => 2,
                            Severity::Ignore => 1,
                        },
                        filename: filename.clone(),
                    }
                });
            all_diagnostics.extend(file_errors);
        }

        // Include any diagnostics gathered while loading config
        all_diagnostics.extend(self.config_diagnostics.iter().cloned());
        all_diagnostics
    }

    fn to_text_size(&self, transaction: &Transaction, pos: Position) -> Option<TextSize> {
        let handle = self.handles.get(&self.active_filename)?;
        let info = transaction.get_module_info(handle)?;
        Some(info.lined_buffer().from_display_pos(pos.to_display_pos()?))
    }

    pub fn query_type(&self, pos: Position) -> Option<TypeQueryResult> {
        let handle = self.handles.get(&self.active_filename)?;
        let transaction = self.state.transaction();
        let position = self.to_text_size(&transaction, pos)?;
        let t = transaction.get_type_at(handle, position)?;
        Some(TypeQueryResult {
            contents: vec![TypeQueryContent {
                language: "python".to_owned(),
                value: t.to_string(),
            }],
        })
    }

    pub fn goto_definition(&mut self, pos: Position) -> Option<Range> {
        let handle = self.handles.get(&self.active_filename)?;
        let transaction = self.state.transaction();
        let position = self.to_text_size(&transaction, pos)?;
        // TODO: Support goto multiple definitions
        transaction
            .goto_definition(handle, position)
            .into_iter()
            .next()
            .map(|r| Range::new(r.module.display_range(r.range)))
    }

    pub fn autocomplete(&self, pos: Position) -> Vec<AutoCompletionItem> {
        let handle = match self.handles.get(&self.active_filename) {
            Some(h) => h,
            None => return Vec::new(),
        };
        let transaction = self.state.transaction();
        self.to_text_size(&transaction, pos)
            .map_or(Vec::new(), |position| {
                transaction.completion(handle, position, Default::default())
            })
            .into_map(
                |CompletionItem {
                     label,
                     detail,
                     sort_text,
                     kind,
                     ..
                 }| AutoCompletionItem {
                    label,
                    detail,
                    kind,
                    sort_text,
                },
            )
    }

    pub fn inlay_hint(&self) -> Vec<InlayHint> {
        let handle = match self.handles.get(&self.active_filename) {
            Some(h) => h,
            None => return Vec::new(),
        };
        let transaction = self.state.transaction();
        transaction
            .get_module_info(handle)
            .zip(transaction.inlay_hints(handle, Default::default()))
            .map(|(info, hints)| {
                hints.into_map(|(position, label)| {
                    let position = Position::from_display_pos(info.display_pos(position));
                    InlayHint { label, position }
                })
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::error_kind::ErrorKind;

    #[test]
    fn test_regular_import() {
        let mut state = Playground::new(None).unwrap();
        let expected_errors: Vec<String> = Vec::new();

        let mut files = SmallMap::new();
        files.insert("main.py".to_owned(), "from typing import *".to_owned());
        state.update_sandbox_files(files, true);
        state.set_active_file("main.py");

        assert_eq!(
            state
                .get_errors()
                .into_iter()
                .map(|x| x.message_header)
                .collect::<Vec<_>>(),
            expected_errors,
        );
    }

    #[test]
    fn test_invalid_import() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();
        files.insert("main.py".to_owned(), "from t".to_owned());
        state.update_sandbox_files(files, true);
        state.set_active_file("main.py");

        let expected_headers = &[
            "Could not find import of `t`",
            "Parse error: Expected 'import', found newline",
        ];
        let expected_details = &[
            "  Looked in these locations:\n  Build system source database",
            "",
        ];
        let expected_error_kinds = &[ErrorKind::ImportError, ErrorKind::ParseError];

        assert_eq!(
            &state.get_errors().into_map(|x| x.message_header),
            expected_headers
        );
        assert_eq!(
            &state.get_errors().into_map(|x| x.message_details),
            expected_details
        );

        assert_eq!(
            state.get_errors().into_map(|x| x.kind),
            expected_error_kinds.map(|k| k.to_name()),
        );
    }

    #[test]
    fn test_cross_file_import() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();
        files.insert(
            "sandbox.py".to_owned(),
            "from utils import helper_function\nresult = helper_function()".to_owned(),
        );
        files.insert(
            "utils.py".to_owned(),
            "def helper_function() -> str:\n    return \"Hello from utils!\"".to_owned(),
        );

        state.update_sandbox_files(files, true);
        state.set_active_file("sandbox.py");

        let errors = state.get_errors();

        let import_errors: Vec<_> = errors.iter().filter(|e| e.kind == "ImportError").collect();

        assert_eq!(
            import_errors.len(),
            0,
            "Should have no import errors with cross-file support"
        );

        for error in &errors {
            assert!(!error.filename.is_empty(), "Error should include filename");
            assert!(
                error.filename.ends_with(".py") || error.filename.ends_with(".pyi"),
                "Filename should end with .py or .pyi"
            );
        }
    }

    #[test]
    fn test_multi_file_errors_with_filenames() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();
        files.insert(
            "sandbox.py".to_owned(),
            "from utils import get_number\nresult: str = get_number()".to_owned(),
        );
        files.insert(
            "utils.py".to_owned(),
            "def get_number() -> int:\n    return \"not a number\"".to_owned(),
        );

        state.update_sandbox_files(files, true);
        state.set_active_file("sandbox.py");

        let errors = state.get_errors();

        let _sandbox_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.filename == "sandbox.py")
            .collect();
        let _utils_errors: Vec<_> = errors.iter().filter(|e| e.filename == "utils.py").collect();

        assert!(
            !errors.is_empty(),
            "Should have some errors from type mismatches"
        );

        for error in &errors {
            assert!(!error.filename.is_empty(), "Error should include filename");
            assert!(
                error.filename == "sandbox.py" || error.filename == "utils.py",
                "Error filename should be one of the test files, got: {}",
                error.filename
            );
        }
    }

    #[test]
    fn test_incremental_update_with_cross_file_errors() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();
        files.insert(
            "sandbox.py".to_owned(),
            "from utils import get_number\nresult: int = get_number()".to_owned(),
        );
        files.insert(
            "utils.py".to_owned(),
            "def get_number() -> int:\n    return 42".to_owned(),
        );

        state.update_sandbox_files(files, true);
        state.set_active_file("sandbox.py");

        let errors = state.get_errors();
        assert_eq!(errors.len(), 0, "Should have no errors initially");

        state.update_single_file(
            "utils.py".to_owned(),
            "def get_number() -> int:\n    return \"not a number\"".to_owned(),
        );

        let errors_after_update = state.get_errors();

        let utils_errors: Vec<_> = errors_after_update
            .iter()
            .filter(|e| e.filename == "utils.py")
            .collect();

        assert!(
            !utils_errors.is_empty(),
            "Should detect error in utils.py after incremental update"
        );

        for error in &errors_after_update {
            assert!(!error.filename.is_empty(), "Error should include filename");
        }
    }

    #[test]
    fn test_pyi_stub_file_support() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();

        // Create a .pyi stub file
        files.insert(
            "mymodule.pyi".to_owned(),
            "def greet(name: str) -> str: ...".to_owned(),
        );

        // Create a .py file that uses the stub
        files.insert(
            "main.py".to_owned(),
            "from mymodule import greet\nresult: int = greet(\"test\")".to_owned(),
        );

        state.update_sandbox_files(files, true);
        state.set_active_file("main.py");

        let errors = state.get_errors();

        // Should have a type error: greet returns str, not int
        let type_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.message_header.contains("not assignable")
                    || e.message_header.contains("incompatible")
            })
            .collect();

        assert!(
            !type_errors.is_empty(),
            "Should detect type mismatch when using .pyi stub file"
        );

        // Verify that .pyi files are processed
        assert!(
            state.handles.contains_key("mymodule.pyi"),
            ".pyi file should be in handles"
        );
    }

    #[test]
    fn test_mixed_py_and_pyi_files() {
        let mut state = Playground::new(None).unwrap();
        let mut files = SmallMap::new();

        // Both .py and .pyi files should be accepted
        files.insert("module1.py".to_owned(), "x: int = 1".to_owned());
        files.insert("module2.pyi".to_owned(), "y: str".to_owned());
        files.insert(
            "pyrefly.toml".to_owned(),
            "python-version = \"3.12\"".to_owned(),
        );

        state.update_sandbox_files(files, true);

        assert_eq!(
            state.handles.len(),
            2,
            "Should have 2 module handles (.py and .pyi)"
        );
        assert!(state.handles.contains_key("module1.py"));
        assert!(state.handles.contains_key("module2.pyi"));
        assert!(
            !state.handles.contains_key("pyrefly.toml"),
            "Config file should not be a module"
        );
    }
}
