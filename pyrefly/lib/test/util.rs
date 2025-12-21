/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

use anstream::ColorChoice;
use anyhow::anyhow;
use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_config::error::ErrorDisplayConfig;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::prelude::SliceExt;
use pyrefly_util::thread_pool::ThreadCount;
use pyrefly_util::thread_pool::init_thread_pool;
use pyrefly_util::trace::init_tracing;
use ruff_python_ast::name::Name;
use ruff_source_file::LineIndex;
use ruff_source_file::OneIndexed;
use ruff_source_file::PositionEncoding;
use ruff_source_file::SourceLocation;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::binding::binding::KeyExport;
use crate::config::base::UntypedDefBehavior;
use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::print_errors;
use crate::module::finder::find_import;
use crate::state::errors::Errors;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::subscriber::TestSubscriber;
use crate::types::class::Class;
use crate::types::types::Type;

#[macro_export]
macro_rules! testcase {
    (bug = $explanation:literal, $name:ident, $imports:expr, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro($imports, $contents, file!(), line!() + 1)
        }
    };
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::util::TestEnv::new(),
                $contents,
                file!(),
                line!() + 1,
            )
        }
    };
    ($name:ident, $imports:expr, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro($imports, $contents, file!(), line!())
        }
    };
    ($name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::util::TestEnv::new(),
                $contents,
                file!(),
                line!(),
            )
        }
    };
}

fn default_path(module: ModuleName) -> PathBuf {
    PathBuf::from(format!("{}.py", module.as_str().replace('.', "/")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestEnv {
    modules: Vec<(ModuleName, ModulePath, Option<Arc<FileContents>>)>,
    version: PythonVersion,
    untyped_def_behavior: UntypedDefBehavior,
    infer_with_first_use: bool,
    site_package_path: Vec<PathBuf>,
    implicitly_defined_attribute_error: bool,
    implicit_any_error: bool,
    unannotated_return_error: bool,
    unannotated_parameter_error: bool,
    unannotated_attribute_error: bool,
    implicit_abstract_class_error: bool,
    open_unpacking_error: bool,
    missing_override_decorator_error: bool,
    default_require_level: Require,
}

impl TestEnv {
    pub fn new() -> Self {
        // We aim to init the tracing before now, but if not, better now than never
        init_test();
        TestEnv {
            modules: Vec::new(),
            version: PythonVersion::default(),
            untyped_def_behavior: UntypedDefBehavior::default(),
            infer_with_first_use: true,
            site_package_path: Vec::new(),
            implicitly_defined_attribute_error: false,
            implicit_any_error: false,
            unannotated_return_error: false,
            unannotated_parameter_error: false,
            unannotated_attribute_error: false,
            implicit_abstract_class_error: false,
            open_unpacking_error: false,
            missing_override_decorator_error: false,
            default_require_level: Require::Exports,
        }
    }

    pub fn new_with_site_package_path(path: &str) -> Self {
        let mut res = Self::new();
        res.site_package_path = vec![PathBuf::from(path)];
        res
    }

    pub fn new_with_version(version: PythonVersion) -> Self {
        let mut res = Self::new();
        res.version = version;
        res
    }

    pub fn new_with_untyped_def_behavior(untyped_def_behavior: UntypedDefBehavior) -> Self {
        let mut res = Self::new();
        res.untyped_def_behavior = untyped_def_behavior;
        res
    }

    pub fn new_with_infer_with_first_use(infer_with_first_use: bool) -> Self {
        let mut res = Self::new();
        res.infer_with_first_use = infer_with_first_use;
        res
    }

    pub fn enable_implicitly_defined_attribute_error(mut self) -> Self {
        self.implicitly_defined_attribute_error = true;
        self
    }

    pub fn enable_implicit_any_error(mut self) -> Self {
        self.implicit_any_error = true;
        self
    }

    pub fn enable_unannotated_attribute_error(mut self) -> Self {
        self.unannotated_attribute_error = true;
        self
    }

    pub fn enable_unannotated_return_error(mut self) -> Self {
        self.unannotated_return_error = true;
        self
    }

    pub fn enable_unannotated_parameter_error(mut self) -> Self {
        self.unannotated_parameter_error = true;
        self
    }

    pub fn enable_implicit_abstract_class_error(mut self) -> Self {
        self.implicit_abstract_class_error = true;
        self
    }

    pub fn enable_open_unpacking_error(mut self) -> Self {
        self.open_unpacking_error = true;
        self
    }

    pub fn enable_missing_override_decorator_error(mut self) -> Self {
        self.missing_override_decorator_error = true;
        self
    }

    pub fn with_default_require_level(mut self, level: Require) -> Self {
        self.default_require_level = level;
        self
    }

    pub fn with_version(mut self, version: PythonVersion) -> Self {
        self.version = version;
        self
    }

    pub fn add_with_path(&mut self, name: &str, path: &str, code: &str) {
        assert!(
            path.ends_with(".py") || path.ends_with(".pyi") || path.ends_with(".rs"),
            "{path} doesn't look like a reasonable path"
        );
        self.modules.push((
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(path)),
            Some(Arc::new(FileContents::from_source(code.to_owned()))),
        ));
    }

    pub fn add(&mut self, name: &str, code: &str) {
        let module_name = ModuleName::from_str(name);
        let relative_path = ModulePath::memory(default_path(module_name));
        self.modules.push((
            module_name,
            relative_path,
            Some(Arc::new(FileContents::from_source(code.to_owned()))),
        ));
    }

    pub fn one(name: &str, code: &str) -> Self {
        let mut res = Self::new();
        res.add(name, code);
        res
    }

    pub fn one_with_path(name: &str, path: &str, code: &str) -> Self {
        let mut res = Self::new();
        res.add_with_path(name, path, code);
        res
    }

    pub fn add_real_path(&mut self, name: &str, path: PathBuf) {
        let module_name = ModuleName::from_str(name);
        self.modules
            .push((module_name, ModulePath::filesystem(path), None));
    }

    pub fn sys_info(&self) -> SysInfo {
        SysInfo::new(self.version, PythonPlatform::linux())
    }

    pub fn get_memory(&self) -> Vec<(PathBuf, Option<Arc<FileContents>>)> {
        self.modules
            .iter()
            .filter_map(|(_, path, contents)| match path.details() {
                ModulePathDetails::Memory(path) => Some(((**path).clone(), contents.dupe())),
                _ => None,
            })
            .collect()
    }

    pub fn config(&self) -> ArcId<ConfigFile> {
        let mut config = ConfigFile::default();
        config.python_environment.python_version = Some(self.version);
        config.python_environment.python_platform = Some(PythonPlatform::linux());
        config.python_environment.site_package_path = Some(self.site_package_path.clone());
        config.root.untyped_def_behavior = Some(self.untyped_def_behavior);
        config.root.infer_with_first_use = Some(self.infer_with_first_use);
        if config.root.errors.is_none() {
            config.root.errors = Some(ErrorDisplayConfig::new(HashMap::new()));
        };
        let errors = config.root.errors.as_mut().unwrap();
        if self.implicitly_defined_attribute_error {
            errors.set_error_severity(ErrorKind::ImplicitlyDefinedAttribute, Severity::Error);
        }
        if self.implicit_any_error {
            errors.set_error_severity(ErrorKind::ImplicitAny, Severity::Error);
        }
        if self.unannotated_attribute_error {
            errors.set_error_severity(ErrorKind::UnannotatedAttribute, Severity::Error);
        }
        if self.unannotated_return_error {
            errors.set_error_severity(ErrorKind::UnannotatedReturn, Severity::Error);
        }
        if self.unannotated_parameter_error {
            errors.set_error_severity(ErrorKind::UnannotatedParameter, Severity::Error);
        }
        if self.implicit_abstract_class_error {
            errors.set_error_severity(ErrorKind::ImplicitAbstractClass, Severity::Error);
        }
        if self.open_unpacking_error {
            errors.set_error_severity(ErrorKind::OpenUnpacking, Severity::Error);
        }
        if self.missing_override_decorator_error {
            errors.set_error_severity(ErrorKind::MissingOverrideDecorator, Severity::Error);
        }
        let mut sourcedb = MapDatabase::new(config.get_sys_info());
        for (name, path, _) in self.modules.iter() {
            sourcedb.insert(*name, path.dupe());
        }
        config.source_db = Some(ArcId::new(Box::new(sourcedb)));
        config.interpreters.skip_interpreter_query = true;
        config.configure();
        ArcId::new(config)
    }

    pub fn config_finder(&self) -> ConfigFinder {
        ConfigFinder::new_constant(self.config())
    }

    pub fn to_state(self) -> (State, impl Fn(&str) -> Handle) {
        let config = self.sys_info();
        let config_file = self.config();
        let handles = self
            .modules
            .iter()
            // Reverse so we start at the last file, which is likely to be what the user
            // would have opened, so make it most faithful.
            .rev()
            .map(|(x, path, _)| Handle::new(*x, path.dupe(), config.dupe()))
            .collect::<Vec<_>>();
        let state = State::new(self.config_finder());
        let subscriber = TestSubscriber::new();
        let mut transaction = state.new_committable_transaction(
            self.default_require_level,
            Some(Box::new(subscriber.dupe())),
        );
        transaction.as_mut().set_memory(self.get_memory());
        transaction.as_mut().run(&handles, Require::Everything);
        state.commit_transaction(transaction, None);
        subscriber.finish();
        let project_root = PathBuf::new();
        print_errors(
            project_root.as_path(),
            &state
                .transaction()
                .get_errors(handles.iter())
                .collect_errors()
                .shown,
        );
        (state, move |module| {
            let name = ModuleName::from_str(module);
            Handle::new(
                name,
                find_import(&config_file, name, None).finding().unwrap(),
                config.dupe(),
            )
        })
    }
}

pub fn code_frame_of_source_at_range(source: &str, range: TextRange) -> String {
    let index = LineIndex::from_source_text(source);
    let start_loc = index.line_column(range.start(), source);
    let end_loc = index.line_column(range.end(), source);
    if (range.start().checked_add(TextSize::from(1))) == Some(range.end()) {
        let full_line = source
            .lines()
            .nth(start_loc.line.to_zero_indexed())
            .unwrap();
        format!(
            "{} | {}\n{}   {}^",
            start_loc.line,
            full_line,
            " ".repeat(start_loc.line.to_string().len()),
            " ".repeat(start_loc.column.to_zero_indexed())
        )
    } else if start_loc.line == end_loc.line {
        let full_line = source
            .lines()
            .nth(start_loc.line.to_zero_indexed())
            .unwrap();
        format!(
            "{} | {}\n{}   {}{}",
            start_loc.line,
            full_line,
            " ".repeat(start_loc.line.to_string().len()),
            " ".repeat(start_loc.column.to_zero_indexed()),
            "^".repeat(std::cmp::max(
                end_loc.column.to_zero_indexed() - start_loc.column.to_zero_indexed(),
                1
            ))
        )
    } else {
        panic!("Computing multi-line code frame is unsupported for now.")
    }
}

pub fn code_frame_of_source_at_position(source: &str, position: TextSize) -> String {
    code_frame_of_source_at_range(
        source,
        TextRange::new(position, position.checked_add(TextSize::new(1)).unwrap()),
    )
}

/// Given `source`, this function will find all the positions pointed by the special `# ^` comments.
///
/// e.g. for
/// ```
/// Line 1: x = 42
/// Line 2: #    ^
/// ```
///
/// The position will be the position of `2` in Line 1.
pub fn extract_cursors_for_test(source: &str) -> Vec<TextSize> {
    let mut ranges = Vec::new();
    let mut prev_line = "";
    let index = LineIndex::from_source_text(source);
    for (line_index, line_str) in source.lines().enumerate() {
        for (row_index, _) in line_str.match_indices('^') {
            if prev_line.len() < row_index {
                panic!("Invalid cursor at {line_index}:{row_index}");
            }
            let position = index.offset(
                SourceLocation {
                    line: OneIndexed::from_zero_indexed(line_index - 1),
                    character_offset: OneIndexed::from_zero_indexed(row_index),
                },
                source,
                PositionEncoding::Utf32,
            );
            ranges.push(position);
        }
        prev_line = line_str;
    }
    ranges
}

pub fn mk_multi_file_state(
    files: &[(&'static str, &str)],
    default_require_level: Require,
    assert_zero_errors: bool,
) -> (HashMap<&'static str, Handle>, State) {
    let mut test_env = TestEnv::new();
    for (name, code) in files {
        test_env.add(name, code);
    }
    let (state, handle) = test_env
        .with_default_require_level(default_require_level)
        .to_state();
    let mut handles = HashMap::new();
    for (name, _) in files {
        handles.insert(*name, handle(name));
    }
    if assert_zero_errors {
        assert_eq!(
            state
                .transaction()
                .get_errors(handles.values())
                .collect_errors()
                .shown
                .len(),
            0
        );
    }
    let mut handles = HashMap::new();
    for (name, _) in files {
        handles.insert(*name, handle(name));
    }
    (handles, state)
}

pub fn mk_multi_file_state_assert_no_errors(
    files: &[(&'static str, &str)],
    default_require_level: Require,
) -> (HashMap<&'static str, Handle>, State) {
    mk_multi_file_state(files, default_require_level, true)
}

fn get_batched_lsp_operations_report_helper(
    files: &[(&'static str, &str)],
    assert_zero_errors: bool,
    get_report: impl Fn(&State, &Handle, TextSize) -> String,
) -> String {
    let (handles, state) = mk_multi_file_state(files, Require::indexing(), assert_zero_errors);
    let mut report = String::new();
    for (name, code) in files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        for position in extract_cursors_for_test(code) {
            report.push_str(&code_frame_of_source_at_position(code, position));
            report.push('\n');
            report.push_str(&get_report(&state, handle, position));
            report.push_str("\n\n");
        }
        report.push('\n');
    }

    report
}

/// Given a list of `files`, extract the location pointed by the special `#   ^` comments
/// (See `extract_cursors_for_test`), and perform the operation defined by `get_report`.
/// A human-readable report of the results of all specified operations will be returned.
pub fn get_batched_lsp_operations_report(
    files: &[(&'static str, &str)],
    get_report: impl Fn(&State, &Handle, TextSize) -> String,
) -> String {
    get_batched_lsp_operations_report_helper(files, true, get_report)
}

pub fn get_batched_lsp_operations_report_allow_error(
    files: &[(&'static str, &str)],
    get_report: impl Fn(&State, &Handle, TextSize) -> String,
) -> String {
    get_batched_lsp_operations_report_helper(files, false, get_report)
}

pub fn get_batched_lsp_operations_report_no_cursor(
    files: &[(&'static str, &str)],
    get_report: impl Fn(&State, &Handle) -> String,
) -> String {
    let (handles, state) = mk_multi_file_state(files, Require::indexing(), true);
    let mut report = String::new();
    for (name, _code) in files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        report.push('\n');
        report.push_str(&get_report(&state, handle));
        report.push_str("\n\n");
        report.push('\n');
    }

    report
}

pub fn init_test() {
    ColorChoice::write_global(ColorChoice::Always);
    init_tracing(true, true);
    // Enough threads to see parallelism bugs, but not too many to debug through.
    init_thread_pool(ThreadCount::NumThreads(NonZeroUsize::new(3).unwrap()));
}

/// Shared state with all the builtins already initialized (by a dummy module).
static SHARED_STATE: LazyLock<State> =
    LazyLock::new(|| TestEnv::one("_shared_state", "").to_state().0);

/// Should only be used from the `testcase!` macro.
pub fn testcase_for_macro(
    mut env: TestEnv,
    contents: &str,
    file: &str,
    line: u32,
) -> anyhow::Result<()> {
    init_test();
    let is_empty_env = env == TestEnv::new();
    let mut start_line = line as usize + 1;
    if !env.modules.is_empty() || !env.site_package_path.is_empty() {
        start_line += 1;
    }
    let contents = format!("{}{}", "\n".repeat(start_line), contents);
    env.add_with_path("main", file, &contents);
    // If any given test regularly takes > 20s, that's probably a bug.
    // Currently all are less than 3s in debug, even when running in parallel.
    let limit = 20;
    let check = |errors: Errors| {
        errors.check_against_expectations()?;
        errors.check_var_leak()?;
        Ok::<(), anyhow::Error>(())
    };
    for _ in 0..3 {
        let start = Instant::now();
        if is_empty_env {
            // Optimisation: For simple tests, just reuse the base state, to avoid rechecking stdlib.
            let mut t = SHARED_STATE.transaction();
            let h = Handle::new(
                ModuleName::from_str("main"),
                ModulePath::memory(PathBuf::from(file)),
                env.sys_info(),
            );
            t.set_memory(vec![(
                PathBuf::from(file),
                Some(Arc::new(FileContents::from_source(contents.clone()))),
            )]);
            t.run(&[h.dupe()], Require::Everything);
            let errors = t.get_errors([&h]);
            let project_root = PathBuf::new();
            print_errors(project_root.as_path(), &errors.collect_errors().shown);
            check(errors)?;
        } else {
            let (state, handle) = env.clone().to_state();
            let t = state.transaction();
            // First check against main, so we can capture any import order errors.
            check(t.get_errors(&[handle("main")]))?;
            // THen check all handles, so we make sure the rest of the TestEnv is valid.
            let handles = env.modules.map(|(x, _, _)| handle(x.as_str()));
            check(state.transaction().get_errors(handles.iter()))?;
        }
        if start.elapsed().as_secs() <= limit {
            return Ok(());
        }
        // Give a bit of a buffer if the machine is very busy
        sleep(Duration::from_secs(limit / 2));
    }
    Err(anyhow!("Test took too long (> {limit}s)"))
}

pub fn mk_state(code: &str) -> (Handle, State) {
    let (state, handle) = TestEnv::one("main", code).to_state();
    (handle("main"), state)
}

pub fn get_class(name: &str, handle: &Handle, state: &State) -> Class {
    let solutions = state.transaction().get_solutions(handle).unwrap();

    match &**solutions.get(&KeyExport(Name::new(name))) {
        Type::ClassDef(cls) => cls.dupe(),
        _ => unreachable!(),
    }
}

#[test]
fn test_inception() {
    assert!(testcase_for_macro(TestEnv::new(), "i: int = 'test'", file!(), line!()).is_err());
}
