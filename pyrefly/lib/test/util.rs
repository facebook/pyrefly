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
use starlark_map::small_map::SmallMap;

use crate::binding::binding::KeyExport;
use crate::config::base::UntypedDefBehavior;
use crate::config::config::ConfigFile;
use crate::config::finder::ConfigFinder;
use crate::error::error::print_errors;
use crate::module::finder::find_import;
use crate::state::handle::Handle;
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

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TestEnv {
    modules: SmallMap<ModuleName, (ModulePath, Option<Arc<String>>)>,
    version: PythonVersion,
    untyped_def_behavior: UntypedDefBehavior,
    site_package_path: Vec<PathBuf>,
}

impl TestEnv {
    pub fn new() -> Self {
        // We aim to init the tracing before now, but if not, better now than never
        init_test();
        Self::default()
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

    pub fn add_with_path(&mut self, name: &str, path: &str, code: &str) {
        assert!(
            path.ends_with(".py") || path.ends_with(".pyi") || path.ends_with(".rs"),
            "{path} doesn't look like a reasonable path"
        );
        self.modules.insert(
            ModuleName::from_str(name),
            (
                ModulePath::memory(PathBuf::from(path)),
                Some(Arc::new(code.to_owned())),
            ),
        );
    }

    pub fn add(&mut self, name: &str, code: &str) {
        let module_name = ModuleName::from_str(name);
        let relative_path = ModulePath::memory(default_path(module_name));
        self.modules.insert(
            module_name,
            (relative_path, Some(Arc::new(code.to_owned()))),
        );
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
            .insert(module_name, (ModulePath::filesystem(path), None));
    }

    pub fn sys_info(&self) -> SysInfo {
        SysInfo::new(self.version, PythonPlatform::linux())
    }

    pub fn get_memory(&self) -> Vec<(PathBuf, Option<Arc<String>>)> {
        self.modules
            .values()
            .filter_map(|(path, contents)| match path.details() {
                ModulePathDetails::Memory(path) => Some((path.clone(), contents.dupe())),
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
        for (name, (path, _)) in self.modules.iter() {
            config.custom_module_paths.insert(*name, path.clone());
        }
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
            .map(|(x, (path, _))| Handle::new(*x, path.clone(), config.dupe()))
            .collect::<Vec<_>>();
        let state = State::new(self.config_finder());
        let subscriber = TestSubscriber::new();
        let mut transaction =
            state.new_committable_transaction(Require::Exports, Some(Box::new(subscriber.dupe())));
        transaction.as_mut().set_memory(self.get_memory());
        transaction
            .as_mut()
            .run(&handles.map(|x| (x.dupe(), Require::Everything)));
        state.commit_transaction(transaction);
        subscriber.finish();
        print_errors(
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
                find_import(&config_file, name, None, true).unwrap(),
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
    assert_zero_errors: bool,
) -> (HashMap<&'static str, Handle>, State) {
    let mut test_env = TestEnv::new();
    for (name, code) in files {
        test_env.add(name, code);
    }
    let (state, handle) = test_env.to_state();
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
) -> (HashMap<&'static str, Handle>, State) {
    mk_multi_file_state(files, true)
}

fn get_batched_lsp_operations_report_helper(
    files: &[(&'static str, &str)],
    assert_zero_errors: bool,
    get_report: impl Fn(&State, &Handle, TextSize) -> String,
) -> String {
    let (handles, state) = mk_multi_file_state(files, assert_zero_errors);
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
    let (handles, state) = mk_multi_file_state(files, true);
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

/// Shared state with all the builtins already initialised (by a dummy module).
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
    let is_empty_env = env == TestEnv::default();
    let mut start_line = line as usize + 1;
    if !env.modules.is_empty() {
        start_line += 1;
    }
    let contents = format!("{}{}", "\n".repeat(start_line), contents);
    env.add_with_path("main", file, &contents);
    // If any given test regularly takes > 10s, that's probably a bug.
    // Currently all are less than 3s in debug, even when running in parallel.
    let limit = 10;
    for _ in 0..3 {
        let start = Instant::now();
        // Disable the empty test optimisation for now, because it doesn't catch breakages
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
                Some(Arc::new(contents.clone())),
            )]);
            t.run(&[(h.dupe(), Require::Everything)]);
            let errors = t.get_errors([&h]);
            print_errors(&errors.collect_errors().shown);
            errors.check_against_expectations()?;
        } else {
            let (state, handle) = env.clone().to_state();
            let t = state.transaction();
            // First check against main, so we can capture any import order errors.
            t.get_errors([&handle("main")])
                .check_against_expectations()?;
            // THen check all handles, so we make sure the rest of the TestEnv is valid.
            let handles = env
                .modules
                .keys()
                .map(|x| handle(x.as_str()))
                .collect::<Vec<_>>();
            state
                .transaction()
                .get_errors(handles.iter())
                .check_against_expectations()?;
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
