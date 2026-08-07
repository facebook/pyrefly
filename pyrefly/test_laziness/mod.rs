/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Markdown-based snapshot tests for laziness properties.
//!
//! Each `.md` file in `test/laziness/` defines a test case with Python files
//! and expected output showing which steps and keys are computed for each module.
//! The test runner checks the actual output against the expected snapshot.
//! If they differ, the `.md` file is updated in-place and the test fails,
//! so the diff shows up in the working directory for review.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly::state::load::FileContents;
use pyrefly::state::require::Require;
use pyrefly::state::state::State;
use pyrefly::state::steps::Step;
use pyrefly::state::subscriber::TestSubscriber;
use pyrefly_build::handle::Handle;
use pyrefly_build::source_db::map_db::MapDatabase;
use pyrefly_config::config::ConfigFile;
use pyrefly_config::finder::ConfigFinder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::demand_tree::DemandCollector;
use pyrefly_util::demand_tree::DemandEdge;
use pyrefly_util::demand_tree::DemandKind;
use pyrefly_util::thread_pool::ThreadCount;
use starlark_map::small_map::SmallMap;

/// Parsed laziness test from a markdown file.
struct LazinessTest {
    /// Python files: (module_name, contents)
    files: Vec<(String, String)>,
    /// Which module(s) to check
    check_targets: Vec<String>,
}

/// Parse a laziness test from markdown content.
fn parse_test(content: &str) -> LazinessTest {
    let mut files: Vec<(String, String)> = Vec::new();
    let mut check_targets: Vec<String> = Vec::new();

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];

        // Look for file definitions: `module_name.py`:
        if line.starts_with('`') && line.ends_with("`:") && line.contains(".py") {
            let filename = line.trim_start_matches('`').trim_end_matches("`:").trim();
            let module_name = filename.trim_end_matches(".py").replace('/', ".");
            // Next line should be ```python
            i += 1;
            if i < lines.len() && lines[i].starts_with("```python") {
                i += 1;
                let mut code = String::new();
                while i < lines.len() && lines[i] != "```" {
                    if !code.is_empty() {
                        code.push('\n');
                    }
                    code.push_str(lines[i]);
                    i += 1;
                }
                files.push((module_name, code));
            }
        }

        // Look for check target: ## Check `module_name.py`
        if line.starts_with("## Check ") {
            let target = line
                .trim_start_matches("## Check ")
                .trim_start_matches('`')
                .trim_end_matches('`')
                .trim_end_matches(".py")
                .replace('/', ".");
            check_targets.push(target);
        }

        i += 1;
    }

    LazinessTest {
        files,
        check_targets,
    }
}

/// Run a laziness test and return the actual output as a string.
fn run_test(test: &LazinessTest) -> String {
    let collector = DemandCollector::new();

    let module_names: Vec<&str> = test.files.iter().map(|(n, _)| n.as_str()).collect();

    let mut config = ConfigFile::default();
    config.python_environment.set_empty_to_default();
    let mut sourcedb = MapDatabase::new(config.get_sys_info());
    for name in &module_names {
        sourcedb.insert(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(*name)),
        );
    }
    config.source_db = Some(ArcId::new(Box::new(sourcedb)));
    config.configure();
    let config = ArcId::new(config);

    // Single-threaded pyrefly execution for deterministic demand tree ordering.
    let thread_count = ThreadCount::NumThreads(NonZeroUsize::new(1).unwrap());
    let state = State::new(ConfigFinder::new_constant(config), thread_count);
    let subscriber = TestSubscriber::new();

    let mut transaction =
        state.new_committable_transaction(Require::Exports, Some(Box::new(subscriber.dupe())));
    transaction
        .as_mut()
        .set_demand_collector(Some(collector.clone()));

    for (name, contents) in &test.files {
        let contents = Arc::new(contents.clone());
        transaction.as_mut().set_memory(vec![(
            PathBuf::from(name),
            Some(Arc::new(FileContents::Source(contents))),
        )]);
    }

    let handles: Vec<_> = test
        .check_targets
        .iter()
        .map(|name| {
            Handle::new(
                ModuleName::from_str(name),
                ModulePath::memory(PathBuf::from(name.as_str())),
                SysInfo::default(),
            )
        })
        .collect();

    state.run_with_committing_transaction(transaction, &handles, Require::Errors, None, None);

    // Collect step info from subscriber
    let mut module_steps: SmallMap<String, Step> = SmallMap::new();
    for (handle, info) in subscriber.finish_detailed() {
        if let Some(step) = info.last_step {
            module_steps.insert(handle.module().as_str().to_owned(), step);
        }
    }

    // Format the output
    let mut output = String::new();
    // Sort modules: check targets first (as Solutions), then others alphabetically
    let mut sorted_modules: Vec<&str> = module_names.to_vec();
    sorted_modules.sort_by(|a, b| {
        let a_is_target = test.check_targets.iter().any(|t| t == a);
        let b_is_target = test.check_targets.iter().any(|t| t == b);
        match (a_is_target, b_is_target) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.cmp(b),
        }
    });

    for name in &sorted_modules {
        let step_str = module_steps
            .get(*name)
            .copied()
            .map_or("Nothing", Step::label);
        output.push_str(&format!("{}: {}\n", name, step_str));
    }

    // Append demand tree.
    // - Exclude roots where the source module is not a user module (stdlib→stdlib)
    // - Count and summarize builtin demands (user→builtins) instead of listing them
    // - Deduplicate identical root-level entries (e.g. repeated Exports demands)
    // - Filter out stdlib Exports children from answer demand edges
    let demand_roots = collector.take_roots();
    let user_set: HashSet<&str> = module_names.iter().copied().collect();
    let mut builtin_count: usize = 0;
    let mut seen_roots: HashSet<String> = HashSet::new();

    /// Builtins and typing are pervasive dependencies that would otherwise
    /// drown out interesting demands; we count them in aggregate.
    fn is_builtin_target(target: &str) -> bool {
        target == "builtins" || target == "typing"
    }

    /// Filter children of a demand edge: remove stdlib Exports demands
    /// since they're internal plumbing, not interesting cross-module demands.
    fn filter_children(node: &mut DemandEdge, builtin_count: &mut usize) {
        node.children.retain_mut(|child| {
            if is_builtin_target(&child.target) {
                *builtin_count += 1;
                return false;
            }
            filter_children(child, builtin_count);
            true
        });
    }

    let filtered: Vec<_> = demand_roots
        .into_iter()
        .filter_map(|mut node| {
            if !user_set.contains(node.from.as_str()) {
                return None; // stdlib→anything: exclude
            }
            if is_builtin_target(&node.target) {
                builtin_count += 1;
                return None; // user→builtins/typing: count but exclude
            }
            // Deduplicate identical leaf roots (e.g. repeated Exports demands).
            // Parent roots with children are never deduped — their subtree
            // content is significant even when labels match.
            if node.children.is_empty() && !seen_roots.insert(render_node_line(&node, 0)) {
                return None;
            }
            filter_children(&mut node, &mut builtin_count);
            Some(node)
        })
        .collect();

    if builtin_count > 0 || !filtered.is_empty() {
        output.push('\n');
    }
    if builtin_count > 0 {
        output.push_str(&format!("({builtin_count} builtin demands hidden)\n"));
    }
    for root in &filtered {
        render_tree(root, 0, &mut output);
    }

    output
}

/// Render a single demand node as one indented line for snapshot output.
fn render_node_line(node: &DemandEdge, depth: usize) -> String {
    let indent = "  ".repeat(depth);
    match &node.kind {
        DemandKind::Load { reason } => {
            format!("{indent}{} -> {}::Load({reason})", node.from, node.target)
        }
        DemandKind::Exports { reason } => {
            format!(
                "{indent}{} -> {}::Exports({reason})",
                node.from, node.target
            )
        }
        DemandKind::Answer { key } => {
            format!("{indent}{} -> {}::{key}", node.from, node.target)
        }
    }
}

/// Recursively render a node and its children into the output buffer.
fn render_tree(node: &DemandEdge, depth: usize, out: &mut String) {
    out.push_str(&render_node_line(node, depth));
    out.push('\n');
    for child in &node.children {
        render_tree(child, depth + 1, out);
    }
}

/// Replace the expected block in the markdown content with new content.
fn update_expected(content: &str, new_expected: &str) -> String {
    let mut result = String::new();
    let mut in_expected_block = false;
    let mut replaced = false;

    for line in content.lines() {
        if line.starts_with("```expected") {
            result.push_str(line);
            result.push('\n');
            result.push_str(new_expected);
            in_expected_block = true;
            replaced = true;
            continue;
        }
        if in_expected_block {
            if line == "```" {
                in_expected_block = false;
                result.push_str(line);
                result.push('\n');
            }
            // Skip old content
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }

    if !replaced {
        // No expected block found — append one
        result.push_str("\n```expected\n");
        result.push_str(new_expected);
        result.push_str("```\n");
    }

    result
}

/// Extract the expected output from the markdown content.
fn extract_expected(content: &str) -> String {
    let mut in_expected = false;
    let mut expected = String::new();
    for line in content.lines() {
        if line.starts_with("```expected") {
            in_expected = true;
            continue;
        }
        if in_expected {
            if line == "```" {
                break;
            }
            expected.push_str(line);
            expected.push('\n');
        }
    }
    expected
}

/// Map a test file path to the writable source file path.
/// The input `path` may point into a build-tool-specific read-only copy
/// (e.g. a buck sandbox); the source location is always derivable from
/// `file!()` and the test filename.
fn source_path_for(path: &Path) -> PathBuf {
    let filename = path.file_name().expect("test path must have a filename");
    // file!() is relative to the build tool's workspace root (repo root under
    // buck, crate root under cargo). In both cases, joining its parent with
    // the filename yields a path to the source .md file that writes land in.
    let source_dir = Path::new(file!())
        .parent()
        .expect("file!() must have a parent");
    source_dir.join(filename)
}

/// Produce a unified diff between two strings.
fn unified_diff(expected: &str, actual: &str) -> String {
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();
    let mut output = String::new();
    output.push_str("  --- expected\n  +++ actual\n");
    let max = expected_lines.len().max(actual_lines.len());
    for i in 0..max {
        match (expected_lines.get(i), actual_lines.get(i)) {
            (Some(e), Some(a)) if e == a => {
                output.push_str(&format!("  {e}\n"));
            }
            (Some(e), Some(a)) => {
                output.push_str(&format!("- {e}\n"));
                output.push_str(&format!("+ {a}\n"));
            }
            (Some(e), None) => {
                output.push_str(&format!("- {e}\n"));
            }
            (None, Some(a)) => {
                output.push_str(&format!("+ {a}\n"));
            }
            (None, None) => {}
        }
    }
    output
}

/// Run a single markdown laziness test file. Returns Ok(()) on match,
/// Ok(message) if the snapshot was updated, or Err(message) on mismatch.
fn run_laziness_test(path: &Path) -> Result<(), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let test = parse_test(&content);
    if test.files.is_empty() {
        return Err(format!("No Python files found in {}", path.display()));
    }
    if test.check_targets.is_empty() {
        return Err(format!("No check target found in {}", path.display()));
    }

    let actual = run_test(&test);
    let expected = extract_expected(&content);

    if actual == expected {
        return Ok(());
    }

    let source = source_path_for(path);

    // If UPDATE_SNAPSHOTS=1, auto-update the source file.
    if std::env::var("UPDATE_SNAPSHOTS").is_ok_and(|v| v == "1") && source.exists() {
        let source_content = std::fs::read_to_string(&source)
            .expect("failed to read source file for snapshot update");
        let updated = update_expected(&source_content, &actual);
        if std::fs::write(&source, &updated).is_ok() {
            return Err(format!(
                "Laziness snapshot updated: {}. Review the diff and commit.",
                source.display(),
            ));
        }
    }

    let diff = unified_diff(&expected, &actual);
    let test_name = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    let rerecord_cmd = if std::env::var("LAZINESS_TEST_PATH").is_ok() {
        format!(
            "buck test pyrefly:pyrefly_laziness_tests -- --env UPDATE_SNAPSHOTS=1 -r {test_name}"
        )
    } else {
        format!("UPDATE_SNAPSHOTS=1 cargo test {test_name} -- --test-threads=1")
    };
    // Print to stderr so buck renders with real newlines.
    eprintln!(
        "\nLaziness snapshot mismatch: {}\n\n{diff}\nTo re-record:\n  {rerecord_cmd}\n",
        source.display(),
    );
    Err(format!("snapshot mismatch: {}", source.display()))
}

/// Find the laziness test directory.
/// Tries multiple strategies: LAZINESS_TEST_PATH env var, file!() relative,
/// and cargo's current directory.
fn test_dir() -> PathBuf {
    if let Ok(path) = std::env::var("LAZINESS_TEST_PATH") {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }
    // Derive from file!() — works in buck where file paths are relative to workspace
    let source_file = Path::new(file!());
    let test_dir = source_file.parent().unwrap();
    if test_dir.exists() {
        return test_dir.to_path_buf();
    }
    // Cargo: try relative to current dir (cargo runs from crate root)
    let cargo_path = PathBuf::from("test_laziness");
    if cargo_path.exists() {
        return cargo_path;
    }
    panic!(
        "Laziness test directory not found. Tried:\n  - $LAZINESS_TEST_PATH\n  - {}\n  - lib/test/laziness\n\
         Set LAZINESS_TEST_PATH or run from the pyrefly crate directory.",
        test_dir.display(),
    );
}

/// Run a laziness test by filename. Each test uses its own
/// DemandCollector, so tests can run in parallel.
fn run_laziness_test_by_name(filename: &str) -> Result<(), String> {
    let dir = test_dir();
    let path = dir.join(filename);
    if !path.exists() {
        return Err(format!("Test file not found: {}", path.display()));
    }

    run_laziness_test(&path)
}

macro_rules! laziness_test {
    ($name:ident) => {
        #[test]
        fn $name() -> Result<(), String> {
            run_laziness_test_by_name(concat!(stringify!($name), ".md"))
        }
    };
}

// Generated at build time by build.rs — one laziness_test!() per test_*.md file.
include!(concat!(env!("OUT_DIR"), "/laziness_tests_generated.rs"));
