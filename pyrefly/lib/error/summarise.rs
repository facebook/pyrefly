/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use dupe::Dupe;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display;
use starlark_map::small_map::SmallMap;

use crate::config::error_kind::ErrorKind;
use crate::error::error::Error;

type ErrorCounts = SmallMap<ErrorKind, usize>;

/// Returns a Vec of (T, error_counts) where the Vec is sorted by the total
/// error count, descending, and the error_counts are also sorted by the error
/// count, descending.
fn collect_and_sort<I, T>(errors: I) -> Vec<(T, Vec<(ErrorKind, usize)>)>
where
    I: IntoIterator<Item = (T, ErrorCounts)>,
{
    let mut errors = errors
        .into_iter()
        .map(|(p, error_counts)| {
            let mut error_kind_counts = error_counts.into_iter().collect::<Vec<_>>();
            error_kind_counts.sort_by_key(|(_, c)| -(*c as isize));
            (p, error_kind_counts)
        })
        .collect::<Vec<_>>();
    errors.sort_by_key(|(_, m)| -(m.iter().map(|x| x.1).sum::<usize>() as isize));
    errors
}

/// Groups the errors by the path_index'th component of the path by error_kind,
/// and then collects and sorts the errors as in collect_and_sort.
pub fn get_top_error_dirs_by_error_kind(
    path_errors: &SmallMap<ModulePath, ErrorCounts>,
    path_index: usize,
) -> Vec<(PathBuf, Vec<(ErrorKind, usize)>)> {
    let mut dirs: SmallMap<_, ErrorCounts> = SmallMap::new();
    for (path, errors) in path_errors {
        let dir = PathBuf::from(path.to_string())
            .components()
            .take(path_index + 1)
            .collect::<PathBuf>();

        let error_counts = dirs.entry(dir).or_default();
        for (kind, count) in errors {
            *error_counts.entry(*kind).or_default() += count;
        }
    }
    collect_and_sort(dirs)
}

/// Groups the errors by the path_index'th component of the path, using config_path if provided,
/// and returns a vector of (PathBuf, usize) pairs sorted by error count in descending order.
///
/// If config_path is provided, only files that belong to the config path's parent directory
/// will be included in the results.
fn get_top_error_dirs(
    path_errors: &SmallMap<ModulePath, ErrorCounts>,
    path_index: usize,
    config_path: Option<&PathBuf>,
) -> Vec<(PathBuf, usize)> {
    let mut dirs: SmallMap<PathBuf, usize> = SmallMap::new();

    // If config_path is provided, get its parent directory
    let config_dir = config_path.and_then(|p| p.parent());

    for (path, errors) in path_errors {
        let file_path = PathBuf::from(path.to_string());

        // If config_dir is provided, only include files that are relative to it
        if let Some(config_dir) = config_dir {
            if let Ok(rel_path) = file_path.strip_prefix(config_dir) {
                // Calculate the directory based on config_path and path_index
                let mut dir = config_dir.to_path_buf();
                for component in rel_path.components().take(path_index + 1) {
                    dir.push(component);
                }

                let total_count: usize = errors.values().sum();
                *dirs.entry(dir).or_default() += total_count;
            }
        } else {
            // If no config_path is provided, use the original behavior
            let dir = file_path
                .components()
                .take(path_index + 1)
                .collect::<PathBuf>();
            let total_count: usize = errors.values().sum();
            *dirs.entry(dir).or_default() += total_count;
        }
    }

    let mut result: Vec<(PathBuf, usize)> = dirs.into_iter().collect();
    result.sort_by_key(|(_, count)| -(*count as isize));
    result
}

fn get_errors_per_file(errors: &[Error]) -> SmallMap<ModulePath, SmallMap<ErrorKind, usize>> {
    let mut map: SmallMap<ModulePath, SmallMap<ErrorKind, usize>> = SmallMap::new();
    for err in errors {
        *map.entry(err.path().dupe())
            .or_default()
            .entry(err.error_kind())
            .or_default() += 1;
    }
    map
}

/// Returns the top 10 directories by error count, specifically formatted for the init command.
/// This function automatically determines the optimal path_index by trying multiple values,
/// starting with path_index = 0 and incrementally increasing it until it finds 10 directories
/// with error counts between 10-100, or until it reaches a maximum number of attempts.
///
/// If config_path is provided, only files that belong to the config path's parent directory
/// will be included in the results.
///
/// In cases where multiple path_index values produce directories with 10-100 errors but none reach 10 directories,
/// the function will use results from the path_index that produced the most such directories.
///
/// Returns a tuple containing:
/// - The best path_index used for grouping
/// - A vector of (PathBuf, usize) pairs representing directories and their error counts
pub fn get_top_error_dirs_for_init(
    errors: &[Error],
    config_path: Option<&PathBuf>,
) -> (usize, Vec<(PathBuf, usize)>) {
    let path_errors = get_errors_per_file(errors);

    // Maximum number of path_index increments to try
    const MAX_ATTEMPTS: usize = 10;

    // Store filtered directories for each path_index
    let mut filtered_dirs_by_index: Vec<(usize, Vec<(PathBuf, usize)>)> = Vec::new();

    // Try increasing path_index until we find 10 directories with 10-100 errors
    // or until we reach the maximum number of attempts
    for current_path_index in 0..MAX_ATTEMPTS {
        let top_dirs = get_top_error_dirs(&path_errors, current_path_index, config_path);

        // Filter directories with 10-100 errors (for optimization)
        let dirs_with_desired_errors: Vec<_> = top_dirs
            .into_iter()
            .filter(|(_, error_count)| *error_count >= 10 && *error_count <= 100)
            .collect();

        // Store the filtered directories for this path_index
        filtered_dirs_by_index.push((current_path_index, dirs_with_desired_errors));

        // If we have at least 10 directories for this path_index, we can stop and use these results
        if filtered_dirs_by_index.last().unwrap().1.len() >= 10 {
            break;
        }
    }

    // Find the path_index that produced the most directories with 10-100 errors
    // If there's a tie, use the lower path_index as a tie breaker
    let best_result = filtered_dirs_by_index
        .into_iter()
        .max_by_key(|(idx, dirs)| (dirs.len(), -((*idx) as isize)));

    // Use the best path_index if found, otherwise default to path_index = 0
    let best_path_index = best_result.map_or(0, |(idx, _)| idx);

    // Get all directories for the best path_index, but filter to only show those with <= 100 errors
    let all_dirs = get_top_error_dirs(&path_errors, best_path_index, config_path);
    let dirs_to_show: Vec<_> = all_dirs
        .into_iter()
        .filter(|(_, error_count)| *error_count <= 100)
        .take(10)
        .collect();

    (best_path_index, dirs_to_show)
}

/// Prints a summary of errors found in the input.
/// The summary shows the top directories by error count, top files by error count,
/// and the top errors by count.
/// path_index controls how directories are grouped. For example, for the directory /alpha/beta/gamma/...,
/// path_index = 0 groups by /alpha, path_index = 1 groups by /alpha/beta,
/// and path_index = 2 groups by alpha/beta/gamma.
/// If the path_index is larger than the number of components in the path, then the entire path is used.
pub fn print_error_summary(errors: &[Error], path_index: usize) {
    // TODO: Sort errors by count and then name. More consistent and human-readable.
    // TODO: Consider output formatting.
    let path_errors = get_errors_per_file(errors);
    eprintln!("=== Error Summary ===");
    if path_errors.is_empty() {
        eprintln!("No errors found!");
        return;
    }
    eprintln!("Top 30 Directories by Error Count:");
    let top_dirs = get_top_error_dirs_by_error_kind(&path_errors, path_index);
    for (dir, error_kind_counts) in top_dirs.into_iter().take(30) {
        let total_error_count = error_kind_counts.iter().fold(0, |count, (_, c)| count + *c);
        eprintln!(
            "{}: {}",
            dir.display(),
            display::count(total_error_count, "error")
        );
        for (kind, count) in error_kind_counts {
            eprintln!("  {}: {}", kind.to_name(), display::number_thousands(count));
        }
    }

    eprintln!("\nTop 30 Files by Error Count:");
    let top_files = collect_and_sort(path_errors.clone());
    for (path, error_kind_counts) in top_files.into_iter().take(30) {
        let total_error_count = error_kind_counts.iter().fold(0, |count, (_, c)| count + *c);
        eprintln!("{path}: {}", display::count(total_error_count, "error"));
        for (kind, count) in error_kind_counts {
            eprintln!("  {}: {}", kind.to_name(), display::number_thousands(count));
        }
    }

    eprintln!("\nTop Errors by Count:");
    let error_counts = {
        let mut ec = path_errors
            .values()
            .fold(SmallMap::<ErrorKind, usize>::new(), |mut acc, m| {
                for (kind, count) in m {
                    *acc.entry(*kind).or_default() += count;
                }
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();
        ec.sort_by_key(|(_, c)| -(*c as isize));
        ec
    };
    for (kind, count) in error_counts {
        eprintln!("{}: {}", kind.to_name(), display::count(count, "instance"));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyrefly_python::module_name::ModuleName;
    use ruff_python_ast::name::Name;

    use super::*;
    use crate::module::module_info::ModuleInfo;

    fn mpfs(s: &str) -> ModulePath {
        ModulePath::filesystem(PathBuf::from(s))
    }

    #[test]
    fn test_get_top_error_dirs_for_init() {
        // Helper function to create PathBuf from string
        fn pb(s: &str) -> PathBuf {
            PathBuf::from(s)
        }

        // Helper function to create multiple test errors for a file
        fn create_test_errors(path: &str, count: usize) -> Vec<Error> {
            let mut errors = Vec::with_capacity(count);
            for _ in 0..count {
                // Create the ModuleInfo in a single line with non-empty content
                let module_info = ModuleInfo::new(
                    ModuleName::from_name(&Name::new_static("test")),
                    mpfs(path),
                    Arc::new("dummy content".to_owned()),
                );

                // Use a default error kind - the specific kind doesn't matter for this test
                let error = Error::new(
                    module_info,
                    ruff_text_size::TextRange::new(
                        ruff_text_size::TextSize::from(0),
                        ruff_text_size::TextSize::from(1),
                    ),
                    vec1::vec1!["Test error".to_owned()],
                    ErrorKind::BadArgumentType,
                );
                errors.push(error);
            }
            errors
        }

        // Create all test errors at once using a more declarative approach
        let errors: Vec<Error> = vec![
            create_test_errors("base/proj/sub/a.py", 5),
            create_test_errors("base/proj/sub/b.py", 20),
            create_test_errors("base/proj/dub/z.py", 40),
            create_test_errors("base/other/x.py", 90),
            // 5 errors - too few to be included in optimal results
            create_test_errors("base/other/y.py", 15),
            // 150 errors - too many to be included in optimal results
            create_test_errors("another/dir/file.py", 150),
        ]
        .into_iter()
        .flatten()
        .collect();

        // Test without config_path
        let (path_index, dirs) = get_top_error_dirs_for_init(&errors, None);

        // The function should choose path_index = 3 as optimal
        assert_eq!(path_index, 2);

        // Verify the top directories
        assert_eq!(dirs.len(), 4); // Should have 4 directories with <= 100 errors

        // Check the order and error counts
        assert_eq!(dirs[0].0, pb("base/other/x.py"));
        assert_eq!(dirs[0].1, 90);

        assert_eq!(dirs[1].0, pb("base/proj/dub"));
        assert_eq!(dirs[1].1, 40);

        assert_eq!(dirs[2].0, pb("base/proj/sub"));
        assert_eq!(dirs[2].1, 25); // 5 + 20 errors

        assert_eq!(dirs[3].0, pb("base/other/y.py"));
        assert_eq!(dirs[3].1, 15);

        // Test with config_path
        let config_path = pb("base/proj/pyrefly.toml");
        let (path_index_with_config, dirs_with_config) =
            get_top_error_dirs_for_init(&errors, Some(&config_path));

        // With config_path, it should only include files under base/proj
        assert_eq!(path_index_with_config, 0);

        // Verify the directories
        assert!(dirs_with_config.len() >= 2); // Should have at least 2 directories

        // Check the order and error counts
        assert_eq!(dirs_with_config[0].0, pb("base/proj/dub"));
        assert_eq!(dirs_with_config[0].1, 40);

        assert_eq!(dirs_with_config[1].0, pb("base/proj/sub"));
        assert_eq!(dirs_with_config[1].1, 25); // 5 + 20 errors
    }

    #[test]
    fn test_collect_and_sort() {
        let errors: Vec<(usize, SmallMap<ErrorKind, usize>)> = vec![
            (3, SmallMap::from_iter([(ErrorKind::AsyncError, 1)])),
            (
                2,
                SmallMap::from_iter([(ErrorKind::ReadOnly, 1), (ErrorKind::AsyncError, 2)]),
            ),
            (
                1,
                SmallMap::from_iter([
                    (ErrorKind::ReadOnly, 2),
                    (ErrorKind::AsyncError, 1),
                    (ErrorKind::ParseError, 3),
                ]),
            ),
        ];
        let want = vec![
            (
                1,
                vec![
                    (ErrorKind::ParseError, 3),
                    (ErrorKind::ReadOnly, 2),
                    (ErrorKind::AsyncError, 1),
                ],
            ),
            (
                2,
                vec![(ErrorKind::AsyncError, 2), (ErrorKind::ReadOnly, 1)],
            ),
            (3, vec![(ErrorKind::AsyncError, 1)]),
        ];
        let got = collect_and_sort(errors);
        assert_eq!(want, got);
    }

    #[test]
    fn test_get_top_error_dirs_by_error_kind() {
        fn pb(s: &str) -> PathBuf {
            PathBuf::from(s)
        }
        let errors = SmallMap::from_iter(vec![
            (
                mpfs("base/proj/sub/a.py"),
                SmallMap::from_iter([(ErrorKind::ReadOnly, 2)]),
            ),
            (
                mpfs("base/proj/sub/b.py"),
                SmallMap::from_iter([(ErrorKind::AsyncError, 3)]),
            ),
            (
                mpfs("base/proj/dub/z.py"),
                SmallMap::from_iter([(ErrorKind::ReadOnly, 4)]),
            ),
            (
                mpfs("base/short.py"),
                SmallMap::from_iter([(ErrorKind::AnnotationMismatch, 10)]),
            ),
        ]);

        let want = vec![(
            pb("base"),
            vec![
                (ErrorKind::AnnotationMismatch, 10),
                (ErrorKind::ReadOnly, 6),
                (ErrorKind::AsyncError, 3),
            ],
        )];
        assert_eq!(want, get_top_error_dirs_by_error_kind(&errors, 0));

        let want = vec![
            (
                pb("base/short.py"),
                vec![(ErrorKind::AnnotationMismatch, 10)],
            ),
            (
                pb("base/proj"),
                vec![(ErrorKind::ReadOnly, 6), (ErrorKind::AsyncError, 3)],
            ),
        ];
        assert_eq!(want, get_top_error_dirs_by_error_kind(&errors, 1));

        let want = vec![
            (
                pb("base/short.py"),
                vec![(ErrorKind::AnnotationMismatch, 10)],
            ),
            (
                pb("base/proj/sub"),
                vec![(ErrorKind::AsyncError, 3), (ErrorKind::ReadOnly, 2)],
            ),
            (pb("base/proj/dub"), vec![(ErrorKind::ReadOnly, 4)]),
        ];
        assert_eq!(want, get_top_error_dirs_by_error_kind(&errors, 2));

        let want = vec![
            (
                pb("base/short.py"),
                vec![(ErrorKind::AnnotationMismatch, 10)],
            ),
            (pb("base/proj/dub/z.py"), vec![(ErrorKind::ReadOnly, 4)]),
            (pb("base/proj/sub/b.py"), vec![(ErrorKind::AsyncError, 3)]),
            (pb("base/proj/sub/a.py"), vec![(ErrorKind::ReadOnly, 2)]),
        ];
        assert_eq!(want, get_top_error_dirs_by_error_kind(&errors, 3));
        assert_eq!(want, get_top_error_dirs_by_error_kind(&errors, 30000));
    }

    #[test]
    fn test_get_top_error_dirs() {
        fn pb(s: &str) -> PathBuf {
            PathBuf::from(s)
        }
        let errors = SmallMap::from_iter(vec![
            (
                mpfs("base/proj/sub/a.py"),
                SmallMap::from_iter([(ErrorKind::ReadOnly, 2)]),
            ),
            (
                mpfs("base/proj/sub/b.py"),
                SmallMap::from_iter([(ErrorKind::AsyncError, 3)]),
            ),
            (
                mpfs("base/proj/dub/z.py"),
                SmallMap::from_iter([(ErrorKind::ReadOnly, 4)]),
            ),
            (
                mpfs("base/short.py"),
                SmallMap::from_iter([(ErrorKind::AnnotationMismatch, 10)]),
            ),
        ]);

        // Test with path_index = 0 (group by first component)
        let want = vec![(pb("base"), 19)];
        assert_eq!(want, get_top_error_dirs(&errors, 0, None));

        // Test with path_index = 1 (group by first two components)
        let want = vec![(pb("base/short.py"), 10), (pb("base/proj"), 9)];
        assert_eq!(want, get_top_error_dirs(&errors, 1, None));

        // Test with path_index = 2 (group by first three components)
        let want = vec![
            (pb("base/short.py"), 10),
            (pb("base/proj/sub"), 5),
            (pb("base/proj/dub"), 4),
        ];
        assert_eq!(want, get_top_error_dirs(&errors, 2, None));

        // Test with path_index = 3 and beyond (group by full path)
        let want = vec![
            (pb("base/short.py"), 10),
            (pb("base/proj/dub/z.py"), 4),
            (pb("base/proj/sub/b.py"), 3),
            (pb("base/proj/sub/a.py"), 2),
        ];
        assert_eq!(want, get_top_error_dirs(&errors, 3, None));
        assert_eq!(want, get_top_error_dirs(&errors, 30000, None));

        // Test with config_path
        let config_path = pb("base/proj/pyrefly.toml");

        // Test with path_index = 0 and config_path (group by first component relative to config_path)
        let want = vec![(pb("base/proj/sub"), 5), (pb("base/proj/dub"), 4)];
        assert_eq!(want, get_top_error_dirs(&errors, 0, Some(&config_path)));

        // Test with path_index = 1 and config_path (group by first two components relative to config_path)
        let want = vec![
            (pb("base/proj/dub/z.py"), 4),
            (pb("base/proj/sub/b.py"), 3),
            (pb("base/proj/sub/a.py"), 2),
        ];
        assert_eq!(want, get_top_error_dirs(&errors, 1, Some(&config_path)));

        // Test with files not relative to config_path
        let want: Vec<(PathBuf, usize)> = vec![];
        assert_eq!(
            want,
            get_top_error_dirs(&errors, 0, Some(&pb("other/path/pyrefly.toml")))
        );
    }
}
