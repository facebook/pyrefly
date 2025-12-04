/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

/// Prints a summary of errors found in the input.
/// The summary shows the top directories by error count, top files by error count,
/// and the top errors by count.
/// path_index controls how directories are grouped. For example, for the directory /alpha/beta/gamma/...,
/// path_index = 0 groups by /alpha, path_index = 1 groups by /alpha/beta,
/// and path_index = 2 groups by alpha/beta/gamma.
/// If the path_index is larger than the number of components in the path, then the entire path is used.
pub fn print_error_summary(errors: &[Error]) {
    // TODO: Sort errors by count and then name. More consistent and human-readable.
    // TODO: Consider output formatting.
    let path_errors = get_errors_per_file(errors);
    eprintln!("=== Error Summary ===");
    if path_errors.is_empty() {
        eprintln!("No errors found!");
        return;
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
