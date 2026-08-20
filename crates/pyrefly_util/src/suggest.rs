/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;
use strsim::levenshtein;

/// Pick the closest candidate to `missing` by edit distance, preferring smaller
/// `priority` on ties. The distance threshold scales with the longer of the two
/// strings (≤1 for short names, up to ≤3 for long ones); single-letter
/// candidates are skipped to reduce noise. `to_str` projects a candidate to the
/// string compared against `missing`. Returns `None` when nothing is close
/// enough.
fn closest<T, I>(missing: &str, candidates: I, to_str: impl Fn(&T) -> &str) -> Option<T>
where
    I: IntoIterator<Item = (T, usize)>,
{
    let mut best: Option<(T, usize, usize)> = None;
    for (candidate, priority) in candidates {
        let candidate_str = to_str(&candidate);
        // Skip single-letter candidates to reduce noise
        if candidate_str.len() == 1 {
            continue;
        }
        let distance = levenshtein(missing, candidate_str);
        let max_distance = match missing.len().max(candidate_str.len()) {
            0..=4 => 1,
            5..=8 => 2,
            _ => 3,
        };
        if distance == 0 || distance > max_distance {
            continue;
        }
        let is_better = match &best {
            Some((_, best_distance, best_priority)) => {
                distance < *best_distance
                    || (distance == *best_distance && priority < *best_priority)
            }
            None => true,
        };
        if is_better {
            best = Some((candidate, distance, priority));
        }
    }
    best.map(|(candidate, _, _)| candidate)
}

/// Pick the closest candidate to `missing`, preferring smaller `priority` on ties.
pub fn best_suggestion<'a, I>(missing: &Name, candidates: I) -> Option<Name>
where
    I: IntoIterator<Item = (&'a Name, usize)>,
{
    closest(missing.as_str(), candidates, |c| c.as_str()).cloned()
}

/// Like [`best_suggestion`], but over plain string slices (e.g. config keys).
pub fn best_suggestion_str<'a, I>(missing: &str, candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = (&'a str, usize)>,
{
    closest(missing, candidates, |c| *c)
}
