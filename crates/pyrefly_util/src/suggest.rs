/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;
use strsim::levenshtein;

/// Pick the closest candidate to `missing`, preferring smaller `priority` on ties.
pub fn best_suggestion<'a, I>(missing: &Name, candidates: I) -> Option<Name>
where
    I: IntoIterator<Item = (&'a Name, usize)>,
{
    let missing_str = missing.as_str();
    let mut best: Option<(Name, usize, usize)> = None;
    for (candidate, priority) in candidates {
        let candidate_str = candidate.as_str();
        // Skip single-letter candidates to reduce noise
        if candidate_str.len() == 1 {
            continue;
        }
        let distance = levenshtein(missing_str, candidate_str);
        let max_distance = match missing_str.len().max(candidate_str.len()) {
            0..=4 => 1,
            5..=8 => 2,
            _ => 3,
        };
        if distance == 0 || distance > max_distance {
            continue;
        }
        match &best {
            Some((_, best_distance, best_priority))
                if distance > *best_distance
                    || (distance == *best_distance && priority >= *best_priority) => {}
            _ => best = Some((candidate.clone(), distance, priority)),
        }
    }
    best.map(|(name, _, _)| name)
}
