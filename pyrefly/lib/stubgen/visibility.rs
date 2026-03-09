/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Determines which names from a Python module should appear in the
//! generated `.pyi` stub, based on `__all__` or public-name heuristics.

use crate::export::definitions::Definitions;
use crate::export::definitions::DunderAllEntry;
use crate::export::definitions::DunderAllKind;

/// The set of names that should appear in the generated stub.
pub(crate) enum VisibilityFilter {
    /// `__all__` was explicitly defined; only these names are exported.
    Explicit(Vec<String>),
    /// No `__all__`; export all names that pass the privacy check.
    Inferred,
}

impl VisibilityFilter {
    pub(crate) fn from_definitions(defs: &Definitions) -> Self {
        match defs.dunder_all.kind {
            DunderAllKind::Specified => {
                let names = defs
                    .dunder_all
                    .entries
                    .iter()
                    .filter_map(|entry| match entry {
                        DunderAllEntry::Name(_, name) => Some(name.to_string()),
                        _ => None,
                    })
                    .collect();
                VisibilityFilter::Explicit(names)
            }
            _ => VisibilityFilter::Inferred,
        }
    }

    /// Should `name` appear in the stub?
    pub(crate) fn should_include(&self, name: &str, include_private: bool) -> bool {
        match self {
            VisibilityFilter::Explicit(names) => names.iter().any(|n| n == name),
            VisibilityFilter::Inferred => {
                // Dunder names (__init__, __str__, etc.) are always public.
                // Single underscore `_` is also public (commonly used as gettext alias).
                // Only single-leading-underscore names (_helper) are private.
                if name.starts_with("__") && name.ends_with("__") {
                    true
                } else if name.starts_with('_') && name != "_" {
                    include_private
                } else {
                    true
                }
            }
        }
    }
}
