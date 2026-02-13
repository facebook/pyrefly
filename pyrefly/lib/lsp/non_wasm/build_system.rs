/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::sync::LazyLock;

use pyrefly_python::COMPILED_FILE_SUFFIXES;
use pyrefly_python::PYTHON_EXTENSIONS;
use pyrefly_util::events::CategorizedEvents;
use starlark_map::small_set::SmallSet;

pub fn should_requery_build_system(events: &CategorizedEvents) -> bool {
    static PYTHON_SUFFIXES: LazyLock<SmallSet<OsString>> = LazyLock::new(|| {
        PYTHON_EXTENSIONS
            .iter()
            .chain(COMPILED_FILE_SUFFIXES.iter())
            .map(OsString::from)
            .collect()
    });

    events
        .iter()
        .any(|f| !f.extension().is_some_and(|e| PYTHON_SUFFIXES.contains(e)))
}
