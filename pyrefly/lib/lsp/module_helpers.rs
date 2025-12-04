/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::qname::QName;
use pyrefly_types::types::Type;
use starlark_map::small_set::SmallSet;
use tracing::warn;

use crate::module::bundled::BundledStub;
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;
use crate::types::stdlib::Stdlib;

/// Convert to a path we can show to the user. The contents may not match the disk, but it has
/// to be basically right.
pub fn to_real_path(path: &ModulePath) -> Option<PathBuf> {
    match path.details() {
        ModulePathDetails::FileSystem(path)
        | ModulePathDetails::Memory(path)
        | ModulePathDetails::Namespace(path) => Some(path.to_path_buf()),
        ModulePathDetails::BundledTypeshed(path) => {
            let typeshed = typeshed().ok()?;
            let typeshed_path = match typeshed.materialized_path_on_disk() {
                Ok(typeshed_path) => Some(typeshed_path),
                Err(err) => {
                    warn!("Builtins unable to be loaded on disk, {}", err);
                    None
                }
            }?;
            Some(typeshed_path.join(&**path))
        }
        ModulePathDetails::BundledTypeshedThirdParty(path) => {
            let typeshed_third_party = typeshed_third_party().ok()?;
            let typeshed_path = match typeshed_third_party.materialized_path_on_disk() {
                Ok(typeshed_path) => Some(typeshed_path),
                Err(err) => {
                    warn!("Third Party Stubs unable to be loaded on disk, {}", err);
                    None
                }
            }?;
            Some(typeshed_path.join(&**path))
        }
    }
}

#[allow(dead_code)]
pub fn collect_symbol_def_paths(t: &Type) -> Vec<(QName, PathBuf)> {
    collect_symbol_def_paths_with_tuple_qname(t, None)
}

pub fn collect_symbol_def_paths_with_stdlib(t: &Type, stdlib: &Stdlib) -> Vec<(QName, PathBuf)> {
    collect_symbol_def_paths_with_tuple_qname(t, Some(stdlib.tuple_object().qname().clone()))
}

fn collect_symbol_def_paths_with_tuple_qname(
    t: &Type,
    tuple_qname: Option<QName>,
) -> Vec<(QName, PathBuf)> {
    let mut tracked_def_locs = SmallSet::new();
    t.universe(&mut |t| {
        if let Some(qname) = t.qname() {
            tracked_def_locs.insert(qname.clone());
        }
    });

    if let Some(tuple_qname) = tuple_qname {
        let mut has_tuple = false;
        t.universe(&mut |ty| {
            if matches!(ty, Type::Tuple(_)) {
                has_tuple = true;
            }
        });
        if has_tuple {
            tracked_def_locs.insert(tuple_qname);
        }
    }

    tracked_def_locs
        .into_iter()
        .map(|qname| {
            let module_path = qname.module_path();
            let file_path = match module_path.details() {
                ModulePathDetails::BundledTypeshed(_)
                | ModulePathDetails::BundledTypeshedThirdParty(_) => {
                    to_real_path(module_path).unwrap_or_else(|| module_path.as_path().to_path_buf())
                }
                _ => module_path.as_path().to_path_buf(),
            };
            (qname, file_path)
        })
        .collect()
}
