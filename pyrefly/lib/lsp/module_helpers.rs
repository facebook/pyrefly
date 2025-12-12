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
use crate::module::third_party::get_bundled_third_party;
use crate::module::typeshed::typeshed;
use crate::module::typeshed_third_party::typeshed_third_party;

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
        ModulePathDetails::BundledThirdParty(path) => {
            let bundled_third_party = get_bundled_third_party().ok()?;
            let bundled_path = match bundled_third_party.materialized_path_on_disk() {
                Ok(bundled_path) => Some(bundled_path),
                Err(err) => {
                    warn!(
                        "Bundled Third Party Stubs unable to be loaded on disk, {}",
                        err
                    );
                    None
                }
            }?;
            Some(bundled_path.join(&**path))
        }
    }
}

pub fn collect_symbol_def_paths(t: &Type) -> Vec<(QName, PathBuf)> {
    let mut tracked_def_locs = SmallSet::new();
    t.universe(&mut |t| tracked_def_locs.extend(t.qname()));
    tracked_def_locs
        .into_iter()
        .map(|qname| {
            let module_path = qname.module_path();
            let file_path = match module_path.details() {
                ModulePathDetails::BundledTypeshed(_)
                | ModulePathDetails::BundledTypeshedThirdParty(_)
                | ModulePathDetails::BundledThirdParty(_) => {
                    to_real_path(module_path).unwrap_or_else(|| module_path.as_path().to_path_buf())
                }
                _ => module_path.as_path().to_path_buf(),
            };
            (qname.clone(), file_path)
        })
        .collect()
}
