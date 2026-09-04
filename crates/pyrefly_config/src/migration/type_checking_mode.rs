/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::base::Preset;
use crate::config::ConfigFile;
use crate::migration::config_option_migrater::ConfigOptionMigrater;
use crate::migration::pyright::PyrightConfig;
use crate::migration::pyright::{self};

/// Configuration option for the default errors that are shown.
/// Note: this must be run before [`crate::migration::error_codes::ErrorCodes`]
pub struct TypeCheckingMode;

impl ConfigOptionMigrater for TypeCheckingMode {
    fn migrate_from_mypy(&self, _: &Ini, _: &mut ConfigFile) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Mypy does not have direct equivalents for Pyright's type checking mode"
        ))
    }

    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()> {
        pyrefly_cfg.preset = match pyright_cfg.type_checking_mode {
            // TODO: "recommended" in basedpyright does enable all rules, but sets the severity to warning
            // and turns on failOnWarnings. we could do the same here, but for now we just treat both "All"
            // and "recommended" the same way.
            Some(pyright::TypeCheckingMode::All)
            | Some(pyright::TypeCheckingMode::Recommended)
            | None => {
                if pyright_cfg.is_basedpyright {
                    // basedpyright defaults to the `recommended` type checking mode rather than
                    // pyright's `standard`, so an unset `typeCheckingMode` (`None`) should also be
                    // treated the same as `all`.
                    Some(Preset::All)
                } else {
                    None
                }
            }
            Some(pyright::TypeCheckingMode::Off) => Some(Preset::Off),
            // we intentionally don't match other typeCheckingModes such as "strict", because pyright's
            // isn't necessarily the same as pyrefly's "Strict" preset
            Some(_) => None,
        };
        Ok(())
    }
}
