/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use configparser::ini::Ini;

use crate::config::ConfigFile;
use crate::migration::pyright::PyrightConfig;

/// A trait for configuration options that can be migrated from mypy or pyright to pyrefly
pub trait ConfigOptionMigrater {
    /// Migrate a configuration option from mypy.ini to pyrefly config
    ///
    /// # Arguments
    ///
    /// * `mypy_cfg` - The mypy.ini configuration
    /// * `pyrefly_cfg` - The ConfigFile to update
    fn migrate_from_mypy(&self, mypy_cfg: &Ini, pyrefly_cfg: &mut ConfigFile)
    -> anyhow::Result<()>;

    /// Migrate a configuration option from pyright config to pyrefly config
    ///
    /// # Arguments
    ///
    /// * `pyright_cfg` - The pyright configuration
    /// * `pyrefly_cfg` - The ConfigFile to update
    fn migrate_from_pyright(
        &self,
        pyright_cfg: &PyrightConfig,
        pyrefly_cfg: &mut ConfigFile,
    ) -> anyhow::Result<()>;
}
