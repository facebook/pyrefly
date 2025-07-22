/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;

use anyhow::Context as _;
use clap::Parser;
use parse_display::Display;
use pyrefly_util::fs_anyhow;

use crate::config::ConfigFile;

/// Types of configuration files that can be detected or created.
#[derive(Clone, Debug, Parser, Copy, Display)]
pub enum ConfigFileKind {
    MyPy,
    Pyright,
    Pyrefly,
    Pyproject,
}

impl ConfigFileKind {
    pub fn file_name(&self) -> &str {
        match self {
            Self::MyPy => "mypy.ini",
            Self::Pyright => "pyrightconfig.json",
            Self::Pyrefly => "pyrefly.toml",
            Self::Pyproject => "pyproject.toml",
        }
    }

    pub fn toml_identifier(self) -> String {
        match self {
            // This makes me question if pyproject should be a part of the enum at all
            Self::Pyproject => "".to_owned(),
            _ => format!("[tool.{self}]").to_lowercase(),
        }
    }

    pub fn check_for_existing_config(&self, path: &Path) -> anyhow::Result<bool> {
        let file_name = self.file_name();
        if path.ends_with(file_name) && path.exists() {
            return Ok(true);
        }
        if path.ends_with(ConfigFile::PYPROJECT_FILE_NAME) && path.exists() {
            let raw_pyproject = fs_anyhow::read_to_string(path).with_context(|| {
                format!(
                    "While trying to check for an existing {self} config in `{}`",
                    path.display()
                )
            })?;
            return Ok(raw_pyproject.contains(&self.toml_identifier()));
        }
        if path.is_dir() {
            let custom_file = self.check_for_existing_config(&path.join(file_name));
            let pyproject =
                self.check_for_existing_config(&path.join(ConfigFile::PYPROJECT_FILE_NAME));
            return Ok(custom_file? || pyproject?);
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_file_kinds() -> anyhow::Result<()> {
        let kind = ConfigFileKind::MyPy;
        assert_eq!(kind.toml_identifier(), "[tool.mypy]".to_owned());
        Ok(())
    }
}
