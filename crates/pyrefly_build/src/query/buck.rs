/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context as _;
use serde::Deserialize;
use serde::Serialize;

use crate::query::SourceDbQuerier;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "kebab-case")]
pub struct BxlArgs {
    isolation_dir: Option<String>,
    extras: Option<Vec<String>>,
}

impl BxlArgs {
    pub fn new(isolation_dir: Option<String>, extras: Option<Vec<String>>) -> Self {
        Self {
            isolation_dir,
            extras,
        }
    }

    pub fn get_repo_root(&self, cwd: &Path) -> anyhow::Result<PathBuf> {
        let mut cmd = Command::new("buck2");
        cmd.arg("root");
        cmd.arg("--kind");
        cmd.arg("project");
        cmd.current_dir(cwd);
        let output = cmd
            .output()
            .context("Querying for build system repo root")?;

        let stdout = String::from_utf8(output.stdout).with_context(|| {
            let stderr =
                String::from_utf8(output.stderr).unwrap_or("<Could not decode STDERR>".to_owned());
            format!(
                "Failed to parse stdout while querying build system repo root, STDERR: {stderr}"
            )
        })?;

        Ok(PathBuf::from(stdout.trim()))
    }
}

#[derive(Debug)]
pub struct BxlQuerier(BxlArgs);

impl BxlQuerier {
    pub fn new(args: BxlArgs) -> Self {
        Self(args)
    }
}

impl SourceDbQuerier for BxlQuerier {
    fn construct_command(&self) -> Command {
        let mut cmd = Command::new("buck2");
        if let Some(isolation_dir) = &self.0.isolation_dir {
            cmd.arg("--isolation-dir");
            cmd.arg(isolation_dir);
        }
        cmd.arg("bxl");
        cmd.arg("--reuse-current-config");
        if let Some(metadata) = &self.0.extras {
            cmd.args(metadata);
        }
        cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
        cmd
    }
}
