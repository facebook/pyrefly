/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
#[cfg(fbcode_build)]
use std::sync::LazyLock;

use anyhow::Context as _;
use serde::Deserialize;
use serde::Serialize;

use crate::query::SourceDbQuerier;

/// Check if systemd-run can create user scopes for running buck in a separate cgroup.
/// This helps isolate buck's resource usage from pyrefly's for better observability.
/// We test with a trivial command to ensure scopes actually work (not just that the
/// binary exists). Result is cached after the first check.
fn use_systemd() -> bool {
    #[cfg(fbcode_build)]
    {
        static AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
            Command::new("systemd-run")
                .arg("--user")
                .arg("--scope")
                .arg("--")
                .arg("true")
                .output()
                .is_ok_and(|o| o.status.success())
        });
        *AVAILABLE
    }
    #[cfg(not(fbcode_build))]
    false
}

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

impl Display for BxlArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "isolation_dir: {:?}, extras: {:?}",
            self.isolation_dir, self.extras
        )
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
    fn construct_command(&self, build_id_path: Option<&Path>) -> Command {
        // In fbcode builds, run buck in a separate cgroup via systemd-run for
        // resource isolation. This helps understand buck's resource usage separately
        // from pyrefly's.
        let mut cmd = if use_systemd() {
            let mut cmd = Command::new("systemd-run");
            cmd.arg("--user") // Run as current user, not system
                .arg("--scope") // Run in a transient scope unit
                .arg(format!(
                    "--slice=buck2_{}",
                    self.0.isolation_dir.as_deref().unwrap_or("default")
                ))
                .arg("--") // End of systemd-run args
                .arg("buck2");
            cmd
        } else {
            Command::new("buck2")
        };

        if let Some(isolation_dir) = &self.0.isolation_dir {
            cmd.arg("--isolation-dir");
            cmd.arg(isolation_dir);
        }
        cmd.arg("bxl");
        if let Some(build_id_path) = build_id_path {
            cmd.arg("--write-build-id");
            cmd.arg(build_id_path);
        }
        cmd.arg("--reuse-current-config");
        if let Some(metadata) = &self.0.extras {
            cmd.args(metadata);
        }
        cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
        cmd
    }
}
