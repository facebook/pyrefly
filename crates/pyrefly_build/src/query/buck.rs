/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsString;
use std::fmt::Debug;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use anyhow::Context as _;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use tempfile::NamedTempFile;

use crate::query::Include;
use crate::query::TargetManifestDatabase;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct BxlArgs {
    isolation_dir: Option<String>,
    extras: Option<Vec<String>>,
}

pub(crate) fn query_source_db<'a>(
    files: impl Iterator<Item = &'a Include>,
    cwd: &Path,
    bxl_args: &BxlArgs,
) -> anyhow::Result<TargetManifestDatabase> {
    let mut files = files.peekable();
    if files.peek().is_none() {
        return Ok(TargetManifestDatabase {
            db: SmallMap::new(),
            root: cwd.to_path_buf(),
        });
    }
    let mut argfile = NamedTempFile::with_prefix("pyrefly_buck_query_")
        .with_context(|| "Failed to create temporary argfile for querying Buck".to_owned())?;
    let mut argfile_args = OsString::from("--");
    files.flat_map(Include::to_cli_arg).for_each(|arg| {
        argfile_args.push("\n");
        argfile_args.push(arg);
    });

    argfile
        .as_file_mut()
        .write_all(argfile_args.as_encoded_bytes())
        .with_context(|| "Could not write to argfile when querying Buck".to_owned())?;

    let mut cmd = Command::new("buck2");
    if let Some(isolation_dir) = &bxl_args.isolation_dir {
        cmd.arg("--isolation-dir");
        cmd.arg(isolation_dir);
    }
    cmd.arg("bxl");
    cmd.arg("--reuse-current-config");
    if let Some(metadata) = &bxl_args.extras {
        cmd.args(metadata);
    }
    cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
    cmd.arg(format!("@{}", argfile.path().display()));
    cmd.current_dir(cwd);

    let result = cmd.output()?;
    if !result.status.success() {
        let stdout = String::from_utf8(result.stdout)
            .unwrap_or_else(|_| "<Failed to parse stdout from Buck source db query>".to_owned());
        let stderr = String::from_utf8(result.stderr)
            .unwrap_or_else(|_| "<Failed to parse stderr from Buck source db query>".to_owned());

        return Err(anyhow::anyhow!(
            "Buck source db query failed...\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        ));
    }

    serde_json::from_slice(&result.stdout).with_context(|| {
        "Failed to construct valid `TargetManifestDatabase` from BXL query result".to_owned()
    })
}
