/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[expect(unused)]
fn query_source_db<'a>(
    files: impl Iterator<Item = &'a PathBuf>,
    cwd: &Path,
) -> anyhow::Result<Vec<u8>> {
    // TODO(connernilsen): handle querying targets too later on
    let mut cmd = Command::new("buck2");
    cmd.arg("bxl");
    cmd.arg("--reuse-current-config");
    cmd.arg("prelude//python/sourcedb/pyrefly.bxl:main");
    cmd.arg("--");
    cmd.args(files.flat_map(|f| [OsStr::new("--file"), f.as_os_str()].into_iter()));
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

    Ok(result.stdout)
}
