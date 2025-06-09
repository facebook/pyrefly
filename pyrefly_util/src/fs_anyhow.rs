/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::fs::ReadDir;
use std::io::Write;
use std::path::Path;

use anyhow::Context as _;

pub fn read_to_string(path: &Path) -> anyhow::Result<String> {
    fs::read_to_string(path).with_context(|| format!("When reading file `{}`", path.display()))
}

pub fn read(path: &Path) -> anyhow::Result<Vec<u8>> {
    fs::read(path).with_context(|| format!("When reading file `{}`", path.display()))
}

pub fn write(path: &Path, contents: &[u8]) -> Result<(), anyhow::Error> {
    fs::write(path, contents).with_context(|| format!("When writing file `{}`", path.display()))
}

pub fn append(path: &Path, contents: &[u8]) -> Result<(), anyhow::Error> {
    let context = || format!("When appending to file `{}`", path.display());
    let mut out = fs::File::options()
        .create(true)
        .append(true)
        .open(path)
        .with_context(context)?;
    out.write_all(contents).with_context(context)
}

pub fn read_dir(path: &Path) -> anyhow::Result<ReadDir> {
    fs::read_dir(path).with_context(|| format!("When reading directory `{}`", path.display()))
}

pub fn create_dir_all(path: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("When creating directory `{}`", path.display()))
}
