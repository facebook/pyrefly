/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::Path;
use std::path::PathBuf;

use sha2::Digest;

fn get_input_path() -> PathBuf {
    match env::var_os("TYPESHED_ROOT") {
        Some(root) => {
            // When building with Buck, typeshed filegroup dir is passed in using TYPESHED_ROOT
            Path::new(&root).join("typeshed")
        }
        None => {
            // When building with Cargo, we could locate typeshed directly using relative dir
            PathBuf::from("third_party/typeshed")
        }
    }
}

fn get_output_path() -> Result<PathBuf, std::env::VarError> {
    // When building with Buck, output artifact directory is specified using this env var
    match env::var_os("OUT") {
        Some(path) => Ok(Path::new(&path).join("typeshed.tar.zst")),
        None => {
            // When building with Cargo, this env var is the containing directory of the artifact
            let out_dir = env::var("OUT_DIR")?;
            Ok(Path::new(&out_dir).join("typeshed.tar.zst"))
        }
    }
}

fn main() -> Result<(), std::io::Error> {
    // Only watch for metadata changes to avoid having Cargo repeatedly crawling for
    // changes in the entire typeshed dir.
    println!("cargo::rerun-if-changed=third_party/typeshed_metadata.json");

    let input_path = get_input_path();
    let output_path = get_output_path().unwrap();
    let digest_path = output_path.with_file_name("typeshed.sha256");

    // Create the tar.zst archive
    let mut archive_bytes = Vec::new();
    let encoder = zstd::stream::write::Encoder::new(&mut archive_bytes, 0)?;
    let mut tar = tar::Builder::new(encoder);
    tar.append_dir_all("", input_path)?;
    let encoder = tar.into_inner()?;
    encoder.finish()?;

    // Compute SHA256 hash of the archive
    let hash = sha2::Sha256::digest(&archive_bytes);

    // Write digest to file
    std::fs::write(&output_path, &archive_bytes)?;
    std::fs::write(&digest_path, hash)?;

    Ok(())
}
