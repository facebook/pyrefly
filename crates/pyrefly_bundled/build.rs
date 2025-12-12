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

fn get_typeshed_input_path() -> PathBuf {
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

fn get_stubs_input_path() -> PathBuf {
    match env::var_os("STUBS_ROOT") {
        Some(root) => {
            // When building with Buck, stubs filegroup dir is passed in using STUBS_ROOT
            Path::new(&root).join("stubs")
        }
        None => {
            // When building with Cargo, we could locate stubs directly using relative dir
            PathBuf::from("third_party/stubs")
        }
    }
}

fn get_output_path() -> Result<PathBuf, std::env::VarError> {
    // When building with Buck, output artifact directory is specified using this env var
    match env::var_os("OUT") {
        Some(path) => Ok(Path::new(&path).to_path_buf()),
        None => {
            // When building with Cargo, this env var is the containing directory of the artifact
            let out_dir = env::var("OUT_DIR")?;
            Ok(Path::new(&out_dir).to_path_buf())
        }
    }
}

/// Creates a compressed tar archive from the given input path and writes it to the output path.
/// Also computes and writes a SHA256 digest of the archive.
fn create_archive(
    input_path: &Path,
    output_path: &Path,
    digest_name: &str,
) -> Result<(), std::io::Error> {
    let digest_path = output_path.with_file_name(digest_name);

    // Create the tar.zst archive
    let mut archive_bytes = Vec::new();
    let encoder = zstd::stream::write::Encoder::new(&mut archive_bytes, 0)?;
    let mut tar = tar::Builder::new(encoder);

    if !input_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Input path does not exist: {}", input_path.display()),
        ));
    }
    tar.append_dir_all("", input_path)?;

    let encoder = tar.into_inner()?;
    encoder.finish()?;

    let hash = sha2::Sha256::digest(&archive_bytes);

    std::fs::write(output_path, &archive_bytes)?;
    std::fs::write(&digest_path, hash)?;

    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    // Only watch for metadata changes to avoid having Cargo repeatedly crawling for
    // changes in the entire typeshed dir.
    println!("cargo::rerun-if-changed=third_party/typeshed_metadata.json");

    let output_dir = get_output_path().unwrap();

    // Create typeshed archive
    let typeshed_input = get_typeshed_input_path();
    let typeshed_output = output_dir.join("typeshed.tar.zst");
    create_archive(&typeshed_input, &typeshed_output, "typeshed.sha256")?;

    // Create third-party stubs archive (non-typeshed stubs)
    let stubs_input = get_stubs_input_path();
    let stubs_output = output_dir.join("stubs.tar.zst");
    create_archive(&stubs_input, &stubs_output, "stubs.sha256")?;

    Ok(())
}
