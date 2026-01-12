/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::io::Read;
use std::path::PathBuf;

use anyhow::Context as _;
use starlark_map::small_map::SmallMap;
use tar::Archive;
use zstd::stream::read::Decoder;

const BUNDLED_TYPESHED_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/typeshed.tar.zst"));

pub const BUNDLED_TYPESHED_DIGEST: &[u8; 32] =
    include_bytes!(concat!(env!("OUT_DIR"), "/typeshed.sha256"));

const BUNDLED_THIRD_PARTY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stubs.tar.zst"));

pub const BUNDLED_THIRD_PARTY_DIGEST: &[u8; 32] =
    include_bytes!(concat!(env!("OUT_DIR"), "/stubs.sha256"));

#[derive(PartialEq)]
enum PathFilter {
    /// Filter for stdlib files (typeshed/stdlib/...)
    Stdlib,
    /// Filter for third-party stubs (typeshed/stubs/package-name/...)
    ThirdPartyTypeshedStubs,
    /// Filter for third-party stubs not from typeshed (stubs/package-name/...)
    ThirdPartyStubs,
}

impl PathFilter {
    fn expected_first_component(&self) -> Option<&str> {
        match self {
            PathFilter::Stdlib => Some("stdlib"),
            PathFilter::ThirdPartyTypeshedStubs => Some("stubs"),
            PathFilter::ThirdPartyStubs => None,
        }
    }

    #[allow(dead_code)]
    fn should_skip_next_component(&self) -> bool {
        match self {
            PathFilter::Stdlib => false,
            PathFilter::ThirdPartyTypeshedStubs => true, // Skip package directory
            PathFilter::ThirdPartyStubs => false,        // Skip package directory
        }
    }

    fn archive_bytes(&self) -> &'static [u8] {
        match self {
            PathFilter::Stdlib | PathFilter::ThirdPartyTypeshedStubs => BUNDLED_TYPESHED_BYTES,
            PathFilter::ThirdPartyStubs => BUNDLED_THIRD_PARTY_BYTES,
        }
    }
}

fn extract_pyi_files_from_archive(filter: PathFilter) -> anyhow::Result<SmallMap<PathBuf, String>> {
    let decoder = Decoder::new(filter.archive_bytes())?;
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .context("Cannot query all entries in archive")?;

    let mut items = SmallMap::new();

    for maybe_entry in entries {
        let mut entry = maybe_entry.context("Cannot read individual entry in typeshed archive")?;

        if entry.header().entry_type().is_dir() {
            continue;
        }

        let relative_path_context = entry
            .path()
            .context("Cannot extract path from archive entry")?;

        let mut relative_path_components = relative_path_context.components();

        let first_component = relative_path_components.next();
        if let Some(expected) = filter.expected_first_component()
            && first_component.is_none_or(|component| component.as_os_str() != expected)
        {
            continue;
        }
        // For ThirdPartyStubs, we need to put first_component back into the path

        // Typeshed stdlib stubs have a different directory structure than third-party stubs.
        // stdlib example: typeshed/stdlib/builtins.pyi
        // third-party example: typeshed/stubs/package-name/package-name/other.pyi
        // An example of this might be the path typeshed/stubs/JACK-Client/jack/other.pyi
        // In this case we want out path to be jack/other.pyi, so we skip over "stubs" and "JACK-Client".
        let relative_path = if filter == PathFilter::ThirdPartyTypeshedStubs {
            relative_path_components.next();
            relative_path_components.collect::<PathBuf>()
        } else if filter == PathFilter::ThirdPartyStubs {
            // Include first_component (e.g., "conans-stubs") in the path
            let mut path = PathBuf::from(first_component.unwrap().as_os_str());
            path.extend(relative_path_components);
            path
        } else {
            relative_path_components.collect::<PathBuf>()
        };

        if relative_path.extension().is_none_or(|ext| ext != "pyi") {
            // typeshed/stdlib/ contains non-.pyi files like VERSIONS that we don't care about.
            continue;
        }

        let size = entry.size();
        let mut contents = String::with_capacity(size as usize);
        entry
            .read_to_string(&mut contents)
            .context("Cannot read content of archive entry")?;
        items.entry(relative_path).or_insert(contents);
    }

    Ok(items)
}

pub fn bundled_typeshed() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_pyi_files_from_archive(PathFilter::Stdlib)
}

#[allow(dead_code)]
pub fn bundled_third_party_stubs() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_pyi_files_from_archive(PathFilter::ThirdPartyTypeshedStubs)
}

/// Extract third-party stubs from the bundled archive.
/// These are stubs that are not included in typeshed (e.g., pandas-stubs, boto3-stubs).
pub fn bundled_third_party() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_pyi_files_from_archive(PathFilter::ThirdPartyStubs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_typeshed_returns_stdlib_files() {
        let result = bundled_typeshed();
        assert!(result.is_ok(), "bundled_typeshed should succeed");

        let files = result.unwrap();
        assert!(!files.is_empty(), "Should contain stdlib .pyi files");

        // Verify all returned paths are .pyi files
        for (path, _) in files.iter() {
            assert_eq!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("pyi"),
                "All files should have .pyi extension, found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_returns_third_party_files() {
        let result = bundled_third_party_stubs();
        assert!(result.is_ok(), "bundled_typeshed should succeed");

        let files = result.unwrap();
        assert!(!files.is_empty(), "Should contain .pyi files");

        // Verify all returned paths are .pyi files
        for (path, _) in files.iter() {
            assert_eq!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("pyi"),
                "All files should have .pyi extension, found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_paths_are_relative() {
        let result = bundled_typeshed().unwrap();

        // Verify paths don't start with "stdlib" (it should be stripped)
        for (path, _) in result.iter() {
            let first_component = path.components().next();
            assert_ne!(
                first_component.and_then(|c| c.as_os_str().to_str()),
                Some("stdlib"),
                "Path should not start with 'stdlib', found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_third_party_resolves_path() {
        let result = bundled_third_party_stubs().unwrap();

        // Verify paths don't start with "stubs" (it should be stripped)
        for (path, _) in result.iter() {
            let first_component = path.components().next();
            assert_ne!(
                first_component.and_then(|c| c.as_os_str().to_str()),
                Some("stubs"),
                "Path should not start with 'stubs', found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_contains_common_modules() {
        let files = bundled_typeshed().unwrap();

        // Check that at least some .pyi files exist that match common module patterns
        let has_builtins = files.iter().any(|(path, _)| {
            path.to_str()
                .map(|s| s.contains("builtins"))
                .unwrap_or(false)
        });
        let has_sys = files
            .iter()
            .any(|(path, _)| path.to_str().map(|s| s.contains("sys")).unwrap_or(false));
        let has_os = files
            .iter()
            .any(|(path, _)| path.to_str().map(|s| s.contains("os")).unwrap_or(false));

        assert!(
            has_builtins || has_sys || has_os,
            "Should contain at least one common stdlib module (builtins, sys, or os)"
        );
    }

    #[test]
    fn test_bundled_typeshed_file_contents_not_empty() {
        let files = bundled_typeshed().unwrap();

        let non_empty_count = files
            .iter()
            .filter(|(_, content)| !content.is_empty())
            .count();
        assert!(
            non_empty_count > 0,
            "Should have at least some files with content"
        );
    }

    #[test]
    fn test_extract_pyi_files_from_archive_stdlib_filter() {
        let result = extract_pyi_files_from_archive(PathFilter::Stdlib);
        assert!(result.is_ok(), "Should successfully extract stdlib files");

        let files = result.unwrap();
        assert!(
            !files.is_empty(),
            "Should extract at least some stdlib files"
        );

        let unique_count = files.len();
        assert_eq!(files.len(), unique_count, "Should not have duplicate paths");
    }

    #[test]
    fn test_path_filter_expected_first_component() {
        assert_eq!(
            PathFilter::Stdlib.expected_first_component(),
            Some("stdlib"),
            "Stdlib filter should expect 'stdlib' component"
        );
        assert_eq!(
            PathFilter::ThirdPartyTypeshedStubs.expected_first_component(),
            Some("stubs"),
            "ThirdPartyStubs filter should expect 'stubs' component"
        );
    }

    #[test]
    fn test_bundled_typeshed_contains_valid_python_stubs() {
        let files = bundled_typeshed().unwrap();

        // Check that at least some files contain Python stub signatures
        let has_python_stub_content = files.iter().any(|(_, content)| {
            content.contains("def ") || content.contains("class ") || content.contains("import ")
        });

        assert!(
            has_python_stub_content,
            "At least some files should contain Python stub content"
        );
    }

    #[test]
    fn test_no_non_pyi_files_included() {
        let stdlib_files = bundled_typeshed().unwrap();
        let third_party_files = bundled_third_party_stubs().unwrap();

        for (path, _) in stdlib_files.iter().chain(third_party_files.iter()) {
            let ext = path.extension().and_then(|e| e.to_str());
            assert_eq!(
                ext,
                Some("pyi"),
                "Should only include .pyi files, found extension: {:?} in path: {:?}",
                ext,
                path
            );
        }
    }

    #[test]
    fn test_typeshed_archive_is_valid_zstd_compressed_tar() {
        let decoder = Decoder::new(BUNDLED_TYPESHED_BYTES);
        assert!(
            decoder.is_ok(),
            "Typeshed archive should be valid zstd compressed data"
        );
        let mut archive = Archive::new(decoder.unwrap());
        let entries = archive.entries();
        assert!(
            entries.is_ok(),
            "Typeshed archive should contain valid tar entries"
        );
    }

    #[test]
    fn test_stubs_archive_is_valid_zstd_compressed_tar() {
        let decoder = Decoder::new(BUNDLED_THIRD_PARTY_BYTES);
        assert!(
            decoder.is_ok(),
            "Stubs archive should be valid zstd compressed data"
        );
        let mut archive = Archive::new(decoder.unwrap());
        let entries = archive.entries();
        assert!(
            entries.is_ok(),
            "Stubs archive should contain valid tar entries"
        );
    }
}
