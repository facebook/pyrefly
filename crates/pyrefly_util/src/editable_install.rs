/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use lsp_types::Url;
use serde::Deserialize;
use starlark_map::small_map::SmallMap;

use crate::lock::Mutex;

/// PEP 610 direct_url.json structure for detecting editable installs.
#[derive(Deserialize)]
struct DirectUrl {
    url: String,
    #[serde(default)]
    dir_info: DirInfo,
}

#[derive(Deserialize, Default)]
struct DirInfo {
    #[serde(default)]
    editable: bool,
}

/// Cache for editable source paths, keyed by sorted site-packages paths.
/// This avoids re-scanning site-packages on every check.
static EDITABLE_PATHS_CACHE: LazyLock<Mutex<SmallMap<Vec<PathBuf>, Vec<PathBuf>>>> =
    LazyLock::new(|| Mutex::new(SmallMap::new()));

/// Returns true if the path is an editable-install metadata file.
pub fn is_editable_metadata_file(path: &Path) -> bool {
    match path.file_name().and_then(|name| name.to_str()) {
        Some("direct_url.json") => true,
        Some(name) => name.ends_with(".pth") || name.ends_with(".egg-link"),
        None => false,
    }
}

/// Clear the editable source paths cache.
pub fn clear_editable_source_paths_cache() {
    EDITABLE_PATHS_CACHE.lock().clear();
}

/// Get editable source paths for the given site-packages, using cache.
pub fn get_editable_source_paths(site_packages: &[PathBuf]) -> Vec<PathBuf> {
    let mut key: Vec<PathBuf> = site_packages.to_vec();
    key.sort();

    let mut cache = EDITABLE_PATHS_CACHE.lock();
    if let Some(paths) = cache.get(&key) {
        return paths.clone();
    }

    let paths = detect_editable_packages(site_packages);
    cache.insert(key, paths.clone());
    paths
}

/// Detect editable packages by scanning site-packages for direct_url.json (PEP 610), .pth, and .egg-link files.
fn detect_editable_packages(site_packages: &[PathBuf]) -> Vec<PathBuf> {
    let mut editable_paths = Vec::new();

    for sp in site_packages {
        let Ok(entries) = fs::read_dir(sp) else {
            continue;
        };

        let mut dist_info_dirs = Vec::new();
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();

            if path.is_dir() {
                if path.extension().is_some_and(|ext| ext == "dist-info") {
                    dist_info_dirs.push(path);
                }
                continue;
            }

            // Parse any .pth or .egg-link files in the site-packages root.
            if !path.is_file() && !path.is_symlink() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with(".pth") {
                editable_paths.extend(read_pth_search_paths(&path));
            } else if name.ends_with(".egg-link") {
                if let Some(path) = read_egg_link_path(&path) {
                    editable_paths.push(path);
                }
            }
        }

        for path in dist_info_dirs {
            let direct_url_path = path.join("direct_url.json");
            let Ok(content) = fs::read_to_string(&direct_url_path) else {
                continue;
            };
            let Ok(direct_url) = serde_json::from_str::<DirectUrl>(&content) else {
                continue;
            };

            if !direct_url.dir_info.editable {
                continue;
            }

            let Ok(url) = Url::parse(&direct_url.url) else {
                continue;
            };
            if url.scheme() != "file" {
                continue;
            }
            let Ok(source_path) = url.to_file_path() else {
                continue;
            };
            if source_path.is_dir() {
                editable_paths.push(source_path);
            }
        }
    }

    editable_paths.sort();
    editable_paths.dedup();
    editable_paths
}

fn read_pth_search_paths(pth_file: &Path) -> Vec<PathBuf> {
    let mut search_paths = Vec::new();
    let Ok(metadata) = fs::metadata(pth_file) else {
        return search_paths;
    };
    if metadata.len() > 64 * 1024 {
        return search_paths;
    }
    let Ok(data) = fs::read_to_string(pth_file) else {
        return search_paths;
    };
    let parent = pth_file.parent().unwrap_or_else(|| Path::new(""));
    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("import ") {
            continue;
        }
        let candidate = Path::new(trimmed);
        let path = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            parent.join(candidate)
        };
        if path.is_dir() {
            search_paths.push(path);
        }
    }
    search_paths
}

fn read_egg_link_path(egg_link: &Path) -> Option<PathBuf> {
    let Ok(data) = fs::read_to_string(egg_link) else {
        return None;
    };
    let first_line = data.lines().find(|line| !line.trim().is_empty())?;
    let trimmed = first_line.trim();
    let candidate = Path::new(trimmed);
    let parent = egg_link.parent().unwrap_or_else(|| Path::new(""));
    let path = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        parent.join(candidate)
    };
    if path.is_dir() { Some(path) } else { None }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use lsp_types::Url;

    use super::get_editable_source_paths;

    #[test]
    fn test_get_editable_source_paths_finds_editable_package() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let source_dir = temp_dir.path().join("mypackage_source");
        fs::create_dir(&source_dir).unwrap();

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let source_url = Url::from_file_path(&source_dir).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": true}}}}"#,
            source_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result = get_editable_source_paths(std::slice::from_ref(&site_packages));

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], source_dir);
    }

    #[test]
    fn test_get_editable_source_paths_ignores_non_editable_package() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("requests-2.28.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let source_dir = temp_dir.path().join("requests_source");
        fs::create_dir(&source_dir).unwrap();

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let source_url = Url::from_file_path(&source_dir).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": false}}}}"#,
            source_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result = get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }

    #[test]
    fn test_get_editable_source_paths_ignores_missing_direct_url_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("somepackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let result = get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }

    #[test]
    fn test_get_editable_source_paths_ignores_nonexistent_source_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let dist_info = site_packages.join("mypackage-1.0.0.dist-info");
        fs::create_dir(&dist_info).unwrap();

        let nonexistent_path = temp_dir.path().join("does_not_exist");

        // Use Url::from_file_path to construct a proper file URL that works on all platforms
        let nonexistent_url = Url::from_file_path(&nonexistent_path).unwrap();
        let direct_url_content = format!(
            r#"{{"url": "{}", "dir_info": {{"editable": true}}}}"#,
            nonexistent_url.as_str()
        );
        fs::write(dist_info.join("direct_url.json"), direct_url_content).unwrap();

        let result = get_editable_source_paths(&[site_packages]);

        assert!(result.is_empty());
    }

    #[test]
    fn test_get_editable_source_paths_reads_pth_paths() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let source_dir = temp_dir.path().join("editable_source");
        fs::create_dir(&source_dir).unwrap();

        let pth_content = format!("# comment\n{}\nimport site\n", source_dir.to_string_lossy());
        fs::write(site_packages.join("editable.pth"), pth_content).unwrap();

        let result = get_editable_source_paths(&[site_packages]);

        assert!(
            result.contains(&source_dir),
            "missing source_dir in result: {result:?}"
        );
    }

    #[test]
    fn test_get_editable_source_paths_reads_egg_link() {
        let temp_dir = tempfile::tempdir().unwrap();
        let site_packages = temp_dir.path().join("site-packages");
        fs::create_dir(&site_packages).unwrap();

        let source_dir = temp_dir.path().join("editable_source");
        fs::create_dir(&source_dir).unwrap();

        let egg_link_content = format!("{}\n", source_dir.to_string_lossy());
        fs::write(site_packages.join("editable.egg-link"), egg_link_content).unwrap();

        let result = get_editable_source_paths(&[site_packages]);

        assert!(
            result.contains(&source_dir),
            "missing source_dir in result: {result:?}"
        );
    }
}
