/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;

/// If the module is on the search path, return its name from that path. Otherwise, return None.
pub fn module_from_path<'a>(
    path: &Path,
    includes: impl Iterator<Item = &'a PathBuf>,
) -> Option<ModuleName> {
    // Return a module name, and a boolean as to whether it is any good.
    fn path_to_module(mut path: &Path) -> Option<ModuleName> {
        if path.file_stem() == Some(dunder::INIT.as_str().as_ref()) {
            path = path.parent()?;
        }
        let mut out = Vec::new();
        let path = path.with_extension("");
        for x in path.components() {
            if let Component::Normal(x) = x
                && !x.is_empty()
            {
                out.push(x.to_string_lossy());
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(ModuleName::from_parts(out))
        }
    }

    for include in includes {
        if let Ok(x) = path.strip_prefix(include)
            && let Some(res) = path_to_module(x)
        {
            return Some(res);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_from_path() {
        let includes = [PathBuf::from("/foo/bar")];
        assert_eq!(
            module_from_path(Path::new("/foo/bar/baz.py"), includes.iter()),
            Some(ModuleName::from_str("baz"))
        );
        assert_eq!(
            module_from_path(Path::new("/foo/bar/baz/qux.pyi"), includes.iter()),
            Some(ModuleName::from_str("baz.qux"))
        );
        assert_eq!(
            module_from_path(Path::new("/foo/bar/baz/test/magic.py"), includes.iter()),
            Some(ModuleName::from_str("baz.test.magic"))
        );
        assert_eq!(
            module_from_path(Path::new("/foo/bar/baz/__init__.pyi"), includes.iter()),
            Some(ModuleName::from_str("baz"))
        );
        assert_eq!(
            module_from_path(Path::new("/test.py"), includes.iter()),
            None
        );
        assert_eq!(
            module_from_path(Path::new("/not_foo/test.py"), includes.iter()),
            None
        );
    }
}
