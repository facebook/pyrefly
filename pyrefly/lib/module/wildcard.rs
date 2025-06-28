/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;

use pyrefly_python::module_name::ModuleName;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use serde::de::Visitor;

/// A pattern that allows for matching on module paths
/// with a glob-like syntax. Only full matches are supported,
/// along with a wildcard (`*`) that allows for matching one or more
/// path components. If the wildcard does not start the pattern, it can
/// also match zero components.
///
/// The `*` must be its own path component (`<pattern>.*.<pattern>`), or start
/// (`*.<pattern>`)/end (`<pattern>.*`) the pattern.
#[derive(Debug, Clone)]
pub struct ModuleWildcard {
    pattern: Regex,
    /// The original pattern, before rewriting to escape it and handle wildcards
    origin: String,
}

impl ModuleWildcard {
    pub fn new(value: &str) -> anyhow::Result<Self> {
        Ok(Self {
            pattern: rewrite_pattern_as_regex(value)?,
            origin: value.to_owned(),
        })
    }

    pub fn matches(&self, module_path: ModuleName) -> bool {
        self.pattern.is_match(module_path.as_str())
    }

    pub fn as_str(&self) -> &str {
        self.pattern.as_str()
    }
}

impl PartialEq<ModuleWildcard> for ModuleWildcard {
    fn eq(&self, other: &Self) -> bool {
        self.origin == other.origin
    }
}

impl Eq for ModuleWildcard {}

impl Hash for ModuleWildcard {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.origin.hash(state);
    }
}

impl<'de> Deserialize<'de> for ModuleWildcard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StrVisitor;
        impl<'de> Visitor<'de> for StrVisitor {
            type Value = ModuleWildcard;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("A string regex pattern")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                ModuleWildcard::new(value).map_err(|err| {
                    serde::de::Error::custom(format!("Parsed regex is not valid: {err}"))
                })
            }
        }
        deserializer.deserialize_str(StrVisitor)
    }
}

impl Serialize for ModuleWildcard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.origin)
    }
}

/// Rewrites a module glob pattern into a regex. Uses the same logic
/// as mypy to reduce issues when converting module path globs.
fn rewrite_pattern_as_regex(original: &str) -> anyhow::Result<Regex> {
    let mut pattern = "^".to_owned();
    let mut split = original.split('.');

    match split.next() {
        Some("*") => pattern.push_str(".*"),
        Some(comp) => pattern.push_str(&regex::escape(comp)),
        None => return Err(anyhow::anyhow!("Cannot process an empty module glob")),
    };
    for comp in split {
        if comp == "*" {
            pattern.push_str(r"(\..*)?");
        } else {
            pattern.push_str(r"\.");
            pattern.push_str(&regex::escape(comp));
        }
    }

    pattern.push('$');
    Regex::new(&pattern).map_err(|err| {
        anyhow::anyhow!(
            "Internal Error: invalid module glob pattern constructed: {} from {}, with error {}",
            pattern,
            original,
            err,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_pattern_as_regex() {
        let pattern = rewrite_pattern_as_regex("path.to.my.file").unwrap();
        assert_eq!(pattern.as_str(), r"^path\.to\.my\.file$");
        assert!(pattern.is_match("path.to.my.file"));
        assert!(!pattern.is_match(".path.to.my.file"));
        assert!(!pattern.is_match("path.to.my.file."));
        assert!(!pattern.is_match(".path.to.a.file"));

        let pattern = rewrite_pattern_as_regex("path.to.my.file.*").unwrap();
        assert_eq!(pattern.as_str(), r"^path\.to\.my\.file(\..*)?$");
        assert!(pattern.is_match("path.to.my.file"));
        assert!(pattern.is_match("path.to.my.file.s.submodule"));
        assert!(pattern.is_match("path.to.my.file."));
        assert!(!pattern.is_match("path.to.my.files"));

        let pattern = rewrite_pattern_as_regex("path.to.*.file").unwrap();
        assert_eq!(pattern.as_str(), r"^path\.to(\..*)?\.file$");
        assert!(pattern.is_match("path.to.file"));
        assert!(pattern.is_match("path.to.my.file"));
        assert!(pattern.is_match("path.to.a.something.somewhere.file"));
        assert!(!pattern.is_match("this.ends.with.file"));
        assert!(!pattern.is_match("path.to"));
        assert!(!pattern.is_match("path.to.files"));
        assert!(!pattern.is_match("path.to.myfile"));
        assert!(!pattern.is_match("path.to..myfile"));

        let pattern = rewrite_pattern_as_regex("*.path.to.my.file").unwrap();
        assert_eq!(pattern.as_str(), r"^.*\.path\.to\.my\.file$");
        assert!(pattern.is_match("this.is.a.path.to.my.file"));
        assert!(pattern.is_match("a.path.to.my.file"));
        assert!(pattern.is_match(".path.to.my.file"));
        assert!(!pattern.is_match("path.to.my.file"));
        assert!(!pattern.is_match("apath.to.my.file"));

        assert_eq!(
            rewrite_pattern_as_regex("path.to.any*.file")
                .unwrap()
                .to_string(),
            r"^path\.to\.any\*\.file$"
        );
        assert_eq!(
            rewrite_pattern_as_regex("path.to.*any.file")
                .unwrap()
                .to_string(),
            r"^path\.to\.\*any\.file$",
        );
        assert_eq!(
            rewrite_pattern_as_regex("path.to.*any*.file")
                .unwrap()
                .to_string(),
            r"^path\.to\.\*any\*\.file$"
        );
        assert_eq!(
            rewrite_pattern_as_regex("path.to.so*me.file")
                .unwrap()
                .to_string(),
            r"^path\.to\.so\*me\.file$"
        );
    }
}
