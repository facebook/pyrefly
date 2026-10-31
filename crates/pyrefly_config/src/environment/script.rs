/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PEP 723 scripts, which carry their own dependency metadata.

/// Returns the contents of the PEP 723 `script` metadata block in `source`, if it has one.
pub fn metadata(source: &str) -> Option<&str> {
    let mut start = None;
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let text = line.trim_end_matches(['\n', '\r']);
        match start {
            None if text == "# /// script" => start = Some(offset + line.len()),
            Some(start) if text == "# ///" => return Some(&source[start..offset]),
            Some(_) if text != "#" && !text.starts_with("# ") => start = None,
            _ => {}
        }
        offset += line.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata() {
        assert_eq!(
            metadata(
                "#!/usr/bin/env python\n# /// script\n# dependencies = [\"six\"]\n#\n# ///\nimport six\n"
            ),
            Some("# dependencies = [\"six\"]\n#\n"),
        );
        assert_eq!(
            metadata("# /// script\r\n# requires-python = \">=3.12\"\r\n# ///\r\n"),
            Some("# requires-python = \">=3.12\"\r\n"),
        );
        assert_eq!(metadata("# /// script\nimport six\n# ///\n"), None);
        assert_eq!(metadata("# /// pyproject\n# ///\n"), None);
        assert_eq!(metadata("x = 1\n"), None);
    }
}
