/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::error::error::Error;
use crate::module::module_info::ModuleInfo;

#[derive(Clone, Debug)]
pub struct Expectation {
    module: ModuleInfo,
    /// Expected errors: the error message on this line must contain this substring.
    error: Vec<(usize, String)>,
    /// Negative assertions: the error message on this line must NOT contain this substring.
    not_error: Vec<(usize, String)>,
}

impl Expectation {
    fn parse_line(&mut self, line_no: usize, mut s: &str) {
        // Parse negative assertions (# !E:) first, since they appear after positive assertions
        // on the same line: `# E: error msg # !E: should not contain`
        while let Some((prefix, err)) = s.trim().rsplit_once("# !E:") {
            self.not_error.push((line_no, err.trim().to_owned()));
            s = prefix.trim_end();
        }
        while let Some((prefix, err)) = s.trim().rsplit_once("# E:") {
            self.error.push((line_no, err.trim().to_owned()));
            s = prefix.trim_end();
        }
    }

    pub fn parse(module: ModuleInfo, s: &str) -> Self {
        let mut res = Self {
            module,
            error: Vec::new(),
            not_error: Vec::new(),
        };
        for (line_no, line) in s.lines().enumerate() {
            res.parse_line(line_no + 1, line)
        }
        res
    }

    pub fn check(&self, errors: &[Error]) -> anyhow::Result<()> {
        if self.error.len() != errors.len() {
            Err(anyhow::anyhow!(
                "Expectations failed for {}: expected {} errors, but got {}",
                self.module.path(),
                self.error.len(),
                errors.len(),
            ))
        } else {
            for (line_no, msg) in &self.error {
                if !errors.iter().any(|e| {
                    e.msg().replace("\n", "\\n").contains(msg)
                        && e.display_range().start.line_within_file().get() as usize == *line_no
                }) {
                    return Err(anyhow::anyhow!(
                        "Expectations failed for {}: can't find error (line {line_no}): {msg}",
                        self.module.path()
                    ));
                }
            }
            // Check negative assertions: error messages must NOT contain these substrings
            for (line_no, msg) in &self.not_error {
                if errors.iter().any(|e| {
                    e.msg().replace("\n", "\\n").contains(msg)
                        && e.display_range().start.line_within_file().get() as usize == *line_no
                }) {
                    return Err(anyhow::anyhow!(
                        "Expectations failed for {}: error unexpectedly contains (line {line_no}): {msg}",
                        self.module.path()
                    ));
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use vec1::vec1;

    use super::*;
    use crate::config::error_kind::ErrorKind;

    fn make_module(contents: &str) -> Module {
        Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new(contents.to_owned()),
        )
    }

    fn make_error(module: Module, line: u32, msg: &str) -> Error {
        // Calculate the byte offset for the start of the given line
        let contents = module.contents();
        let mut offset = 0u32;
        for (i, line_content) in contents.lines().enumerate() {
            if i + 1 == line as usize {
                break;
            }
            offset += line_content.len() as u32 + 1; // +1 for newline
        }
        Error::new(
            module,
            TextRange::new(TextSize::new(offset), TextSize::new(offset + 1)),
            vec1![msg.to_owned()],
            ErrorKind::BadReturn,
        )
    }

    #[test]
    fn test_parse_negative_assertion() {
        let module = make_module("line1\nx = 1  # !E: should not appear\n");
        let exp = Expectation::parse(module, "line1\nx = 1  # !E: should not appear\n");
        assert_eq!(exp.error.len(), 0);
        assert_eq!(exp.not_error.len(), 1);
        assert_eq!(exp.not_error[0], (2, "should not appear".to_owned()));
    }

    #[test]
    fn test_parse_both_positive_and_negative() {
        let module = make_module("x = 1  # E: some error # !E: unwanted\n");
        let exp = Expectation::parse(module, "x = 1  # E: some error # !E: unwanted\n");
        assert_eq!(exp.error.len(), 1);
        assert_eq!(exp.error[0], (1, "some error".to_owned()));
        assert_eq!(exp.not_error.len(), 1);
        assert_eq!(exp.not_error[0], (1, "unwanted".to_owned()));
    }

    #[test]
    fn test_check_negative_assertion_passes() {
        let contents = "x = 1  # E: actual error # !E: unwanted\n";
        let module = make_module(contents);
        let exp = Expectation::parse(module.clone(), contents);
        // Error contains "actual error" but not "unwanted"
        let errors = vec![make_error(module, 1, "actual error here")];
        assert!(exp.check(&errors).is_ok());
    }

    #[test]
    fn test_check_negative_assertion_fails() {
        let contents = "x = 1  # E: error # !E: unwanted\n";
        let module = make_module(contents);
        let exp = Expectation::parse(module.clone(), contents);
        // Error contains both the expected text AND the unwanted text
        let errors = vec![make_error(module, 1, "error with unwanted text")];
        assert!(exp.check(&errors).is_err());
    }
}
