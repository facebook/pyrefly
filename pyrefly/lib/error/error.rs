/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp;
use std::fmt::Debug;
use std::io;
use std::io::Write;
use std::path::Path;

use itertools::Itertools;
use lsp_types::CodeDescription;
use lsp_types::Diagnostic;
use lsp_types::Url;
use pyrefly_python::ignore::Tool;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::number_thousands;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_annotate_snippets::Level;
use ruff_annotate_snippets::Message;
use ruff_annotate_snippets::Renderer;
use ruff_annotate_snippets::Snippet;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use yansi::Paint;

use crate::config::error_kind::ErrorKind;
use crate::config::error_kind::Severity;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Error {
    module: Module,
    range: TextRange,
    display_range: DisplayRange,
    error_kind: ErrorKind,
    severity: Severity,
    /// First line of the error message
    msg_header: Box<str>,
    /// The rest of the error message after the first line.
    /// Note that this is formatted for pretty-printing, with two spaces at the beginning and after every newline.
    msg_details: Option<Box<str>>,
}

impl Ranged for Error {
    fn range(&self) -> TextRange {
        self.range
    }
}

impl Error {
    pub fn write_line(
        &self,
        mut f: impl Write,
        project_root: &Path,
        verbose: bool,
    ) -> io::Result<()> {
        if verbose && self.severity.is_enabled() {
            writeln!(
                f,
                "{} {} [{}]",
                self.severity.label(),
                self.msg_header,
                self.error_kind.to_name(),
            )?;
            let origin = self.path_string_with_fragment(project_root);
            let snippet = self.get_source_snippet(&origin);
            let renderer = Renderer::plain();
            writeln!(f, "{}", renderer.render(snippet))?;
            if let Some(details) = &self.msg_details {
                writeln!(f, "{details}")?;
            }
        } else if self.severity.is_enabled() {
            writeln!(
                f,
                "{} {}:{}: {} [{}]",
                self.severity.label(),
                self.path_string_with_fragment(project_root),
                self.display_range,
                self.msg_header,
                self.error_kind.to_name(),
            )?;
        }
        Ok(())
    }

    pub fn print_colors(&self, project_root: &Path, verbose: bool) {
        if verbose && self.severity.is_enabled() {
            anstream::println!(
                "{} {} {}",
                self.severity.painted(),
                Paint::new(&*self.msg_header),
                Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            );
            let origin = self.path_string_with_fragment(project_root);
            let snippet = self.get_source_snippet(&origin);
            let renderer = Renderer::styled();
            anstream::println!("{}", renderer.render(snippet));
            if let Some(details) = &self.msg_details {
                anstream::println!("{details}");
            }
        } else if self.severity.is_enabled() {
            anstream::println!(
                "{} {}:{}: {} {}",
                self.severity.painted(),
                Paint::blue(&self.path_string_with_fragment(project_root)),
                Paint::dim(self.display_range()),
                Paint::new(&*self.msg_header),
                Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            );
        }
    }

    /// Return the path with a cell fragment if the error is in a notebook cell.
    fn path_string_with_fragment(&self, project_root: &Path) -> String {
        let path = self.path().as_path();
        let path = path.strip_prefix(project_root).unwrap_or(path);
        if let Some(cell) = self.display_range.start.cell() {
            format!("{}#{cell}", path.to_string_lossy())
        } else {
            path.to_string_lossy().to_string()
        }
    }

    fn get_source_snippet<'a>(&'a self, origin: &'a str) -> Message<'a> {
        // Maximum number of lines to print in the snippet.
        const MAX_LINES: u32 = 5;

        // Warning: The SourceRange is char indexed, while the snippet is byte indexed.
        //          Be careful in the conversion.
        let source = self.module.lined_buffer().content_in_line_range(
            self.display_range.start.line_within_file(),
            cmp::min(
                LineNumber::from_zero_indexed(
                    self.display_range
                        .start
                        .line_within_file()
                        .to_zero_indexed()
                        + MAX_LINES,
                ),
                self.display_range.end.line_within_file(),
            ),
        );
        let line_start = self
            .module
            .lined_buffer()
            .line_start(self.display_range.start.line_within_file());

        let level = match self.severity {
            Severity::Error => Level::Error,
            Severity::Warn => Level::Warning,
            Severity::Info => Level::Info,
            Severity::Ignore => Level::None,
        };
        let span_start = (self.range.start() - line_start).to_usize();
        let span_end = cmp::min(span_start + self.range.len().to_usize(), source.len());
        Level::None.title("").snippet(
            Snippet::source(source)
                .line_start(self.display_range.start.line_within_cell().get() as usize)
                .origin(origin)
                .annotation(level.span(span_start..span_end)),
        )
    }

    pub fn with_severity(&self, severity: Severity) -> Self {
        let mut res = self.clone();
        res.severity = severity;
        res
    }

    pub fn severity(&self) -> Severity {
        self.severity
    }

    /// Create a diagnostic suitable for use in LSP.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let code = self.error_kind().to_name().to_owned();
        let code_description = Url::parse(&self.error_kind().docs_url())
            .ok()
            .map(|href| CodeDescription { href });
        Diagnostic {
            range: self.module.to_lsp_range(self.range()),
            severity: Some(match self.severity() {
                Severity::Error => lsp_types::DiagnosticSeverity::ERROR,
                Severity::Warn => lsp_types::DiagnosticSeverity::WARNING,
                Severity::Info => lsp_types::DiagnosticSeverity::INFORMATION,
                // Ignored errors shouldn't be here
                Severity::Ignore => lsp_types::DiagnosticSeverity::INFORMATION,
            }),
            source: Some("Pyrefly".to_owned()),
            message: self.msg().to_owned(),
            code: Some(lsp_types::NumberOrString::String(code)),
            code_description,
            tags: None,
            ..Default::default()
        }
    }

    pub fn get_notebook_cell(&self) -> Option<usize> {
        self.module.to_cell_for_lsp(self.range().start())
    }

    pub fn module(&self) -> &Module {
        &self.module
    }
}

#[cfg(test)]
pub fn print_errors(project_root: &Path, errors: &[Error]) {
    for err in errors {
        err.print_colors(project_root, true);
    }
}

fn count_error_kinds(errors: &[Error]) -> Vec<(ErrorKind, usize)> {
    let mut map = SmallMap::new();
    for err in errors {
        let kind = err.error_kind();
        *map.entry(kind).or_default() += 1;
    }
    let mut res = map.into_iter().collect::<Vec<_>>();
    res.sort_by_key(|x| x.1);
    res
}

pub fn print_error_counts(errors: &[Error], limit: usize) {
    let items = count_error_kinds(errors);
    let limit = if limit > 0 { limit } else { items.len() };
    for (error, count) in items.iter().rev().take(limit) {
        eprintln!(
            "{} instances of {}",
            number_thousands(*count),
            error.to_name()
        );
    }
}

impl Error {
    pub fn new(module: Module, range: TextRange, msg: Vec1<String>, error_kind: ErrorKind) -> Self {
        let display_range = module.display_range(range);
        let msg_has_details = msg.len() > 1;
        let mut msg = msg.into_iter();
        let msg_header = msg.next().unwrap().into_boxed_str();
        let msg_details = if msg_has_details {
            Some(msg.map(|s| format!("  {s}")).join("\n").into_boxed_str())
        } else {
            None
        };
        Self {
            module,
            range,
            display_range,
            error_kind,
            severity: error_kind.default_severity(),
            msg_header,
            msg_details,
        }
    }

    pub fn display_range(&self) -> &DisplayRange {
        &self.display_range
    }

    pub fn lined_buffer(&self) -> &LinedBuffer {
        self.module.lined_buffer()
    }

    pub fn path(&self) -> &ModulePath {
        self.module.path()
    }

    pub fn msg_header(&self) -> &str {
        &self.msg_header
    }

    pub fn msg_details(&self) -> Option<&str> {
        self.msg_details.as_deref()
    }

    pub fn msg(&self) -> String {
        if let Some(details) = &self.msg_details {
            format!("{}\n{}", self.msg_header, details)
        } else {
            (*self.msg_header).to_owned()
        }
    }

    pub fn is_ignored(&self, enabled_ignores: &SmallSet<Tool>) -> bool {
        self.module.is_ignored(
            &self.display_range,
            self.error_kind.to_name(),
            enabled_ignores,
        )
    }

    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module_name::ModuleName;
    use ruff_text_size::TextSize;
    use vec1::vec1;

    use super::*;

    #[test]
    fn test_error_render() {
        let module_info = Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new("def f(x: int) -> str:\n    return x".to_owned()),
        );
        let error = Error::new(
            module_info,
            TextRange::new(TextSize::new(26), TextSize::new(34)),
            vec1!["bad return".to_owned()],
            ErrorKind::BadReturn,
        );
        let root = PathBuf::new();
        let mut normal = Vec::new();
        error
            .write_line(&mut Cursor::new(&mut normal), root.as_path(), false)
            .unwrap();
        let mut verbose = Vec::new();
        error
            .write_line(&mut Cursor::new(&mut verbose), root.as_path(), true)
            .unwrap();

        assert_eq!(
            str::from_utf8(&normal).unwrap(),
            "ERROR test.py:2:5-13: bad return [bad-return]\n"
        );
        assert_eq!(
            str::from_utf8(&verbose).unwrap(),
            r#"ERROR bad return [bad-return]
 --> test.py:2:5
  |
2 |     return x
  |     ^^^^^^^^
  |
"#,
        );
    }

    #[test]
    fn test_error_too_long() {
        let contents = format!("Start\n{}\nEnd", "X\n".repeat(1000));
        let module_info = Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new(contents.clone()),
        );
        let error = Error::new(
            module_info,
            TextRange::new(TextSize::new(0), TextSize::new(contents.len() as u32)),
            vec1!["oops".to_owned()],
            ErrorKind::BadReturn,
        );
        let mut output = Vec::new();
        let root = PathBuf::new();
        error
            .write_line(&mut Cursor::new(&mut output), root.as_path(), true)
            .unwrap();

        assert_eq!(
            str::from_utf8(&output).unwrap(),
            r#"ERROR oops [bad-return]
 --> test.py:1:1
  |
1 | / Start
2 | | X
3 | | X
4 | | X
5 | | X
6 | | X
  | |__^
  |
"#,
        );
    }
}
