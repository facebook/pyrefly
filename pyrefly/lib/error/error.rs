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
use lsp_types::DiagnosticTag;
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

/// A secondary annotation that labels a span in the same file as the primary error.
/// Used to show additional context, e.g. the types of both operands in a binary operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SecondaryAnnotation {
    pub range: TextRange,
    pub label: Box<str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorQuickFix {
    ReplaceWithEnumMember { replacement: String },
}

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
    /// Additional labeled spans in the same file for richer diagnostics.
    secondary_annotations: Vec<SecondaryAnnotation>,
    /// Structured fixes that can be exposed by editor integrations.
    quick_fixes: Vec<ErrorQuickFix>,
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
        // Maximum number of lines to show in a single snippet. Annotations further apart
        // than this are shown as separate snippets rather than dumping all lines in between.
        // The primary span is also capped to this many lines for very large multi-line spans.
        const MAX_LINES: u32 = 10;

        // Partition secondary annotations into nearby (shown inline with the primary span)
        // and distant (shown as separate snippets to avoid printing excessive context).
        let primary_start_line = self.display_range.start.line_within_file();
        let primary_end_line = self.display_range.end.line_within_file();
        let mut start_line = primary_start_line;
        // Cap the primary span to MAX_LINES to avoid dumping huge multi-line spans.
        let mut end_line = cmp::min(
            LineNumber::from_zero_indexed(primary_start_line.to_zero_indexed() + MAX_LINES),
            primary_end_line,
        );
        let mut nearby_annotations = Vec::new();
        let mut distant_annotations = Vec::new();
        for ann in &self.secondary_annotations {
            let ann_display = self.module.display_range(ann.range);
            let ann_start = ann_display.start.line_within_file();
            let ann_end = ann_display.end.line_within_file();
            let is_nearby = ann_start
                .to_zero_indexed()
                .abs_diff(primary_end_line.to_zero_indexed())
                <= MAX_LINES
                && ann_end
                    .to_zero_indexed()
                    .abs_diff(primary_start_line.to_zero_indexed())
                    <= MAX_LINES;
            if is_nearby {
                start_line = cmp::min(start_line, ann_start);
                end_line = cmp::max(end_line, ann_end);
                nearby_annotations.push(ann);
            } else {
                distant_annotations.push((ann, ann_display));
            }
        }

        let level = match self.severity {
            Severity::Error => Level::Error,
            Severity::Warn => Level::Warning,
            Severity::Info => Level::Info,
            Severity::Ignore => Level::None,
        };

        // Primary snippet with nearby annotations inline.
        let primary_snippet = self.make_snippet(
            origin,
            start_line,
            end_line,
            Some((self.range, level)),
            &nearby_annotations,
        );

        // Distant annotations each get their own snippet covering their full span.
        let mut message = Level::None.title("").snippet(primary_snippet);
        for (ann, ann_display) in &distant_annotations {
            let ann_start_line = ann_display.start.line_within_file();
            let ann_end_line = ann_display.end.line_within_file();
            message = message.snippet(self.make_snippet(
                origin,
                ann_start_line,
                ann_end_line,
                None,
                &[ann],
            ));
        }
        message
    }

    /// Build a source snippet for a line range with an optional primary annotation and
    /// secondary annotations. Used for both the main error snippet and distant annotation snippets.
    fn make_snippet<'a>(
        &'a self,
        origin: &'a str,
        from_line: LineNumber,
        to_line: LineNumber,
        primary: Option<(TextRange, Level)>,
        annotations: &[&'a SecondaryAnnotation],
    ) -> Snippet<'a> {
        // Warning: The SourceRange is char indexed, while the snippet is byte indexed.
        let source = self
            .module
            .lined_buffer()
            .content_in_line_range(from_line, to_line);
        let line_start = self.module.lined_buffer().line_start(from_line);
        let cell_line = self
            .module
            .display_range(TextRange::new(line_start, line_start))
            .start
            .line_within_cell()
            .get() as usize;
        let mut snippet = Snippet::source(source).line_start(cell_line).origin(origin);
        if let Some((range, lvl)) = primary {
            let start = (range.start() - line_start).to_usize();
            let end = cmp::min(start + range.len().to_usize(), source.len());
            snippet = snippet.annotation(lvl.span(start..end));
        }
        for ann in annotations {
            let start = ann
                .range
                .start()
                .to_usize()
                .saturating_sub(line_start.to_usize());
            let end = cmp::min(start + ann.range.len().to_usize(), source.len());
            if start <= end && end <= source.len() {
                snippet = snippet.annotation(Level::Warning.span(start..end).label(&ann.label));
            }
        }
        snippet
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
        // TODO: Map secondary_annotations to DiagnosticRelatedInformation for LSP clients.
        // This requires constructing a Url from the module path, which may not always succeed.
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
            tags: if self.error_kind() == ErrorKind::Deprecated {
                Some(vec![DiagnosticTag::DEPRECATED])
            } else {
                None
            },
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
            secondary_annotations: Vec::new(),
            quick_fixes: Vec::new(),
        }
    }

    /// Add a secondary labeled annotation to this error. These appear as additional
    /// underlined spans with labels in the source snippet.
    pub fn with_annotation(mut self, range: TextRange, label: String) -> Self {
        self.secondary_annotations.push(SecondaryAnnotation {
            range,
            label: label.into_boxed_str(),
        });
        self
    }

    pub fn with_quick_fix(mut self, quick_fix: ErrorQuickFix) -> Self {
        self.quick_fixes.push(quick_fix);
        self
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
        // UnusedIgnore errors cannot be suppressed - this prevents infinite loops
        // where suppressing an unused-ignore creates another unused-ignore.
        if self.error_kind == ErrorKind::UnusedIgnore {
            return false;
        }
        // Check both this kind's name and any parent kind's name, so that e.g.
        // `# pyrefly: ignore[bad-override]` also suppresses `bad-override-mutable-attribute`.
        self.error_kind.suppression_names().any(|name| {
            self.module
                .is_ignored(&self.display_range, name, enabled_ignores)
        })
    }

    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }

    /// Return the secondary annotations attached to this error.
    pub fn secondary_annotations(&self) -> &[SecondaryAnnotation] {
        &self.secondary_annotations
    }

    pub fn quick_fixes(&self) -> &[ErrorQuickFix] {
        &self.quick_fixes
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
    use crate::test::util::TestEnv;

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
 7 | | X
 8 | | X
 9 | | X
10 | | X
11 | | X
   | |__^
   |
"#,
        );
    }

    #[test]
    fn test_error_with_secondary_annotations() {
        // Source: "val * 2" where val is at bytes 0..3, * at 4, 2 at 6
        let source = "val * 2";
        let module_info = Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(PathBuf::from("test.py")),
            Arc::new(source.to_owned()),
        );
        let error = Error::new(
            module_info,
            // Primary span covers the whole expression
            TextRange::new(TextSize::new(0), TextSize::new(7)),
            vec1!["`*` is not supported between `int | str` and `int`".to_owned()],
            ErrorKind::UnsupportedOperation,
        )
        .with_annotation(
            TextRange::new(TextSize::new(0), TextSize::new(3)),
            "has type `int | str`".to_owned(),
        )
        .with_annotation(
            TextRange::new(TextSize::new(6), TextSize::new(7)),
            "has type `int`".to_owned(),
        );
        let root = PathBuf::new();
        let mut output = Vec::new();
        error
            .write_line(&mut Cursor::new(&mut output), root.as_path(), true)
            .unwrap();

        assert_eq!(
            str::from_utf8(&output).unwrap(),
            r#"ERROR `*` is not supported between `int | str` and `int` [unsupported-operation]
 --> test.py:1:1
  |
1 | val * 2
  | ---^^^-
  | |     |
  | |     has type `int`
  | has type `int | str`
  |
"#,
        );
    }

    /// Integration test: verify that binary operator errors from the type checker
    /// produce secondary annotations labeling both operands with their types.
    #[test]
    fn test_binop_error_has_type_annotations() {
        let code = r#"
def f(x: None) -> None:
    y = x * 2  # E: `*` is not supported between `None` and `Literal[2]`
"#;
        let (state, handle) = TestEnv::one("main", code).to_state();
        let errors = state
            .transaction()
            .get_errors(&[handle("main")])
            .collect_errors()
            .ordinary;
        assert_eq!(errors.len(), 1);
        let err = &errors[0];
        let annotations = err.secondary_annotations();
        assert_eq!(annotations.len(), 2);
        assert_eq!(&*annotations[0].label, "has type `None`");
        assert_eq!(&*annotations[1].label, "has type `Literal[2]`");
    }
}
