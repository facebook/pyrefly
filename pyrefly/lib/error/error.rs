/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::borrow::Cow;
use std::fmt::Debug;
use std::io;
use std::io::Write;

use itertools::Itertools;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::display::number_thousands;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LinedBuffer;
use ruff_annotate_snippets::Level;
use ruff_annotate_snippets::Message;
use ruff_annotate_snippets::Renderer;
use ruff_annotate_snippets::Snippet;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;
use yansi::Paint;

use crate::error::kind::ErrorKind;
use crate::error::kind::Severity;
use crate::module::module_info::ModuleInfo;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Error {
    module_info: ModuleInfo,
    range: TextRange,
    display_range: DisplayRange,
    error_kind: ErrorKind,
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
    pub fn write_line(&self, mut f: impl Write, verbose: bool) -> io::Result<()> {
        if verbose {
            writeln!(
                f,
                "{} {} [{}]",
                self.error_kind().severity().label(),
                self.msg_header,
                self.error_kind.to_name(),
            )?;
            let origin = self.lossy_origin();
            let snippet = self.get_source_snippet(&origin);
            let renderer = Renderer::plain();
            writeln!(f, "{}", renderer.render(snippet))?;
            if let Some(details) = &self.msg_details {
                writeln!(f, "{details}")?;
            }
        } else {
            writeln!(
                f,
                "{} {}:{}: {} [{}]",
                self.error_kind().severity().label(),
                self.path(),
                self.display_range,
                self.msg_header,
                self.error_kind.to_name(),
            )?;
        }
        Ok(())
    }

    pub fn print_colors(&self, verbose: bool) {
        if verbose {
            anstream::println!(
                "{} {} {}",
                self.error_kind().severity().painted(),
                Paint::new(&*self.msg_header),
                Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            );
            let origin = self.lossy_origin();
            let snippet = self.get_source_snippet(&origin);
            let renderer = Renderer::styled();
            anstream::println!("{}", renderer.render(snippet));
            if let Some(details) = &self.msg_details {
                anstream::println!("{details}");
            }
        } else {
            anstream::println!(
                "{} {}:{}: {} {}",
                self.error_kind().severity().painted(),
                Paint::blue(&self.path().as_path().display()),
                Paint::dim(self.display_range()),
                Paint::new(&*self.msg_header),
                Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            );
        }
    }

    fn lossy_origin(&self) -> Cow<'_, str> {
        self.path().as_path().to_string_lossy()
    }

    fn get_source_snippet<'a>(&'a self, origin: &'a str) -> Message<'a> {
        // Warning: The SourceRange is char indexed, while the snippet is byte indexed.
        //          Be careful in the conversion.
        // Question: Should we just keep the original TextRange around?
        let source = self
            .module_info
            .lined_buffer()
            .content_in_line_range(self.display_range.start.line, self.display_range.end.line);
        let line_start = self
            .module_info
            .lined_buffer()
            .line_start(self.display_range.start.line);

        let level = match self.error_kind().severity() {
            Severity::Error => Level::Error,
            Severity::Warn => Level::Warning,
            Severity::Info => Level::Info,
        };
        let span_start = (self.range.start() - line_start).to_usize();
        let span_end = span_start + self.range.len().to_usize();
        Level::None.title("").snippet(
            Snippet::source(source)
                .line_start(self.display_range.start.line.get() as usize)
                .origin(origin)
                .annotation(level.span(span_start..span_end)),
        )
    }
}

#[cfg(test)]
pub fn print_errors(errors: &[Error]) {
    for err in errors {
        err.print_colors(true);
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
    pub fn new(
        module_info: ModuleInfo,
        range: TextRange,
        msg: Vec1<String>,
        error_kind: ErrorKind,
    ) -> Self {
        let display_range = module_info.display_range(range);
        let msg_has_details = msg.len() > 1;
        let mut msg = msg.into_iter();
        let msg_header = msg.next().unwrap().into_boxed_str();
        let msg_details = if msg_has_details {
            Some(msg.map(|s| format!("  {s}")).join("\n").into_boxed_str())
        } else {
            None
        };
        Self {
            module_info,
            range,
            display_range,
            error_kind,
            msg_header,
            msg_details,
        }
    }

    pub fn display_range(&self) -> &DisplayRange {
        &self.display_range
    }

    pub fn lined_buffer(&self) -> &LinedBuffer {
        self.module_info.lined_buffer()
    }

    pub fn path(&self) -> &ModulePath {
        self.module_info.path()
    }

    pub fn msg_header(&self) -> &str {
        &self.msg_header
    }

    pub fn msg(&self) -> String {
        if let Some(details) = &self.msg_details {
            format!("{}\n{}", self.msg_header, details)
        } else {
            (*self.msg_header).to_owned()
        }
    }

    pub fn is_ignored(&self, permissive_ignores: bool) -> bool {
        self.module_info
            .is_ignored(&self.display_range, self.error_kind, permissive_ignores)
    }

    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}
