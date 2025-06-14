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
use pyrefly_util::display::number_thousands;
use ruff_annotate_snippets::Level;
use ruff_annotate_snippets::Message;
use ruff_annotate_snippets::Renderer;
use ruff_annotate_snippets::Snippet;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;
use yansi::Paint;

use crate::error::kind::ErrorKind;
use crate::error::kind::Severity;
use crate::module::module_info::ModuleInfo;
use crate::module::module_info::SourceRange;
use crate::module::module_path::ModulePath;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Error {
    module_info: ModuleInfo,
    range: SourceRange,
    error_kind: ErrorKind,
    /// First line of the error message
    msg_header: Box<str>,
    /// The rest of the error message after the first line.
    /// Note that this is formatted for pretty-printing, with two spaces at the beginning and after every newline.
    msg_details: Option<Box<str>>,
    is_ignored: bool,
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
                self.range,
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
                Paint::dim(self.source_range()),
                Paint::new(&*self.msg_header),
                Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            );
        }
    }

    fn lossy_origin(&self) -> Cow<'_, str> {
        self.path().as_path().to_string_lossy()
    }

    fn get_source_snippet<'a>(&'a self, origin: &'a str) -> Message<'a> {
        let range = self.source_range();
        let source = self
            .module_info
            .content_in_line_range(range.start.line, range.end.line);
        let level = match self.error_kind().severity() {
            Severity::Error => Level::Error,
            Severity::Warn => Level::Warning,
            Severity::Info => Level::Info,
        };
        let span_start = range.start.column.to_zero_indexed();
        let span_end = span_start + self.module_info.to_text_range(range).len().to_usize();
        Level::None.title("").snippet(
            Snippet::source(source)
                .line_start(range.start.line.get())
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
        range: SourceRange,
        msg: Vec1<String>,
        is_ignored: bool,
        error_kind: ErrorKind,
    ) -> Self {
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
            error_kind,
            msg_header,
            msg_details,
            is_ignored,
        }
    }

    pub fn source_range(&self) -> &SourceRange {
        &self.range
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

    pub fn is_ignored(&self) -> bool {
        self.is_ignored
    }

    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}
