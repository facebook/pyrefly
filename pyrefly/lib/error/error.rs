/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::io;
use std::io::Write;

use itertools::Itertools;
use pyrefly_util::display::number_thousands;
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
    /// Note that this is formatted for pretty-printing, with a newline at the
    /// beginning and two spaces after every newline.
    msg_details: Option<Box<str>>,
    is_ignored: bool,
}

impl Error {
    pub fn write_line(&self, mut f: impl Write) -> io::Result<()> {
        writeln!(
            f,
            "{} {}:{}: {} [{}]{}",
            match self.error_kind().severity() {
                Severity::Error => "ERROR",
                Severity::Warn => " WARN",
                Severity::Info => " INFO",
            },
            self.path(),
            self.range,
            self.msg_header,
            self.error_kind.to_name(),
            self.msg_details(),
        )
    }

    pub fn print_colors(&self) {
        anstream::println!(
            "{} {}:{}: {} {}{}",
            match self.error_kind().severity() {
                Severity::Error => Paint::red("ERROR"),
                Severity::Warn => Paint::yellow(" WARN"),
                Severity::Info => Paint::green(" INFO"),
            },
            Paint::blue(&self.path().as_path().display()),
            Paint::dim(self.source_range()),
            Paint::new(&*self.msg_header),
            Paint::dim(format!("[{}]", self.error_kind().to_name()).as_str()),
            Paint::new(self.msg_details()),
        );
    }
}

#[cfg(test)]
pub fn print_errors(errors: &[Error]) {
    for err in errors {
        err.print_colors();
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
            Some(msg.map(|s| format!("\n  {s}")).join("").into_boxed_str())
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

    fn msg_details(&self) -> &str {
        self.msg_details.as_ref().map_or("", |s| s)
    }

    pub fn msg(&self) -> String {
        format!("{}{}", self.msg_header, self.msg_details())
    }

    pub fn is_ignored(&self) -> bool {
        self.is_ignored
    }

    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}
