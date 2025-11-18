/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_python::qname::QName;

use crate::display::TypeDisplayContext;
use crate::literal::Lit;
use crate::types::TArgs;
use crate::types::Type;

/// A trait that will be will be used for formatting types, but also allow
/// additional functionality. The major difference between implementations
/// of this trait will be the location that type information is written to
/// For example, this will be used to write type information to a formatter as we do now,
/// and also allow us to collect the same types along with their location into a vector.
pub trait TypeOutput {
    fn write_str(&mut self, s: &str) -> fmt::Result;
    fn write_qname(&mut self, qname: &QName) -> fmt::Result;
    fn write_lit(&mut self, lit: &Lit) -> fmt::Result;
    fn write_targs(&mut self, targs: &TArgs) -> fmt::Result;
    fn write_type(&mut self, ty: &Type) -> fmt::Result;
}

pub struct DisplayOutput<'a, 'b, 'f> {
    context: &'a TypeDisplayContext<'a>,
    formatter: &'b mut fmt::Formatter<'f>,
}

impl<'a, 'b, 'f> DisplayOutput<'a, 'b, 'f> {
    pub fn new(context: &'a TypeDisplayContext<'a>, formatter: &'b mut fmt::Formatter<'f>) -> Self {
        Self { context, formatter }
    }
}

impl<'a, 'b, 'f> TypeOutput for DisplayOutput<'a, 'b, 'f> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.formatter.write_str(s)
    }

    fn write_qname(&mut self, q: &QName) -> fmt::Result {
        self.context.fmt_qname(q, self.formatter)
    }

    fn write_lit(&mut self, lit: &Lit) -> fmt::Result {
        self.context.fmt_lit(lit, self.formatter)
    }

    fn write_targs(&mut self, targs: &TArgs) -> fmt::Result {
        self.context.fmt_targs(targs, self.formatter)
    }

    fn write_type(&mut self, ty: &Type) -> fmt::Result {
        write!(self.formatter, "{}", self.context.display(ty))
    }
}
