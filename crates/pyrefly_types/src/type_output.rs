/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_python::module::TextRangeWithModule;
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

/// This struct is used to collect the type to be displayed as a vector. Each element
/// in the vector will be tuple of (String, Option<TextRangeWithModule>).
/// The String the actual part of the string that will be displayed. When displaying the type
/// each of these will be concatenated to create the final type.
/// The second element of the vector is an optional location. For any part that do have
/// a location this will be included. For separators like '|', '[', etc. this will be None.
pub struct OutputWithLocations<'a> {
    parts: Vec<(String, Option<TextRangeWithModule>)>,
    #[expect(dead_code)]
    context: &'a TypeDisplayContext<'a>,
}

impl<'a> OutputWithLocations<'a> {
    pub fn new(context: &'a TypeDisplayContext<'a>) -> Self {
        Self {
            parts: Vec::new(),
            context,
        }
    }

    pub fn parts(&self) -> &[(String, Option<TextRangeWithModule>)] {
        &self.parts
    }
}

impl TypeOutput for OutputWithLocations<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.parts.push((s.to_owned(), None));
        Ok(())
    }

    fn write_qname(&mut self, _qname: &QName) -> fmt::Result {
        Ok(())
    }

    fn write_lit(&mut self, _lit: &Lit) -> fmt::Result {
        Ok(())
    }

    fn write_targs(&mut self, _targs: &TArgs) -> fmt::Result {
        Ok(())
    }

    fn write_type(&mut self, _ty: &Type) -> fmt::Result {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_with_locations_write_str() {
        let context = TypeDisplayContext::default();
        let mut output = OutputWithLocations::new(&context);

        assert_eq!(output.parts().len(), 0);

        output.write_str("hello").unwrap();
        assert_eq!(output.parts().len(), 1);
        assert_eq!(output.parts()[0].0, "hello");
        assert!(output.parts()[0].1.is_none());

        output.write_str(" world").unwrap();
        assert_eq!(output.parts().len(), 2);
        assert_eq!(output.parts()[1].0, " world");
        assert!(output.parts()[1].1.is_none());

        let parts = output.parts();
        assert_eq!(parts[0].0, "hello");
        assert_eq!(parts[1].0, " world");
    }
}
