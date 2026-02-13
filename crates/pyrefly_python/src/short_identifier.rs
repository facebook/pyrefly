/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_util::display::DisplayWith;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::ExprName;
use ruff_python_ast::Identifier;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::module::Module;

/// An identifier, where we can drop the `Name` part because it came from a `Module`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShortIdentifier(TextRange);

impl ShortIdentifier {
    pub fn new(name: &Identifier) -> Self {
        Self(name.range)
    }

    pub fn expr_name(x: &ExprName) -> Self {
        // Not represented as an Identifier, but literally in the source code in the same way
        Self(x.range)
    }

    // Prefer `::new` or `::expr_name` over this, since this could create invalid identifiers.
    pub fn from_text_range(range: TextRange) -> Self {
        Self(range)
    }
}

impl PartialOrd for ShortIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShortIdentifier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.start().cmp(&other.0.start()) {
            std::cmp::Ordering::Equal => self.0.end().cmp(&other.0.end()),
            ord => ord,
        }
    }
}

impl Ranged for ShortIdentifier {
    fn range(&self) -> TextRange {
        self.0
    }
}

impl DisplayWith<Module> for ShortIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Module) -> fmt::Result {
        write!(f, "{}", ctx.code_at(self.0))
    }
}

impl<To: 'static> Visit<To> for ShortIdentifier {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}

impl<To: 'static> VisitMut<To> for ShortIdentifier {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use ruff_python_ast::Expr;

    use super::*;
    use crate::ast::Ast;
    use crate::module_name::ModuleName;
    use crate::module_path::ModulePath;

    fn from_expr(x: &Expr) -> ShortIdentifier {
        match x {
            Expr::Name(x) => ShortIdentifier::expr_name(x),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_short_identifier() {
        let module = Module::new(
            ModuleName::from_str("foo"),
            ModulePath::filesystem(Path::new("foo.py").to_owned()),
            Arc::new("hello_world = Baz123.attribute".to_owned()),
        );
        let ast = Ast::parse(module.contents(), module.source_type()).0;
        let show = |x: &ShortIdentifier| module.display(x).to_string();

        let assign = &ast.body[0].as_assign_stmt().unwrap();
        let attribute = assign.value.as_attribute_expr().unwrap();
        assert_eq!(show(&from_expr(&assign.targets[0])), "hello_world");
        assert_eq!(show(&from_expr(&attribute.value)), "Baz123");
        assert_eq!(show(&ShortIdentifier::new(&attribute.attr)), "attribute");
    }
}
