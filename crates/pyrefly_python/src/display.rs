/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::DisplayWithCtx;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprYield;
use ruff_python_ast::ExprYieldFrom;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAugAssign;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::Module;

// Special module-specific types

impl Module {
    pub fn display<'a>(&'a self, x: &'a impl DisplayWith<Module>) -> impl Display + 'a {
        x.display_with(self)
    }
}

impl DisplayWith<Module> for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        // We have a special case for NoneLiteral because we might manufacture these
        // during type checking (e.g. in the return position)
        if let Expr::NoneLiteral(_) = self {
            write!(f, "None")
        } else {
            write!(f, "{}", m.code_at(self.range()))
        }
    }
}

impl DisplayWith<Module> for ExprYield {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for ExprYieldFrom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for ExprCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for Arguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for StmtAugAssign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.code_at(self.range()))
    }
}

impl DisplayWith<Module> for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.display_range(*self))
    }
}

impl DisplayWith<Module> for TextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, m: &Module) -> fmt::Result {
        write!(f, "{}", m.display_pos(*self))
    }
}
