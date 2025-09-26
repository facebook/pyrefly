/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A qualified name - name plus its location.

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::Module;
use crate::module_name::ModuleName;
use crate::module_path::ModulePath;
use crate::nesting_context::NestingContext;
/// A name, plus where it is defined.
#[derive(Clone)]
pub struct QName {
    name: Identifier,
    parent: NestingContext,
    module: Module,
}

impl Debug for QName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QName")
            .field("name", &self.name)
            .field("parent", &self.parent)
            // The full details of Module are pretty boring in most cases,
            // and we only cache it so we can defer expanding the range.
            // Therefore, shorten the Debug output, as Module is pretty big.
            .field("module", &self.module.name())
            .field("path", &self.module.path())
            .finish()
    }
}

impl PartialEq for QName {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Hash for QName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl Eq for QName {}

impl PartialOrd for QName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QName {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl Display for QName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_location(f)
    }
}

impl QName {
    fn key(&self) -> (&Name, TextSize, TextSize, ModuleName, &ModulePath) {
        (
            &self.name.id,
            self.name.range.start(),
            self.name.range.end(),
            self.module.name(),
            self.module.path(),
        )
    }

    pub fn new(name: Identifier, parent: NestingContext, module: Module) -> Self {
        Self {
            name,
            parent,
            module,
        }
    }

    pub fn id(&self) -> &Name {
        &self.name.id
    }

    pub fn range(&self) -> TextRange {
        self.name.range
    }

    pub fn parent(&self) -> &NestingContext {
        &self.parent
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn module_name(&self) -> ModuleName {
        self.module.name()
    }

    pub fn module_path(&self) -> &ModulePath {
        self.module.path()
    }

    pub fn fmt_name(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.parent().is_toplevel() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}", self.module().display(self.parent()))?;
            write!(f, ".{}", self.name)
        }
    }

    pub fn fmt_with_module(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.module_name())?;
        if !self.parent().is_toplevel() {
            write!(f, ".{}", self.module().display(self.parent()))?;
        }
        write!(f, ".{}", self.name)
    }

    pub fn fmt_with_location(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.module_name())?;
        if !self.parent().is_toplevel() {
            write!(f, ".{}", self.module().display(self.parent()))?;
        }
        write!(
            f,
            ".{}@{}",
            self.name,
            self.module.display_range(self.name.range)
        )
    }
}
