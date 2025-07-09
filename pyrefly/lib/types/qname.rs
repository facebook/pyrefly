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

use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::module_info::ModuleInfo;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;

/// A name, plus where it is defined.
#[derive(Clone)]
pub struct QName {
    name: Identifier,
    module: ModuleInfo,
}

impl Debug for QName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QName")
            .field("name", &self.name)
            // The full details of ModuleInfo are pretty boring in most cases,
            // and we only cache it so we can defer expanding the range.
            // Therefore, shorten the Debug output, as ModuleInfo is pretty big.
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

impl TypeEq for QName {
    fn type_eq(&self, other: &Self, _: &mut TypeEqCtx) -> bool {
        self.name.id == other.name.id
            && self.module.name() == other.module.name()
            && self.module.path() == other.module.path()
    }
}

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

    pub fn new(name: Identifier, module: ModuleInfo) -> Self {
        Self { name, module }
    }

    pub fn id(&self) -> &Name {
        &self.name.id
    }

    pub fn range(&self) -> TextRange {
        self.name.range
    }

    pub fn module_info(&self) -> &ModuleInfo {
        &self.module
    }

    pub fn module_name(&self) -> ModuleName {
        self.module.name()
    }

    pub fn module_path(&self) -> &ModulePath {
        self.module.path()
    }

    pub fn fmt_name(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }

    pub fn fmt_with_module(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.module_name(), self.name)
    }

    pub fn fmt_with_location(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}@{}",
            self.module_name(),
            self.name,
            self.module.display_range(self.name.range)
        )
    }
}
