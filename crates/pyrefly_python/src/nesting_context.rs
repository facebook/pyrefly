/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Syntactical nesting context for classes and functions.

use std::fmt;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::intersperse_iter;
use ruff_text_size::Ranged;

use crate::module::Module;
use crate::module_name::ModuleName;
use crate::short_identifier::ShortIdentifier;

/// Represents the syntactical nesting context of a given class or function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Dupe)]
pub struct NestingContext(Arc<NestingContextInner>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NestingContextInner {
    Toplevel,
    Class(ShortIdentifier, NestingContext),
    Function(ShortIdentifier, NestingContext),
}

impl NestingContext {
    pub fn toplevel() -> Self {
        Self(Arc::new(NestingContextInner::Toplevel))
    }

    pub fn class(identifier: ShortIdentifier, parent: NestingContext) -> Self {
        Self(Arc::new(NestingContextInner::Class(identifier, parent)))
    }

    pub fn function(identifier: ShortIdentifier, parent: NestingContext) -> Self {
        Self(Arc::new(NestingContextInner::Function(identifier, parent)))
    }

    pub fn is_toplevel(&self) -> bool {
        matches!(self.0.as_ref(), NestingContextInner::Toplevel)
    }

    pub fn is_class(&self) -> bool {
        matches!(self.0.as_ref(), NestingContextInner::Class(_, _))
    }

    pub fn is_function(&self) -> bool {
        matches!(self.0.as_ref(), NestingContextInner::Function(_, _))
    }

    pub fn identifier(&self) -> Option<&ShortIdentifier> {
        match self.0.as_ref() {
            NestingContextInner::Class(id, _) => Some(id),
            NestingContextInner::Function(id, _) => Some(id),
            _ => None,
        }
    }

    pub fn parent(&self) -> Option<&NestingContext> {
        match self.0.as_ref() {
            NestingContextInner::Toplevel => None,
            NestingContextInner::Class(_, parent) => Some(parent),
            NestingContextInner::Function(_, parent) => Some(parent),
        }
    }

    /// Build a fully-qualified owner string for type parameter disambiguation,
    /// e.g. `"module.Outer.Inner.name"`. Includes all ancestor classes and functions.
    /// Returns `None` for builtins (where disambiguation is unnecessary).
    pub fn owner_path(&self, module: &Module, name: &str) -> Option<ruff_python_ast::name::Name> {
        let module_name = module.name();
        if module_name == ModuleName::builtins() {
            return None;
        }
        let ancestors = module.display(self).to_string();
        let owner_str = if ancestors.is_empty() {
            format!("{module_name}.{name}")
        } else {
            format!("{module_name}.{ancestors}.{name}")
        };
        Some(ruff_python_ast::name::Name::new(owner_str))
    }

    /// Build a dotted path of only the enclosing *function* ancestors (skipping classes),
    /// e.g. `"f1.f2"` for a function nested inside `f1` which is nested inside `f2`.
    /// Returns `None` if there are no enclosing functions.
    pub fn ancestor_function_path(&self, module: &Module) -> Option<ruff_python_ast::name::Name> {
        let mut parts: Vec<&str> = Vec::new();
        let mut ctx = self;
        loop {
            match ctx.0.as_ref() {
                NestingContextInner::Toplevel => break,
                NestingContextInner::Function(id, parent) => {
                    parts.push(module.code_at(id.range()));
                    ctx = parent;
                }
                NestingContextInner::Class(_, parent) => {
                    ctx = parent;
                }
            }
        }
        if parts.is_empty() {
            None
        } else {
            parts.reverse();
            Some(ruff_python_ast::name::Name::new(parts.join(".")))
        }
    }
}

impl DisplayWith<Module> for NestingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &Module) -> fmt::Result {
        let mut components = Vec::new();
        let mut current = self;
        loop {
            current = match current.0.as_ref() {
                NestingContextInner::Toplevel => break,
                NestingContextInner::Function(id, parent) => {
                    components.push(id);
                    parent
                }
                NestingContextInner::Class(id, parent) => {
                    components.push(id);
                    parent
                }
            }
        }
        write!(
            f,
            "{}",
            intersperse_iter(".", || components.iter().rev().map(|id| ctx.display(*id)))
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use ruff_python_ast::Identifier;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::module_name::ModuleName;
    use crate::module_path::ModulePath;

    #[test]
    fn test_nesting_context_display() {
        let module = Module::new(
            ModuleName::from_str("test"),
            ModulePath::filesystem(Path::new("test.py").to_owned()),
            Arc::new("class TestClass:\n    def test_func():\n        pass".to_owned()),
        );

        // NOTE: Use AI to fix these ranges if the module content ever changes
        let class_range = TextRange::new(TextSize::from(6), TextSize::from(15)); // "TestClass"
        let func_range = TextRange::new(TextSize::from(25), TextSize::from(34)); // "test_func"

        let class_id = ShortIdentifier::new(&Identifier::new("TestClass", class_range));
        let func_id = ShortIdentifier::new(&Identifier::new("test_func", func_range));

        let toplevel = NestingContext::toplevel();
        let class_context = NestingContext::class(class_id, toplevel.clone());
        let function_context = NestingContext::function(func_id, class_context.clone());

        assert_eq!(module.display(&toplevel).to_string(), "");
        assert_eq!(module.display(&class_context).to_string(), "TestClass");
        assert_eq!(
            module.display(&function_context).to_string(),
            "TestClass.test_func"
        );
    }
}
