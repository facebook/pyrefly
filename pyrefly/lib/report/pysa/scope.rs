/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::ast::Ast;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::FuncDefIndex;
use ruff_python_ast::AnyNodeRef;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use serde::Serialize;
use starlark_map::Hashed;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::KeyClass;
use crate::binding::binding::KeyDecoratedFunction;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::context::ModuleAnswersContext;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeParent {
    Function { func_def_index: FuncDefIndex },
    Class { class_id: ClassId },
    TopLevel,
}

impl Serialize for ScopeParent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        enum ScopeParentHelper {
            Function { func_def_index: u32 },
            Class { class_id: u32 },
            TopLevel,
        }

        let helper = match self {
            ScopeParent::Function { func_def_index } => ScopeParentHelper::Function {
                func_def_index: func_def_index.0,
            },
            ScopeParent::Class { class_id } => ScopeParentHelper::Class {
                class_id: class_id.to_int(),
            },
            ScopeParent::TopLevel => ScopeParentHelper::TopLevel,
        };
        helper.serialize(serializer)
    }
}

pub fn get_scope_parent(context: &ModuleAnswersContext, range: TextRange) -> ScopeParent {
    Ast::locate_node(&context.ast, range.start())
        .iter()
        .find_map(|node| match node {
            AnyNodeRef::Identifier(id) if id.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) if class_def.name.range() == range => None,
            AnyNodeRef::StmtFunctionDef(fun_def) if fun_def.name.range() == range => None,
            AnyNodeRef::StmtClassDef(class_def) => {
                let key = KeyClass(ShortIdentifier::new(&class_def.name));
                let idx = context
                    .bindings
                    .key_to_idx_hashed_opt(Hashed::new(&key))
                    .unwrap();
                let class = context.answers.get_idx(idx).unwrap().0.dupe().unwrap();
                Some(ScopeParent::Class {
                    class_id: ClassId::from_class(&class),
                })
            }
            AnyNodeRef::StmtFunctionDef(fun_def) => {
                let key = KeyDecoratedFunction(ShortIdentifier::new(&fun_def.name));
                let idx = context
                    .bindings
                    .key_to_idx_hashed_opt(Hashed::new(&key))
                    .unwrap();
                let decorated_function = DecoratedFunction::from_bindings_answers(
                    idx,
                    &context.bindings,
                    &context.answers,
                );
                Some(ScopeParent::Function {
                    func_def_index: decorated_function.undecorated.def_index,
                })
            }
            _ => None,
        })
        .unwrap_or(ScopeParent::TopLevel)
}
