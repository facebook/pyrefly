/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::Union;

use crate::types::class::Class;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

pub(crate) const OVERLOAD_PARAM_EXPANSION_GAS: usize = 100;

pub(crate) fn expand_type_for_overload<EnumMembers, MakeTypeForm, MakeTuple>(
    ty: Type,
    enum_members: &EnumMembers,
    make_type_form: &MakeTypeForm,
    make_tuple: &MakeTuple,
    gas: usize,
) -> Vec<Type>
where
    EnumMembers: ?Sized + Fn(Class) -> Vec<Lit>,
    MakeTypeForm: ?Sized + Fn(Type) -> Type,
    MakeTuple: ?Sized + Fn(Vec<Type>) -> Type,
{
    match ty {
        Type::Union(box Union { members: ts, .. }) => ts,
        Type::ClassType(cls) if cls.is_builtin("bool") => vec![
            Lit::Bool(true).to_implicit_type(),
            Lit::Bool(false).to_implicit_type(),
        ],
        Type::ClassType(cls) => enum_members(cls.class_object().clone())
            .into_iter()
            .map(Lit::to_implicit_type)
            .collect(),
        Type::Type(box Type::Union(box Union { members: ts, .. })) => {
            ts.into_iter().map(|t| make_type_form(t)).collect()
        }
        Type::Tuple(Tuple::Concrete(elements)) => {
            let mut count: usize = 1;
            let mut changed = false;
            let mut element_expansions = Vec::new();
            for element in elements {
                let element_expansion = expand_type_for_overload(
                    element.clone(),
                    enum_members,
                    make_type_form,
                    make_tuple,
                    gas,
                );
                if element_expansion.is_empty() {
                    element_expansions.push(vec![element].into_iter());
                } else {
                    let len = element_expansion.len();
                    count = count.saturating_mul(len);
                    if count > gas {
                        return Vec::new();
                    }
                    changed = true;
                    element_expansions.push(element_expansion.into_iter());
                }
            }
            if count <= gas && changed {
                element_expansions
                    .into_iter()
                    .multi_cartesian_product()
                    .map(|elements| make_tuple(elements))
                    .collect()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}
