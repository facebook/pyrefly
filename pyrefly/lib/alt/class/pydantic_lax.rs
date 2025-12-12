/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::class::ClassType;
use pyrefly_types::keywords::ConverterMap;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::TArgs;
use pyrefly_types::types::Union;
use starlark_map::ordered_map::OrderedMap;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::class::Class;
use crate::types::types::Type;

const LAX_PREFIX: &str = "Lax";

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn lax_display_name_for_class(&self, cls: &ClassType) -> String {
        format!("{}{}", LAX_PREFIX, capitalize_first(cls.name().as_str()))
    }

    fn types_to_lax_union(&self, base_type: &ClassType, types: &[&ClassType]) -> Type {
        let display_name = self.lax_display_name_for_class(base_type);
        let expanded_types: Vec<Type> = types.iter().map(|cls| (*cls).clone().to_type()).collect();
        let mut union_type = self.unions(expanded_types);
        if let Type::Union(ref mut boxed_union) = union_type {
            boxed_union.display_name = Some(display_name);
        }
        union_type
    }

    fn expand_types(&self, types: &[Type]) -> Vec<Type> {
        types
            .iter()
            .map(|t| self.expand_type_for_lax_mode(t))
            .collect()
    }

    fn get_atomic_lax_conversion(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::ClassType(cls) if cls == self.stdlib.bool() => Some(self.types_to_lax_union(
                self.stdlib.bool(),
                &[
                    self.stdlib.bool(),
                    self.stdlib.int(),
                    self.stdlib.float(),
                    self.stdlib.str(),
                    self.stdlib.decimal(),
                ],
            )),
            Type::ClassType(cls) if cls == self.stdlib.int() || cls == self.stdlib.float() => {
                Some(self.types_to_lax_union(
                    cls,
                    &[
                        self.stdlib.int(),
                        self.stdlib.float(),
                        self.stdlib.bool(),
                        self.stdlib.str(),
                        self.stdlib.bytes(),
                        self.stdlib.decimal(),
                    ],
                ))
            }
            Type::ClassType(cls) if cls == self.stdlib.bytes() || cls == self.stdlib.str() => {
                Some(self.types_to_lax_union(
                    cls,
                    &[
                        self.stdlib.bytes(),
                        self.stdlib.str(),
                        self.stdlib.bytearray(),
                    ],
                ))
            }
            Type::ClassType(cls) if cls == self.stdlib.date() || cls == self.stdlib.datetime() => {
                Some(self.types_to_lax_union(
                    cls,
                    &[
                        self.stdlib.date(),
                        self.stdlib.datetime(),
                        self.stdlib.int(),
                        self.stdlib.float(),
                        self.stdlib.str(),
                        self.stdlib.bytes(),
                        self.stdlib.decimal(),
                    ],
                ))
            }
            Type::ClassType(cls) if cls == self.stdlib.time() || cls == self.stdlib.timedelta() => {
                Some(self.types_to_lax_union(
                    cls,
                    &[
                        cls,
                        self.stdlib.int(),
                        self.stdlib.float(),
                        self.stdlib.str(),
                        self.stdlib.bytes(),
                        self.stdlib.decimal(),
                    ],
                ))
            }
            Type::ClassType(cls) if cls == self.stdlib.decimal() => Some(self.types_to_lax_union(
                self.stdlib.decimal(),
                &[
                    self.stdlib.decimal(),
                    self.stdlib.int(),
                    self.stdlib.float(),
                    self.stdlib.str(),
                ],
            )),
            Type::ClassType(cls) if cls == self.stdlib.path() || cls == self.stdlib.uuid() => {
                Some(self.types_to_lax_union(cls, &[cls, self.stdlib.str()]))
            }
            _ => None,
        }
    }

    fn get_container_lax_conversion(
        &self,
        class_obj: &Class,
        expanded_targs: &[Type],
    ) -> Option<Type> {
        let first_ty = expanded_targs
            .first()
            .cloned()
            .unwrap_or_else(Type::any_implicit);
        // All single-element containers use Iterable to handle invariance issues
        // This allows passing any iterable type (list, set, deque, frozenset, etc.)
        // Note: dict is handled separately in expand_type_for_lax_mode to avoid expanding the key type
        if class_obj == self.stdlib.list_object()
            || class_obj == self.stdlib.set_object()
            || class_obj == self.stdlib.frozenset_object()
            || class_obj.has_toplevel_qname(ModuleName::collections().as_str(), "deque")
            || class_obj.has_toplevel_qname(ModuleName::typing().as_str(), "Sequence")
            || class_obj.has_toplevel_qname(ModuleName::typing().as_str(), "Iterable")
        {
            return Some(self.stdlib.iterable(first_ty).to_type());
        }
        None
    }

    fn get_tuple_element_type(&self, tuple: &Tuple) -> Type {
        match tuple {
            Tuple::Unbounded(elem) => self.expand_type_for_lax_mode(elem),
            Tuple::Concrete(elems) => {
                let expanded_elems = self.expand_types(elems);
                self.unions(expanded_elems)
            }
            // this case is not a valid pydantic case
            Tuple::Unpacked(_) => Type::any_explicit(),
        }
    }

    fn expand_type_for_lax_mode(&self, ty: &Type) -> Type {
        match ty {
            Type::None => ty.clone(),
            Type::Type(box inner) => Type::Type(Box::new(self.expand_type_for_lax_mode(inner))),
            // Tuple types: convert to Iterable[T] where T is a union of expanded element types
            Type::Tuple(tuple) => self
                .stdlib
                .iterable(self.get_tuple_element_type(tuple))
                .to_type(),
            // Container types: recursively expand all type arguments
            Type::ClassType(cls) if !cls.targs().as_slice().is_empty() => {
                let class_obj = cls.class_object();
                let targs = cls.targs().as_slice();

                // Special handling for dict: don't expand key type (Mapping is invariant in key)
                if class_obj == self.stdlib.dict_object() {
                    let key_ty = targs.first().cloned().unwrap_or_else(Type::any_implicit);
                    let val_ty = targs.get(1).cloned().unwrap_or_else(Type::any_implicit);
                    let expanded_val = self.expand_type_for_lax_mode(&val_ty);
                    return self.stdlib.mapping(key_ty, expanded_val).to_type();
                }

                let expanded_targs = self.expand_types(targs);

                // Check for container type conversions
                if let Some(converted) =
                    self.get_container_lax_conversion(class_obj, &expanded_targs)
                {
                    return converted;
                }

                let tparams = self.get_class_tparams(class_obj);
                Type::ClassType(ClassType::new(
                    class_obj.dupe(),
                    TArgs::new(tparams, expanded_targs),
                ))
            }
            Type::Union(box Union { members, .. }) => {
                let expanded_members = self.expand_types(members);
                self.unions(expanded_members)
            }
            // Known atomic types with conversion tables, or Any for everything else
            _ => self
                .get_atomic_lax_conversion(ty)
                .unwrap_or_else(Type::any_explicit),
        }
    }

    pub fn build_pydantic_lax_conversion_table(&self, field_types: &[Type]) -> ConverterMap {
        let mut table = OrderedMap::new();

        for field_ty in field_types {
            if table.contains_key(field_ty) {
                continue;
            }

            let expanded = self.expand_type_for_lax_mode(field_ty);
            table.insert(field_ty.clone(), expanded);
        }

        ConverterMap::from_map(table)
    }
}
