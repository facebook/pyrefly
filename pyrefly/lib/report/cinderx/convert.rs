/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Conversion from pyrefly's internal `Type` enum to the flat
//! `StructuredType` representation used in CinderX reports.

use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::literal::Lit;
use pyrefly_types::literal::Literal;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use pyrefly_util::display::Fmt;

use crate::report::cinderx::types::StructuredType;
use crate::report::cinderx::types::TypeTable;
use crate::report::cinderx::types::hash_bound_method;
use crate::report::cinderx::types::hash_callable;
use crate::report::cinderx::types::hash_class;
use crate::report::cinderx::types::hash_literal;
use crate::report::cinderx::types::hash_other_form;
use crate::report::cinderx::types::hash_variable;

/// Canonicalize a fully qualified class name for the CinderX protocol.
///
/// Currently a pass-through: we use raw pyrefly qnames (e.g. `builtins.int`,
/// `builtins.list`) rather than remapping to typing-module names. This
/// function exists as a hook point for future canonicalization if needed.
pub(crate) fn canonicalize_class_qname(raw: &str) -> String {
    raw.to_owned()
}

/// Format a `QName` as a fully qualified dot-separated string
/// (e.g. `module.Outer.Inner`).
pub(crate) fn qname_to_full_string(qname: &pyrefly_python::qname::QName) -> String {
    format!("{}", Fmt(|f| qname.fmt_with_module(f)))
}

/// Convert Callable/Function params and return type to a structured callable entry.
///
/// `defining_func` is the fully qualified name of the function (module + optional
/// class + function name). It is `None` for purely structural `Type::Callable`.
fn callable_to_structured(
    params: &Params,
    ret: &Type,
    defining_func: Option<String>,
    table: &mut TypeTable,
    pending_class_traits: &mut Vec<(usize, Class)>,
) -> usize {
    let param_indices: Vec<usize> = match params {
        Params::List(param_list) => param_list
            .items()
            .iter()
            .map(|p| type_to_structured(p.as_type(), table, pending_class_traits))
            .collect(),
        Params::Ellipsis | Params::Materialization => vec![],
        Params::ParamSpec(prefix, _) => prefix
            .iter()
            .map(|p| type_to_structured(p.ty(), table, pending_class_traits))
            .collect(),
    };
    let ret_idx = type_to_structured(ret, table, pending_class_traits);
    let param_hashes: Vec<u64> = param_indices.iter().map(|&i| table.hash_at(i)).collect();
    let ret_hash = table.hash_at(ret_idx);
    let hash = hash_callable(&param_hashes, ret_hash, defining_func.as_deref());
    let sty = StructuredType::Callable {
        params: param_indices,
        return_type: ret_idx,
        defining_func,
    };
    table.insert(sty, hash)
}

/// Convert a `Quantified` type parameter to a structured variable entry.
fn quantified_to_structured(
    q: &Quantified,
    table: &mut TypeTable,
    pending_class_traits: &mut Vec<(usize, Class)>,
) -> usize {
    let name = q.name.to_string();
    let bound_indices: Vec<usize> = match &q.restriction {
        Restriction::Bound(bound) => {
            vec![type_to_structured(bound, table, pending_class_traits)]
        }
        Restriction::Constraints(constraints) => constraints
            .iter()
            .map(|c| type_to_structured(c, table, pending_class_traits))
            .collect(),
        Restriction::Unrestricted => vec![],
    };
    let bound_hashes: Vec<u64> = bound_indices.iter().map(|&i| table.hash_at(i)).collect();
    let hash = hash_variable(&name, &bound_hashes);
    let sty = StructuredType::Variable {
        name,
        bounds: bound_indices,
    };
    table.insert(sty, hash)
}

/// Insert a simple class entry with no type arguments.
fn insert_simple_class(qname: &str, table: &mut TypeTable) -> usize {
    let hash = hash_class(qname, &[], &[]);
    let sty = StructuredType::Class {
        qname: qname.to_owned(),
        args: vec![],
        traits: vec![],
    };
    table.insert(sty, hash)
}

/// Insert a class entry with the given type argument indices.
fn insert_class_with_args(qname: &str, arg_indices: Vec<usize>, table: &mut TypeTable) -> usize {
    let arg_hashes: Vec<u64> = arg_indices.iter().map(|&i| table.hash_at(i)).collect();
    let hash = hash_class(qname, &arg_hashes, &[]);
    let sty = StructuredType::Class {
        qname: qname.to_owned(),
        args: arg_indices,
        traits: vec![],
    };
    table.insert(sty, hash)
}

/// Insert a simple other-form entry with no type arguments.
fn insert_simple_other_form(qname: &str, table: &mut TypeTable) -> usize {
    let hash = hash_other_form(qname, &[]);
    let sty = StructuredType::OtherForm {
        qname: qname.to_owned(),
        args: vec![],
    };
    table.insert(sty, hash)
}

/// Insert an other-form entry wrapping a single child type argument.
fn insert_wrapper_other_form(qname: &str, inner_idx: usize, table: &mut TypeTable) -> usize {
    let arg_hashes = vec![table.hash_at(inner_idx)];
    let hash = hash_other_form(qname, &arg_hashes);
    let sty = StructuredType::OtherForm {
        qname: qname.to_owned(),
        args: vec![inner_idx],
    };
    table.insert(sty, hash)
}

/// Insert an other-form entry with the given type argument indices.
fn insert_other_form_with_args(
    qname: &str,
    arg_indices: Vec<usize>,
    table: &mut TypeTable,
) -> usize {
    let arg_hashes: Vec<u64> = arg_indices.iter().map(|&i| table.hash_at(i)).collect();
    let hash = hash_other_form(qname, &arg_hashes);
    let sty = StructuredType::OtherForm {
        qname: qname.to_owned(),
        args: arg_indices,
    };
    table.insert(sty, hash)
}

/// Convert a pyrefly `Type` into a `StructuredType` entry in the type table.
///
/// Returns the table index of the inserted (or deduplicated) entry.
/// Newly inserted `ClassType` entries are appended to `pending_class_traits`
/// for post-processing (traits require solver access, done by the caller).
pub(crate) fn type_to_structured(
    ty: &Type,
    table: &mut TypeTable,
    pending_class_traits: &mut Vec<(usize, Class)>,
) -> usize {
    match ty {
        Type::ClassType(ct) => {
            let arg_indices: Vec<usize> = ct
                .targs()
                .as_slice()
                .iter()
                .map(|arg| type_to_structured(arg, table, pending_class_traits))
                .collect();
            let raw_qname = qname_to_full_string(ct.qname());
            let qname = canonicalize_class_qname(&raw_qname);
            let idx = insert_class_with_args(&qname, arg_indices, table);
            pending_class_traits.push((idx, ct.class_object().clone()));
            idx
        }
        Type::Union(box Union { members, .. }) => {
            let has_none = members.iter().any(|m| matches!(m, Type::None));
            let non_none: Vec<&Type> = members
                .iter()
                .filter(|m| !matches!(m, Type::None))
                .collect();

            if has_none && !non_none.is_empty() {
                // Optional[inner]: wrap the non-None part
                let inner_idx = if non_none.len() == 1 {
                    type_to_structured(non_none[0], table, pending_class_traits)
                } else {
                    let inner_union = Type::Union(Box::new(Union {
                        members: non_none.into_iter().cloned().collect(),
                        display_name: None,
                    }));
                    type_to_structured(&inner_union, table, pending_class_traits)
                };
                insert_wrapper_other_form("typing.Optional", inner_idx, table)
            } else if !has_none {
                // Union without None
                let arg_indices: Vec<usize> = members
                    .iter()
                    .map(|m| type_to_structured(m, table, pending_class_traits))
                    .collect();
                insert_other_form_with_args("typing.Union", arg_indices, table)
            } else {
                // All None (degenerate)
                type_to_structured(&Type::None, table, pending_class_traits)
            }
        }
        Type::Any(_) => insert_simple_other_form("typing.Any", table),
        Type::None => insert_simple_other_form("None", table),
        Type::Never(_) => insert_simple_other_form("typing.Never", table),
        Type::Type(inner) => {
            let inner_idx = type_to_structured(inner, table, pending_class_traits);
            insert_wrapper_other_form("typing.Type", inner_idx, table)
        }
        Type::TypedDict(td) | Type::PartialTypedDict(td) => {
            let (qname, trait_name) = match (ty, td) {
                (Type::PartialTypedDict(_), TypedDict::TypedDict(inner)) => {
                    (qname_to_full_string(inner.qname()), "partial_typed_dict")
                }
                (Type::PartialTypedDict(_), TypedDict::Anonymous(_)) => {
                    ("<anonymous>".to_owned(), "partial_typed_dict")
                }
                (_, TypedDict::TypedDict(inner)) => {
                    (qname_to_full_string(inner.qname()), "typed_dict")
                }
                (_, TypedDict::Anonymous(_)) => ("<anonymous>".to_owned(), "typed_dict"),
            };
            let hash = hash_class(&qname, &[], &[trait_name]);
            let sty = StructuredType::Class {
                qname,
                args: vec![],
                traits: vec![trait_name.to_owned()],
            };
            table.insert(sty, hash)
        }
        Type::BoundMethod(box bm) => {
            let self_idx = type_to_structured(&bm.obj, table, pending_class_traits);
            let func_type = bm.func.clone().as_type();
            let func_idx = type_to_structured(&func_type, table, pending_class_traits);
            let defining_class = bm.func.metadata().kind.class().map(|cls| {
                let raw = qname_to_full_string(cls.qname());
                canonicalize_class_qname(&raw)
            });
            let hash = hash_bound_method(
                table.hash_at(self_idx),
                table.hash_at(func_idx),
                defining_class.as_deref(),
            );
            let sty = StructuredType::BoundMethod {
                self_type: self_idx,
                func_type: func_idx,
                defining_class,
            };
            table.insert(sty, hash)
        }
        Type::TypeGuard(inner) => {
            let inner_idx = type_to_structured(inner, table, pending_class_traits);
            insert_wrapper_other_form("typing.TypeGuard", inner_idx, table)
        }
        Type::TypeIs(inner) => {
            let inner_idx = type_to_structured(inner, table, pending_class_traits);
            insert_wrapper_other_form("typing.TypeIs", inner_idx, table)
        }
        Type::Annotated(inner, _) => {
            // Annotated is transparent for type purposes
            type_to_structured(inner, table, pending_class_traits)
        }
        Type::ClassDef(class) => {
            // ClassDef has type `Type[class]`, so emit as typing.Type wrapping the class
            let raw_qname = qname_to_full_string(class.qname());
            let qname = canonicalize_class_qname(&raw_qname);
            let inner_idx = insert_simple_class(&qname, table);
            insert_wrapper_other_form("typing.Type", inner_idx, table)
        }
        Type::SelfType(ct) => {
            // Unwrap Self and treat as the underlying ClassType
            type_to_structured(&Type::ClassType(ct.clone()), table, pending_class_traits)
        }
        Type::TypeAlias(box data) | Type::UntypedAlias(box data) => match data {
            TypeAliasData::Value(ta) => {
                type_to_structured(&ta.as_type(), table, pending_class_traits)
            }
            TypeAliasData::Ref(_) => {
                // Recursive alias reference — fall back to Any
                insert_simple_other_form("typing.Any", table)
            }
        },
        Type::Ellipsis => insert_simple_other_form("builtins.ellipsis", table),
        Type::Tuple(tuple) => {
            let arg_indices: Vec<usize> = match tuple {
                pyrefly_types::tuple::Tuple::Concrete(elts) => elts
                    .iter()
                    .map(|e| type_to_structured(e, table, pending_class_traits))
                    .collect(),
                pyrefly_types::tuple::Tuple::Unbounded(box inner) => {
                    vec![type_to_structured(inner, table, pending_class_traits)]
                }
                pyrefly_types::tuple::Tuple::Unpacked(box (prefix, middle, suffix)) => {
                    let mut indices: Vec<usize> = prefix
                        .iter()
                        .map(|e| type_to_structured(e, table, pending_class_traits))
                        .collect();
                    indices.push(type_to_structured(middle, table, pending_class_traits));
                    indices.extend(
                        suffix
                            .iter()
                            .map(|e| type_to_structured(e, table, pending_class_traits)),
                    );
                    indices
                }
            };
            insert_other_form_with_args("typing.Tuple", arg_indices, table)
        }
        Type::Module(_) => insert_simple_other_form("types.ModuleType", table),
        Type::Overload(_) => insert_simple_other_form("typing.overload", table),
        Type::LiteralString(_) => insert_simple_other_form("typing.LiteralString", table),
        Type::Forall(box forall) => {
            // Unwrap Forall and recurse into the body
            type_to_structured(&forall.body.clone().as_type(), table, pending_class_traits)
        }
        Type::SuperInstance(box (ct, _)) => {
            // Treat super() as the underlying ClassType
            type_to_structured(&Type::ClassType(ct.clone()), table, pending_class_traits)
        }
        Type::KwCall(box kc) => type_to_structured(&kc.return_ty, table, pending_class_traits),
        // Callable kinds
        Type::Callable(box c) => {
            callable_to_structured(&c.params, &c.ret, None, table, pending_class_traits)
        }
        Type::Function(box f) => {
            let defining_func = {
                let kind = &f.metadata.kind;
                let module = kind.module_name();
                let class_prefix = match kind.class() {
                    Some(cls) => format!("{}.", cls.name()),
                    None => String::new(),
                };
                Some(format!("{module}.{class_prefix}{}", kind.function_name()))
            };
            callable_to_structured(
                &f.signature.params,
                &f.signature.ret,
                defining_func,
                table,
                pending_class_traits,
            )
        }
        // Variable kinds
        Type::Quantified(box q) | Type::QuantifiedValue(box q) => {
            quantified_to_structured(q, table, pending_class_traits)
        }
        Type::TypeVar(tv) => {
            let name = tv.qname().id().to_string();
            let bound_indices: Vec<usize> = match tv.restriction() {
                Restriction::Bound(bound) => {
                    vec![type_to_structured(bound, table, pending_class_traits)]
                }
                Restriction::Constraints(constraints) => constraints
                    .iter()
                    .map(|c| type_to_structured(c, table, pending_class_traits))
                    .collect(),
                Restriction::Unrestricted => vec![],
            };
            let bound_hashes: Vec<u64> = bound_indices.iter().map(|&i| table.hash_at(i)).collect();
            let hash = hash_variable(&name, &bound_hashes);
            let sty = StructuredType::Variable {
                name,
                bounds: bound_indices,
            };
            table.insert(sty, hash)
        }
        // Literal kind
        Type::Literal(box Literal { value, .. }) => {
            let promoted_idx = match value {
                Lit::Str(_) => insert_simple_class("builtins.str", table),
                Lit::Int(_) => insert_simple_class("builtins.int", table),
                Lit::Bool(_) => insert_simple_class("builtins.bool", table),
                Lit::Bytes(_) => insert_simple_class("builtins.bytes", table),
                Lit::Enum(e) => {
                    type_to_structured(&e.class.clone().to_type(), table, pending_class_traits)
                }
            };
            let value_str = format!("{}", value);
            let promoted_hash = table.hash_at(promoted_idx);
            let hash = hash_literal(&value_str, promoted_hash);
            let sty = StructuredType::Literal {
                value: value_str,
                promoted_type: promoted_idx,
            };
            table.insert(sty, hash)
        }
        // TODO(stroxler): These uncommon / internal types silently collapse to
        // typing.Any in the report. This is fine for the initial version but may
        // hide unsupported types; tighten this as CinderX coverage expands.
        Type::Materialization
        | Type::Var(_)
        | Type::Intersect(_)
        | Type::SpecialForm(_)
        | Type::Concatenate(..)
        | Type::ParamSpecValue(_)
        | Type::Args(_)
        | Type::Kwargs(_)
        | Type::ArgsValue(_)
        | Type::KwargsValue(_)
        | Type::Unpack(_)
        | Type::ParamSpec(_)
        | Type::TypeVarTuple(_)
        | Type::TypeForm(_)
        | Type::ElementOfTypeVarTuple(_)
        | Type::Tensor(_)
        | Type::NNModule(_)
        | Type::Size(_)
        | Type::Dim(_) => insert_simple_other_form("typing.Any", table),
    }
}
