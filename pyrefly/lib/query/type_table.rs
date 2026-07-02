/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::hash::Hasher;

use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Param;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::PrefixParam;
use pyrefly_types::callable_residual::CallableResidualKind;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDict;
use pyrefly_types::types::NeverStyle;
use pyrefly_types::types::SuperObj;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use pyrefly_util::lined_buffer::PythonASTRange;
use serde::Serialize;
use xxhash_rust::xxh64::Xxh64;

use super::TypeShapeContext;
use super::TypeShapeTrait;
use super::literal_value_shape_name;
use super::qname_to_string;
use super::typed_dict_traits;

#[derive(Debug, Clone, Serialize)]
pub struct LocatedTypeTableRef {
    pub location: PythonASTRange,
    #[serde(rename = "type")]
    pub type_index: usize,
    pub display: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexedTypeShapeKind {
    Named {
        name: String,
        args: Vec<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        unspecified_type_arg_count: Option<usize>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        traits: Vec<TypeShapeTrait>,
    },
    Callable {
        params: Vec<usize>,
        return_type: usize,
    },
    TypeVariable {
        name: String,
        bounds: Vec<usize>,
    },
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TypeTableResponseData {
    pub type_table: Vec<IndexedTypeShapeKind>,
    pub types: Vec<LocatedTypeTableRef>,
}

struct IndexedTypeTableEntry {
    kind: IndexedTypeShapeKind,
    hash: u64,
}

pub(super) struct TypeTableBuilder {
    entries: Vec<IndexedTypeTableEntry>,
    seen: HashMap<u64, Vec<usize>>,
}

impl TypeTableBuilder {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            seen: HashMap::new(),
        }
    }

    fn insert(&mut self, kind: IndexedTypeShapeKind, hash: u64) -> usize {
        if let Some(index) = self.seen.get(&hash).and_then(|indices| {
            indices
                .iter()
                .copied()
                .find(|index| self.entries[*index].kind == kind)
        }) {
            return index;
        }
        let index = self.entries.len();
        self.entries.push(IndexedTypeTableEntry { kind, hash });
        self.seen.entry(hash).or_default().push(index);
        index
    }

    fn hash_at(&self, index: usize) -> u64 {
        self.entries[index].hash
    }

    pub(super) fn into_type_table(self) -> Vec<IndexedTypeShapeKind> {
        self.entries.into_iter().map(|entry| entry.kind).collect()
    }
}

const HASH_KIND_NAMED: u8 = 0;
const HASH_KIND_CALLABLE: u8 = 1;
const HASH_KIND_TYPE_VARIABLE: u8 = 2;

fn hash_bytes(h: &mut Xxh64, bytes: &[u8]) {
    h.write_usize(bytes.len());
    h.write(bytes);
}

fn hash_hashes(h: &mut Xxh64, hashes: &[u64]) {
    h.write_usize(hashes.len());
    for hash in hashes {
        h.write_u64(*hash);
    }
}

fn hash_named(name: &str, arg_hashes: &[u64], unspecified_type_arg_count: Option<usize>) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_NAMED);
    hash_bytes(&mut h, name.as_bytes());
    hash_hashes(&mut h, arg_hashes);
    match unspecified_type_arg_count {
        Some(count) => {
            h.write_u8(1);
            h.write_usize(count);
        }
        None => h.write_u8(0),
    }
    h.finish()
}

fn hash_callable(param_hashes: &[u64], return_hash: u64) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_CALLABLE);
    hash_hashes(&mut h, param_hashes);
    h.write_u64(return_hash);
    h.finish()
}

fn hash_type_variable(name: &str, bound_hashes: &[u64]) -> u64 {
    let mut h = Xxh64::new(0);
    h.write_u8(HASH_KIND_TYPE_VARIABLE);
    hash_bytes(&mut h, name.as_bytes());
    hash_hashes(&mut h, bound_hashes);
    h.finish()
}

fn insert_indexed_named(
    table: &mut TypeTableBuilder,
    name: impl Into<String>,
    args: Vec<usize>,
    unspecified_type_arg_count: Option<usize>,
    traits: Vec<TypeShapeTrait>,
) -> usize {
    let name = name.into();
    let arg_hashes = args
        .iter()
        .map(|arg| table.hash_at(*arg))
        .collect::<Vec<_>>();
    let hash = hash_named(&name, &arg_hashes, unspecified_type_arg_count);
    table.insert(
        IndexedTypeShapeKind::Named {
            name,
            args,
            unspecified_type_arg_count,
            traits,
        },
        hash,
    )
}

fn insert_indexed_type_variable(
    table: &mut TypeTableBuilder,
    name: impl Into<String>,
    bounds: Vec<usize>,
) -> usize {
    let name = name.into();
    let bound_hashes = bounds
        .iter()
        .map(|bound| table.hash_at(*bound))
        .collect::<Vec<_>>();
    let hash = hash_type_variable(&name, &bound_hashes);
    table.insert(IndexedTypeShapeKind::TypeVariable { name, bounds }, hash)
}

fn insert_indexed_callable(
    table: &mut TypeTableBuilder,
    params: Vec<usize>,
    return_type: usize,
) -> usize {
    let param_hashes = params
        .iter()
        .map(|param| table.hash_at(*param))
        .collect::<Vec<_>>();
    let hash = hash_callable(&param_hashes, table.hash_at(return_type));
    table.insert(
        IndexedTypeShapeKind::Callable {
            params,
            return_type,
        },
        hash,
    )
}

fn indexed_named_leaf(table: &mut TypeTableBuilder, name: impl Into<String>) -> usize {
    insert_indexed_named(table, name, Vec::new(), None, Vec::new())
}

pub(super) fn type_to_indexed_shape(
    context: &TypeShapeContext,
    ty: &Type,
    table: &mut TypeTableBuilder,
) -> usize {
    match ty {
        Type::ClassDef(cls) => {
            let name = qname_to_string(cls.qname());
            let class_index = insert_indexed_named(
                table,
                name,
                Vec::new(),
                context.declared_type_param_arity_for_class(cls),
                Vec::new(),
            );
            insert_indexed_named(table, "typing.Type", vec![class_index], None, Vec::new())
        }
        Type::ClassType(class_type) => {
            let args = class_type
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            insert_indexed_named(
                table,
                qname_to_string(class_type.qname()),
                args,
                None,
                Vec::new(),
            )
        }
        Type::TypedDict(typed_dict) => {
            typed_dict_to_indexed_shape(context, typed_dict, false, table)
        }
        Type::PartialTypedDict(typed_dict) => {
            typed_dict_to_indexed_shape(context, typed_dict, true, table)
        }
        Type::Type(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "typing.Type", vec![inner], None, Vec::new())
        }
        Type::Callable(callable) => callable_to_indexed_shape(context, callable, table),
        Type::Function(function) => callable_to_indexed_shape(context, &function.signature, table),
        Type::BoundMethod(bound_method) => {
            let function_type = bound_method.func.clone().as_type();
            let args = vec![
                type_to_indexed_shape(context, &bound_method.obj, table),
                type_to_indexed_shape(context, &function_type, table),
            ];
            insert_indexed_named(table, "BoundMethod", args, None, Vec::new())
        }
        Type::Overload(overload) => {
            let args = overload
                .signatures
                .iter()
                .map(|signature| type_to_indexed_shape(context, &signature.as_type(), table))
                .collect::<Vec<_>>();
            insert_indexed_named(table, "typing.Overload", args, None, Vec::new())
        }
        Type::Forall(forall) => {
            let body_type = forall.body.clone().as_type();
            type_to_indexed_shape(context, &body_type, table)
        }
        Type::Union(union) => union_to_indexed_shape(context, union, table),
        Type::Intersect(intersection) => {
            let (members, _fallback) = &**intersection;
            let args = members
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            insert_indexed_named(table, "Intersection", args, None, Vec::new())
        }
        Type::Tuple(tuple) => {
            let args = tuple_indexed_args(context, tuple, table);
            insert_indexed_named(
                table,
                "typing.Tuple",
                args,
                None,
                vec![TypeShapeTrait::Tuple],
            )
        }
        Type::Literal(literal) => {
            let value = indexed_named_leaf(table, literal_value_shape_name(&literal.value));
            insert_indexed_named(table, "typing.Literal", vec![value], None, Vec::new())
        }
        Type::Sentinel(sentinel) => {
            let value = indexed_named_leaf(table, format!("{}", sentinel));
            insert_indexed_named(table, "sentinel", vec![value], None, Vec::new())
        }
        Type::LiteralString(_) => indexed_named_leaf(table, "typing_extensions.LiteralString"),
        Type::Quantified(quantified) | Type::QuantifiedValue(quantified) => {
            quantified_to_indexed_shape(context, quantified, table)
        }
        Type::TypeVar(type_var) => {
            let bounds = restriction_indexed_bounds(context, type_var.restriction(), table);
            insert_indexed_type_variable(table, type_var.qname().id().to_string(), bounds)
        }
        Type::ParamSpec(param_spec) => {
            insert_indexed_type_variable(table, param_spec.qname().id().to_string(), Vec::new())
        }
        Type::TypeVarTuple(type_var_tuple) => {
            insert_indexed_type_variable(table, type_var_tuple.qname().id().to_string(), Vec::new())
        }
        Type::ElementOfTypeVarTuple(quantified) => {
            insert_indexed_type_variable(table, quantified.name.to_string(), Vec::new())
        }
        Type::TypeGuard(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "typing.TypeGuard", vec![inner], None, Vec::new())
        }
        Type::TypeIs(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "typing.TypeIs", vec![inner], None, Vec::new())
        }
        Type::Annotated(inner, metadata) => {
            let mut args = vec![type_to_indexed_shape(context, inner, table)];
            args.extend(
                metadata
                    .iter()
                    .map(|ty| type_to_indexed_shape(context, ty, table)),
            );
            insert_indexed_named(table, "typing.Annotated", args, None, Vec::new())
        }
        Type::Unpack(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "typing.Unpack", vec![inner], None, Vec::new())
        }
        Type::Concatenate(prefix, param_spec) => {
            let mut args = prefix
                .iter()
                .map(|param| prefix_param_to_indexed_shape(context, param, table))
                .collect::<Vec<_>>();
            args.push(type_to_indexed_shape(context, param_spec, table));
            insert_indexed_named(table, "typing.Concatenate", args, None, Vec::new())
        }
        Type::ParamSpecValue(params) => {
            let args = param_list_to_indexed_shapes(context, params, table);
            insert_indexed_named(table, "ParamSpecValue", args, None, Vec::new())
        }
        Type::Args(param_spec) | Type::ArgsValue(param_spec) => {
            let param_spec = param_spec_to_indexed_shape(context, param_spec, table);
            insert_indexed_named(table, "ParamSpecArgs", vec![param_spec], None, Vec::new())
        }
        Type::Kwargs(param_spec) | Type::KwargsValue(param_spec) => {
            let param_spec = param_spec_to_indexed_shape(context, param_spec, table);
            insert_indexed_named(table, "ParamSpecKwargs", vec![param_spec], None, Vec::new())
        }
        Type::Module(module) => {
            let module = indexed_named_leaf(table, module.to_string());
            insert_indexed_named(table, "Module", vec![module], None, Vec::new())
        }
        Type::TypeAlias(alias) | Type::UntypedAlias(alias) => {
            alias_to_indexed_shape(context, alias, table)
        }
        Type::SuperInstance(super_instance) => {
            let (start_class, obj) = &**super_instance;
            let object = match obj {
                SuperObj::Instance(class_type) | SuperObj::Class(class_type) => {
                    Type::ClassType(class_type.clone())
                }
            };
            let args = vec![
                type_to_indexed_shape(context, &Type::ClassType(start_class.clone()), table),
                type_to_indexed_shape(context, &object, table),
            ];
            insert_indexed_named(table, "super", args, None, Vec::new())
        }
        Type::SelfType(class_type) => {
            let args = class_type
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            insert_indexed_named(
                table,
                qname_to_string(class_type.qname()),
                args,
                None,
                Vec::new(),
            )
        }
        Type::CallableResidual(residual) => match &residual.kind {
            CallableResidualKind::Generic { quantified } => {
                quantified_to_indexed_shape(context, quantified, table)
            }
            CallableResidualKind::Overload { branches, .. } => {
                let args = branches
                    .iter()
                    .map(|branch| type_to_indexed_shape(context, &branch.ty, table))
                    .collect::<Vec<_>>();
                insert_indexed_named(table, "typing.Overload", args, None, Vec::new())
            }
        },
        Type::KwCall(call) => type_to_indexed_shape(context, &call.return_ty, table),
        Type::Any(_) => indexed_named_leaf(table, "typing.Any"),
        Type::Never(style) => indexed_named_leaf(
            table,
            match style {
                NeverStyle::NoReturn => "typing.NoReturn",
                NeverStyle::Never => "typing.Never",
            },
        ),
        Type::None => indexed_named_leaf(table, "None"),
        Type::SpecialForm(special_form) => indexed_named_leaf(table, special_form.to_string()),
        Type::Ellipsis => indexed_named_leaf(table, "..."),
        Type::Materialization => indexed_named_leaf(table, "Materialization"),
        Type::Var(_) => indexed_named_leaf(table, "typing.Any"),
        Type::ShapedArray(_) => indexed_named_leaf(table, "Tensor"),
        Type::NNModule(module) => {
            let args = module
                .class
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            insert_indexed_named(
                table,
                qname_to_string(module.class.qname()),
                args,
                None,
                Vec::new(),
            )
        }
        Type::Size(_) => indexed_named_leaf(table, "Size"),
        Type::Dim(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "Dim", vec![inner], None, Vec::new())
        }
        Type::TypeForm(inner) => {
            let inner = type_to_indexed_shape(context, inner, table);
            insert_indexed_named(table, "typing.TypeForm", vec![inner], None, Vec::new())
        }
    }
}

fn typed_dict_to_indexed_shape(
    context: &TypeShapeContext,
    typed_dict: &TypedDict,
    is_partial: bool,
    table: &mut TypeTableBuilder,
) -> usize {
    match typed_dict {
        TypedDict::TypedDict(inner) => {
            let args = inner
                .targs()
                .as_slice()
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            insert_indexed_named(
                table,
                qname_to_string(inner.qname()),
                args,
                None,
                typed_dict_traits(is_partial),
            )
        }
        TypedDict::Anonymous(_) if is_partial => insert_indexed_named(
            table,
            "NonTotalTypedDictionary",
            Vec::new(),
            None,
            typed_dict_traits(is_partial),
        ),
        TypedDict::Anonymous(_) => insert_indexed_named(
            table,
            "TypedDictionary",
            Vec::new(),
            None,
            typed_dict_traits(is_partial),
        ),
    }
}

fn callable_to_indexed_shape(
    context: &TypeShapeContext,
    callable: &Callable,
    table: &mut TypeTableBuilder,
) -> usize {
    let params = callable_param_indices(context, &callable.params, table);
    let return_type = type_to_indexed_shape(context, &callable.ret, table);
    insert_indexed_callable(table, params, return_type)
}

fn callable_param_indices(
    context: &TypeShapeContext,
    params: &Params,
    table: &mut TypeTableBuilder,
) -> Vec<usize> {
    match params {
        Params::List(params) => param_list_to_indexed_shapes(context, params, table),
        Params::ParamSpec(prefix, param_spec) => {
            let mut params = prefix
                .iter()
                .map(|param| prefix_param_to_indexed_shape(context, param, table))
                .collect::<Vec<_>>();
            params.push(type_to_indexed_shape(context, param_spec, table));
            params
        }
        Params::Ellipsis | Params::Materialization => Vec::new(),
    }
}

fn param_list_to_indexed_shapes(
    context: &TypeShapeContext,
    params: &ParamList,
    table: &mut TypeTableBuilder,
) -> Vec<usize> {
    params
        .items()
        .iter()
        .map(|param| param_to_indexed_shape(context, param, table))
        .collect()
}

fn prefix_param_to_indexed_shape(
    context: &TypeShapeContext,
    param: &PrefixParam,
    table: &mut TypeTableBuilder,
) -> usize {
    match param {
        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => {
            type_to_indexed_shape(context, ty, table)
        }
    }
}

fn param_to_indexed_shape(
    context: &TypeShapeContext,
    param: &Param,
    table: &mut TypeTableBuilder,
) -> usize {
    match param {
        Param::PosOnly(_, ty, _)
        | Param::Pos(_, ty, _)
        | Param::Varargs(_, ty)
        | Param::KwOnly(_, ty, _)
        | Param::Kwargs(_, ty) => type_to_indexed_shape(context, ty, table),
    }
}

fn tuple_indexed_args(
    context: &TypeShapeContext,
    tuple: &Tuple,
    table: &mut TypeTableBuilder,
) -> Vec<usize> {
    match tuple {
        Tuple::Concrete(elements) => elements
            .iter()
            .map(|ty| type_to_indexed_shape(context, ty, table))
            .collect(),
        Tuple::Unbounded(element) => vec![
            type_to_indexed_shape(context, element, table),
            indexed_named_leaf(table, "..."),
        ],
        Tuple::Unpacked(unpacked) => {
            let (prefix, middle, suffix) = &**unpacked;
            let mut args = prefix
                .iter()
                .map(|ty| type_to_indexed_shape(context, ty, table))
                .collect::<Vec<_>>();
            args.push(type_to_indexed_shape(context, middle, table));
            args.extend(
                suffix
                    .iter()
                    .map(|ty| type_to_indexed_shape(context, ty, table)),
            );
            args
        }
    }
}

fn param_spec_to_indexed_shape(
    context: &TypeShapeContext,
    param_spec: &Quantified,
    table: &mut TypeTableBuilder,
) -> usize {
    debug_assert_eq!(param_spec.kind, QuantifiedKind::ParamSpec);
    quantified_to_indexed_shape(context, param_spec, table)
}

fn quantified_to_indexed_shape(
    context: &TypeShapeContext,
    quantified: &Quantified,
    table: &mut TypeTableBuilder,
) -> usize {
    let bounds = quantified_restriction_indexed_bounds(context, &quantified.restriction, table);
    insert_indexed_type_variable(table, quantified.name.to_string(), bounds)
}

fn quantified_restriction_indexed_bounds(
    context: &TypeShapeContext,
    restriction: &Restriction,
    table: &mut TypeTableBuilder,
) -> Vec<usize> {
    match restriction {
        Restriction::Bound(bound) => vec![type_to_indexed_shape(context, bound, table)],
        Restriction::Constraints(_) | Restriction::Unrestricted => Vec::new(),
    }
}

fn restriction_indexed_bounds(
    context: &TypeShapeContext,
    restriction: &Restriction,
    table: &mut TypeTableBuilder,
) -> Vec<usize> {
    match restriction {
        Restriction::Bound(bound) => vec![type_to_indexed_shape(context, bound, table)],
        Restriction::Constraints(constraints) => constraints
            .iter()
            .map(|ty| type_to_indexed_shape(context, ty, table))
            .collect(),
        Restriction::Unrestricted => Vec::new(),
    }
}

fn alias_to_indexed_shape(
    context: &TypeShapeContext,
    alias: &TypeAliasData,
    table: &mut TypeTableBuilder,
) -> usize {
    match alias {
        TypeAliasData::Value(alias) => {
            let name = indexed_named_leaf(table, alias.name.to_string());
            let alias_type = type_to_indexed_shape(context, &alias.as_type(), table);
            insert_indexed_named(table, "TypeAlias", vec![name, alias_type], None, Vec::new())
        }
        TypeAliasData::Ref(alias) => {
            let args = alias
                .args
                .as_ref()
                .map(|args| {
                    args.as_slice()
                        .iter()
                        .map(|ty| type_to_indexed_shape(context, ty, table))
                        .collect()
                })
                .unwrap_or_default();
            insert_indexed_named(
                table,
                format!("{}.{}", alias.module_name, alias.name),
                args,
                None,
                Vec::new(),
            )
        }
    }
}

fn union_to_indexed_shape(
    context: &TypeShapeContext,
    union: &Union,
    table: &mut TypeTableBuilder,
) -> usize {
    if union
        .members
        .iter()
        .any(|member| matches!(member, Type::None))
    {
        let mut members = union
            .members
            .iter()
            .filter(|member| !matches!(member, Type::None))
            .map(|ty| type_to_indexed_shape(context, ty, table))
            .collect::<Vec<_>>();
        let inner = if members.len() == 1 {
            members.pop().expect("non-empty because members.len() == 1")
        } else {
            insert_indexed_named(table, "typing.Union", members, None, Vec::new())
        };
        insert_indexed_named(table, "typing.Optional", vec![inner], None, Vec::new())
    } else {
        let members = union
            .members
            .iter()
            .map(|ty| type_to_indexed_shape(context, ty, table))
            .collect::<Vec<_>>();
        insert_indexed_named(table, "typing.Union", members, None, Vec::new())
    }
}

pub(super) fn located_type_table_refs(
    types: Vec<(PythonASTRange, (usize, String))>,
) -> Vec<LocatedTypeTableRef> {
    types
        .into_iter()
        .map(|(location, (type_index, display))| LocatedTypeTableRef {
            location,
            type_index,
            display,
        })
        .collect()
}
