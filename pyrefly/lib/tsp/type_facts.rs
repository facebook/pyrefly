/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `pyrefly/typeFacts`: the types of many expressions in one file, in one request.
//!
//! TSP's type requests answer one position at a time with a full `Type` built for
//! re-resolving declarations in an editor. Batch analysis clients such as linters
//! ask about hundreds of expressions per file and want to know what a type *is*:
//! which class (and its MRO), which module, which function and what it returns.
//! This request answers a list of ranges against one transaction with those facts.

use lsp_types::Range;
use pyrefly_build::handle::Handle;
use pyrefly_python::qname::QName;
use pyrefly_types::class::Class;
use pyrefly_types::function::FuncMetadata;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::OverloadType;
use pyrefly_types::types::Type;
use serde::Deserialize;
use serde::Serialize;

use crate::state::state::Transaction;

pub const TYPE_FACTS_METHOD: &str = "pyrefly/typeFacts";

/// How many levels of type arguments, union members and return types to describe.
/// Two covers `Callable[..., ClassType[T]]`-shaped questions without letting a
/// deeply generic type blow up the response.
const NESTING: usize = 2;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeFactsParams {
    pub uri: String,
    pub snapshot: i32,
    pub queries: Vec<TypeFactsQuery>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeFactsQuery {
    pub range: Range,
    /// Ask for the type the surrounding code expects at `range.start` (a call
    /// argument's parameter type, an annotated target's type) instead of the
    /// type computed for `range`. Unlike `typeServer/getExpectedType` there is no
    /// fallback to the computed type, so `null` means "no expectation here".
    #[serde(default)]
    pub expected: bool,
}

/// What a type is, reduced to what an analysis compares on. Qualified names are
/// the defining module plus the name (`pkg.mod.Cls`), not the re-exported path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TypeFact {
    /// A class object: `C` or `type[C]`. Type arguments are present when the
    /// class object is specialized, as in the annotation `list[int]`.
    Class {
        qname: String,
        mro: Vec<String>,
        #[serde(rename = "typeArgs", skip_serializing_if = "Vec::is_empty")]
        type_args: Vec<TypeFact>,
    },
    /// An instance of a class.
    Instance {
        qname: String,
        mro: Vec<String>,
        #[serde(rename = "typeArgs", skip_serializing_if = "Vec::is_empty")]
        type_args: Vec<TypeFact>,
    },
    Module {
        name: String,
    },
    /// A function, bound method, or overloaded function. `qname` is absent for
    /// functions without a definition (lambdas, synthesized callables).
    Function {
        #[serde(skip_serializing_if = "Option::is_none")]
        qname: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        returns: Option<Box<TypeFact>>,
    },
    Union {
        members: Vec<TypeFact>,
    },
    Any,
    /// Anything else, described by Pyrefly's display of the type.
    Other {
        display: String,
    },
}

/// Describe `ty`, resolving MROs through `handle`'s transaction.
pub fn type_fact(transaction: &Transaction, handle: &Handle, ty: &Type) -> TypeFact {
    describe(transaction, handle, ty, NESTING)
}

fn describe(transaction: &Transaction, handle: &Handle, ty: &Type, depth: usize) -> TypeFact {
    let nested = |types: &[Type]| -> Vec<TypeFact> {
        if depth == 0 {
            return Vec::new();
        }
        types
            .iter()
            .map(|t| describe(transaction, handle, t, depth - 1))
            .collect()
    };
    let function = |metadata: &FuncMetadata, ret: Option<&Type>| TypeFact::Function {
        qname: function_qname(metadata),
        returns: ret
            .filter(|_| depth > 0)
            .map(|r| Box::new(describe(transaction, handle, r, depth - 1))),
    };
    match ty {
        Type::ClassType(ct) => TypeFact::Instance {
            qname: qname(ct.class_object().qname()),
            mro: mro(transaction, handle, ct.class_object()),
            type_args: nested(ct.targs().as_slice()),
        },
        Type::ClassDef(cls) => TypeFact::Class {
            qname: qname(cls.qname()),
            mro: mro(transaction, handle, cls),
            type_args: Vec::new(),
        },
        Type::Type(inner) => match inner.as_ref() {
            Type::ClassType(ct) => TypeFact::Class {
                qname: qname(ct.class_object().qname()),
                mro: mro(transaction, handle, ct.class_object()),
                type_args: nested(ct.targs().as_slice()),
            },
            // `type[A | None]` -- what an annotation like `A | None` evaluates to --
            // is described as the union of the classes it names, so a client reading
            // a field's annotation sees each member rather than an opaque display.
            Type::Union(union) => TypeFact::Union {
                members: union
                    .members
                    .iter()
                    .map(|member| {
                        describe(transaction, handle, &Type::type_of(member.clone()), depth)
                    })
                    .collect(),
            },
            _ => TypeFact::Other {
                display: ty.to_string(),
            },
        },
        Type::Module(module) => TypeFact::Module {
            name: module.to_string(),
        },
        Type::Function(f) => function(&f.metadata, Some(&f.signature.ret)),
        Type::Forall(forall) if let Forallable::Function(f) = &forall.body => {
            function(&f.metadata, Some(&f.signature.ret))
        }
        Type::BoundMethod(bound) => match &bound.func {
            BoundMethodType::Function(f) => function(&f.metadata, Some(&f.signature.ret)),
            BoundMethodType::Forall(forall) => {
                function(&forall.body.metadata, Some(&forall.body.signature.ret))
            }
            BoundMethodType::Overload(overload) => {
                function(&overload.metadata, overload_return(&overload.signatures))
            }
        },
        Type::Overload(overload) => {
            function(&overload.metadata, overload_return(&overload.signatures))
        }
        Type::Union(union) => TypeFact::Union {
            members: nested(&union.members),
        },
        Type::Any(_) => TypeFact::Any,
        _ => TypeFact::Other {
            display: ty.to_string(),
        },
    }
}

/// The first signature's return type: overloads of one function almost always
/// share the class an analysis cares about, and the facts are a summary.
fn overload_return(signatures: &[OverloadType]) -> Option<&Type> {
    signatures.first().map(|signature| match signature {
        OverloadType::Function(f) => &f.signature.ret,
        OverloadType::Forall(forall) => &forall.body.signature.ret,
    })
}

fn qname(name: &QName) -> String {
    format!("{}.{}", name.module_name(), name.id())
}

fn function_qname(metadata: &FuncMetadata) -> Option<String> {
    let symbol = metadata.kind.to_func_symbol()?;
    Some(match &symbol.cls {
        Some(cls) => format!("{}.{}.{}", symbol.module.name(), cls.name(), symbol.name),
        None => format!("{}.{}", symbol.module.name(), symbol.name),
    })
}

/// Ancestors of `cls` in MRO order, excluding `cls` itself and `object`. Empty
/// when the class's module cannot be solved from this handle.
fn mro(transaction: &Transaction, handle: &Handle, cls: &Class) -> Vec<String> {
    transaction
        .ad_hoc_solve(handle, "type_facts_mro", |solver| {
            solver
                .get_mro_for_class(cls)
                .ancestors_no_object()
                .iter()
                .map(|ancestor| qname(ancestor.class_object().qname()))
                .collect()
        })
        .unwrap_or_default()
}
