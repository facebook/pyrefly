/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Converts pyrefly's internal `Type` representation to TSP protocol types.
//!
//! The conversion maps pyrefly types to their closest TSP protocol equivalent:
//!  - `ClassType` → TSP `ClassType` with a `RegularDeclaration` pointing to
//!    the class definition in source (or bundled typeshed).
//!  - `ClassDef` → TSP `ClassType` with `Instantiable` flag.
//!  - `Function` → TSP `FunctionType` with declaration and return type.
//!  - `BoundMethod` → TSP `FunctionType` with `bound_to_type`.
//!  - `Literal` → TSP `ClassType` with `literal_value`.
//!  - `Union` → TSP `UnionType` (recursively converting members).
//!  - `Module` → TSP `ModuleType`.
//!  - `TypeVar`/`ParamSpec`/`TypeVarTuple` → TSP `TypeVarType` (`DeclaredType`).
//!  - `Forall` → unwraps body and converts recursively.
//!  - `Callable` → TSP `FunctionType` with synthesized declaration.
//!  - `Tuple` → TSP `ClassType` with type args.
//!  - `Tensor`/`NNModule` → TSP `ClassType` from their base class.
//!  - `TypeAlias` → unwraps to the aliased type.
//!  - `SpecialForm` → TSP `BuiltInType` with the form name.
//!  - `Any`, `Never`, `None`, `Ellipsis` → TSP `BuiltInType`.
//!  - Solver-internal types → TSP `BuiltInType` with a representative name.
//!
//! All `Type` variants are explicitly handled; no types fall through to a
//! generic `SynthesizedType` stub.

use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use lsp_types::Url;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncId;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType as PyreflyClassType;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::Type as PyreflyType;
use ruff_text_size::TextRange;
use tsp_types::BuiltInType;
use tsp_types::ClassType as TspClassType;
use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationKind;
use tsp_types::DeclaredType;
use tsp_types::FunctionType as TspFunctionType;
use tsp_types::LiteralValue;
use tsp_types::ModuleType as TspModuleType;
use tsp_types::Node;
use tsp_types::OverloadedType as TspOverloadedType;
use tsp_types::Position as TspPosition;
use tsp_types::Range as TspRange;
use tsp_types::RegularDeclaration;
use tsp_types::SynthesizedDeclaration;
use tsp_types::Type as TspType;
use tsp_types::TypeFlags;
use tsp_types::TypeKind;
use tsp_types::UnionType;

use crate::lsp::module_helpers::to_real_path;

/// Monotonic counter used to assign unique ids to converted types.
static NEXT_TYPE_ID: AtomicI32 = AtomicI32::new(1);

/// Generate a fresh, unique type id.
fn next_id() -> i32 {
    NEXT_TYPE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Callback that resolves a `FuncId` to the `TextRange` of the function
/// name in source. When available, the resolver looks up the range via the
/// binding table's `KeyUndecoratedFunctionRange` entry for the function's
/// `FuncDefIndex`, avoiding the need to store ranges on every `FuncId`.
pub type FuncRangeResolver<'a> = dyn Fn(&FuncId) -> Option<TextRange> + 'a;

/// Convert a pyrefly `Type` to a TSP protocol `Type`.
///
/// Function declarations will have zero-range nodes since no binding
/// resolver is available. Use [`convert_type_with_resolver`] when source
/// locations are needed.
#[cfg(test)]
pub fn convert_type(ty: &PyreflyType) -> TspType {
    TypeConverter {
        resolve_func_range: None,
    }
    .convert(ty)
}

/// Convert a pyrefly `Type` to a TSP protocol `Type`, using `resolver` to
/// look up function declaration source ranges from `FuncDefIndex`.
pub fn convert_type_with_resolver<'a>(
    ty: &PyreflyType,
    resolver: &'a FuncRangeResolver<'a>,
) -> TspType {
    TypeConverter {
        resolve_func_range: Some(resolver),
    }
    .convert(ty)
}

/// Holds an optional range resolver and drives recursive type conversion.
struct TypeConverter<'a> {
    resolve_func_range: Option<&'a FuncRangeResolver<'a>>,
}

impl TypeConverter<'_> {
    /// Convert a pyrefly `Type` to a TSP protocol `Type`.
    fn convert(&self, ty: &PyreflyType) -> TspType {
        match ty {
            // --- Built-in special types ---
            PyreflyType::Any(_) => builtin("any"),
            PyreflyType::Never(_) => builtin("never"),
            PyreflyType::None => builtin("none"),
            PyreflyType::Ellipsis => builtin("ellipsis"),

            // --- Class instances (int, str, list[int], user-defined classes, etc.) ---
            PyreflyType::ClassType(ct) => self.convert_class_type(ct, TypeFlags::INSTANCE),

            // --- Class definitions (the class object itself, e.g. `type[int]`) ---
            PyreflyType::ClassDef(cls) => convert_class_def(cls),

            // --- Literals (Literal[42], Literal["hi"], etc.) ---
            PyreflyType::Literal(lit) => convert_literal(lit),

            // --- Functions ---
            PyreflyType::Function(func) => {
                self.convert_function(&func.signature, &func.metadata.kind, None)
            }

            // --- Bound methods ---
            PyreflyType::BoundMethod(bm) => {
                let bound_to = Some(Box::new(self.convert(&bm.obj)));
                match &bm.func {
                    BoundMethodType::Function(f) => {
                        self.convert_function(&f.signature, &f.metadata.kind, bound_to)
                    }
                    BoundMethodType::Forall(f) => {
                        self.convert_function(&f.body.signature, &f.body.metadata.kind, bound_to)
                    }
                    BoundMethodType::Overload(overload) => {
                        self.convert_overload_to_tsp(overload, bound_to)
                    }
                }
            }

            // --- Callable (typing.Callable[[int, str], bool]) ---
            PyreflyType::Callable(c) => self.convert_callable(c),

            // --- Unions ---
            PyreflyType::Union(u) => {
                let sub_types: Vec<TspType> = u.members.iter().map(|m| self.convert(m)).collect();
                TspType::Union(UnionType {
                    flags: TypeFlags::NONE,
                    id: next_id(),
                    kind: TypeKind::Union,
                    sub_types,
                    type_alias_info: None,
                })
            }

            // --- Modules ---
            PyreflyType::Module(m) => TspType::Module(TspModuleType {
                flags: TypeFlags::NONE,
                id: next_id(),
                kind: TypeKind::Module,
                module_name: m.to_string(),
                type_alias_info: None,
                uri: String::new(),
            }),

            // --- TypedDicts are instances of their class ---
            PyreflyType::TypedDict(td) | PyreflyType::PartialTypedDict(td) => {
                if let pyrefly_types::typed_dict::TypedDict::TypedDict(inner) = td {
                    let cls = inner.class_object();
                    let declaration = make_class_declaration(cls);
                    TspType::Class(TspClassType {
                        declaration: Declaration::Regular(declaration),
                        flags: TypeFlags::INSTANCE,
                        id: next_id(),
                        kind: TypeKind::Class,
                        literal_value: None,
                        type_alias_info: None,
                        type_args: None,
                    })
                } else {
                    // Anonymous TypedDict — no class backing
                    builtin("TypedDict")
                }
            }

            // --- Overloaded functions ---
            PyreflyType::Overload(overload) => self.convert_overload_to_tsp(overload, None),

            // --- Forall (generic functions/callables) — unwrap body ---
            PyreflyType::Forall(forall) => match &forall.body {
                Forallable::Function(f) => {
                    self.convert_function(&f.signature, &f.metadata.kind, None)
                }
                Forallable::Callable(c) => self.convert_callable(c),
                Forallable::TypeAlias(ta) => match ta {
                    pyrefly_types::type_alias::TypeAliasData::Value(alias) => {
                        self.convert(&alias.as_type())
                    }
                    pyrefly_types::type_alias::TypeAliasData::Ref(r) => builtin(r.name.as_str()),
                },
            },

            // --- Tuples → ClassType for `tuple` with type args ---
            PyreflyType::Tuple(t) => {
                let type_args = match t {
                    pyrefly_types::tuple::Tuple::Concrete(elts) if !elts.is_empty() => {
                        Some(elts.iter().map(|e| self.convert(e)).collect())
                    }
                    pyrefly_types::tuple::Tuple::Unbounded(elem) => {
                        Some(vec![self.convert(elem.as_ref())])
                    }
                    _ => None,
                };
                TspType::Class(TspClassType {
                    declaration: Declaration::Synthesized(SynthesizedDeclaration {
                        kind: DeclarationKind::Synthesized,
                        uri: String::new(),
                    }),
                    flags: TypeFlags::INSTANCE,
                    id: next_id(),
                    kind: TypeKind::Class,
                    literal_value: None,
                    type_alias_info: None,
                    type_args,
                })
            }

            // --- type[X] wrapper ---
            PyreflyType::Type(inner) => {
                let inner_tsp = self.convert(inner);
                // Return the inner type but mark it as instantiable
                match inner_tsp {
                    TspType::Class(mut c) => {
                        c.flags = TypeFlags::INSTANTIABLE;
                        TspType::Class(c)
                    }
                    other => other,
                }
            }

            // --- SelfType is a class type ---
            PyreflyType::SelfType(ct) => self.convert_class_type(ct, TypeFlags::INSTANCE),

            // --- TypeVar, ParamSpec, TypeVarTuple → TSP TypeVarType (DeclaredType) ---
            PyreflyType::TypeVar(tv) => {
                let qname = tv.qname();
                TspType::Var(make_typevar_declared(qname))
            }
            PyreflyType::ParamSpec(ps) => {
                let qname = ps.qname();
                TspType::Var(make_typevar_declared(qname))
            }
            PyreflyType::TypeVarTuple(tvt) => {
                let qname = tvt.qname();
                TspType::Var(make_typevar_declared(qname))
            }

            // --- Quantified / QuantifiedValue (type params during solving) ---
            PyreflyType::Quantified(q) | PyreflyType::QuantifiedValue(q) => {
                builtin(q.name.as_str())
            }

            // --- LiteralString → built-in ---
            PyreflyType::LiteralString(_) => builtin("LiteralString"),

            // --- Annotated[X, ...] → unwrap to X ---
            PyreflyType::Annotated(inner, _) => self.convert(inner),

            // --- TypeGuard[X] / TypeIs[X] → convert as bool (the runtime return type) ---
            PyreflyType::TypeGuard(_) | PyreflyType::TypeIs(_) => builtin("bool"),

            // --- SuperInstance → convert as the class type ---
            PyreflyType::SuperInstance(si) => self.convert_class_type(&si.0, TypeFlags::INSTANCE),

            // --- Tensor → ClassType from base_class ---
            PyreflyType::Tensor(t) => self.convert_class_type(&t.base_class, TypeFlags::INSTANCE),

            // --- NNModule → ClassType from class ---
            PyreflyType::NNModule(m) => self.convert_class_type(&m.class, TypeFlags::INSTANCE),

            // --- TypeAlias → unwrap to the aliased type, or builtin for refs ---
            PyreflyType::TypeAlias(ta) | PyreflyType::UntypedAlias(ta) => match ta.as_ref() {
                pyrefly_types::type_alias::TypeAliasData::Value(alias) => {
                    self.convert(&alias.as_type())
                }
                pyrefly_types::type_alias::TypeAliasData::Ref(r) => builtin(r.name.as_str()),
            },

            // --- SpecialForm → built-in with the form name ---
            PyreflyType::SpecialForm(sf) => builtin(&sf.to_string()),

            // --- Unpack(X) → convert inner ---
            PyreflyType::Unpack(inner) => self.convert(inner),

            // --- TypeForm(X) → convert inner ---
            PyreflyType::TypeForm(inner) => self.convert(inner),

            // --- Intersect → convert the fallback type ---
            PyreflyType::Intersect(pair) => self.convert(&pair.1),

            // --- ElementOfTypeVarTuple → builtin with name ---
            PyreflyType::ElementOfTypeVarTuple(q) => builtin(q.name.as_str()),

            // --- ParamSpec-related internal types → builtin with name ---
            PyreflyType::Args(q)
            | PyreflyType::Kwargs(q)
            | PyreflyType::ArgsValue(q)
            | PyreflyType::KwargsValue(q) => builtin(q.name.as_str()),

            // --- ParamSpecValue → built-in ---
            PyreflyType::ParamSpecValue(_) => builtin("ParamSpec"),

            // --- Concatenate → built-in ---
            PyreflyType::Concatenate(..) => builtin("Concatenate"),

            // --- KwCall → convert the return type ---
            PyreflyType::KwCall(kw) => self.convert(&kw.return_ty),

            // --- Size / Dim → int (they represent integer dimensions) ---
            PyreflyType::Size(_) | PyreflyType::Dim(_) => builtin("int"),

            // --- Solver-internal variable → built-in unknown ---
            PyreflyType::Var(_) => builtin("Unknown"),

            // --- Materialization is a solver artifact ---
            PyreflyType::Materialization => builtin("Unknown"),
        }
    }

    /// Convert a pyrefly `ClassType` (an instantiated class) to a TSP `ClassType`.
    fn convert_class_type(&self, ct: &PyreflyClassType, flags: TypeFlags) -> TspType {
        let cls = ct.class_object();
        let declaration = make_class_declaration(cls);
        let type_args: Option<Vec<TspType>> = {
            let args = ct.targs();
            let slice = args.as_slice();
            if slice.is_empty() {
                None
            } else {
                Some(slice.iter().map(|t| self.convert(t)).collect())
            }
        };

        TspType::Class(TspClassType {
            declaration: Declaration::Regular(declaration),
            flags,
            id: next_id(),
            kind: TypeKind::Class,
            literal_value: None,
            type_alias_info: None,
            type_args,
        })
    }

    /// Convert a `typing.Callable` to a TSP `FunctionType` with synthesized declaration.
    fn convert_callable(&self, callable: &Callable) -> TspType {
        let ret = self.convert(&callable.ret);
        TspType::Function(TspFunctionType {
            bound_to_type: None,
            declaration: Declaration::Synthesized(SynthesizedDeclaration {
                kind: DeclarationKind::Synthesized,
                uri: String::new(),
            }),
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            kind: TypeKind::Function,
            return_type: Some(Box::new(ret)),
            specialized_types: None,
            type_alias_info: None,
        })
    }

    /// Convert a pyrefly function to a TSP `FunctionType` with declaration info.
    ///
    /// For `FunctionKind::Def`, produces a `RegularDeclaration` pointing to the
    /// module where the function is defined. The source range is resolved via
    /// the `resolve_func_range` callback when available; otherwise a zero range
    /// is used.
    fn convert_function(
        &self,
        callable: &Callable,
        kind: &FunctionKind,
        bound_to_type: Option<Box<TspType>>,
    ) -> TspType {
        let ret = self.convert(&callable.ret);
        let declaration = if let FunctionKind::Def(func_id) = kind {
            let module_path = func_id.module.path();
            let uri = path_to_uri(module_path);
            let range = self
                .resolve_func_range
                .and_then(|resolver| resolver(func_id))
                .unwrap_or_default();
            let lsp_range = func_id.module.to_lsp_range(range);
            Declaration::Regular(RegularDeclaration {
                category: DeclarationCategory::Function,
                kind: DeclarationKind::Regular,
                name: Some(func_id.name.to_string()),
                node: Node {
                    range: lsp_range_to_tsp(lsp_range),
                    uri,
                },
            })
        } else {
            Declaration::Synthesized(SynthesizedDeclaration {
                kind: DeclarationKind::Synthesized,
                uri: String::new(),
            })
        };

        TspType::Function(TspFunctionType {
            bound_to_type,
            declaration,
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            kind: TypeKind::Function,
            return_type: Some(Box::new(ret)),
            specialized_types: None,
            type_alias_info: None,
        })
    }

    /// Convert a pyrefly `Overload` to a TSP `OverloadedType`.
    fn convert_overload_to_tsp(
        &self,
        overload: &pyrefly_types::types::Overload,
        bound_to_type: Option<Box<TspType>>,
    ) -> TspType {
        // Wrap in Arc to share across overloads without deep-cloning the Box each time.
        let shared = bound_to_type.map(Arc::from);
        let overloads: Vec<TspType> = overload
            .signatures
            .iter()
            .map(|sig| {
                let bt = shared.as_ref().map(|arc| Box::new(TspType::clone(arc)));
                match sig {
                    pyrefly_types::types::OverloadType::Function(f) => {
                        self.convert_function(&f.signature, &f.metadata.kind, bt)
                    }
                    pyrefly_types::types::OverloadType::Forall(f) => {
                        self.convert_function(&f.body.signature, &f.body.metadata.kind, bt)
                    }
                }
            })
            .collect();
        TspType::Overloaded(TspOverloadedType {
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            implementation: None,
            kind: TypeKind::Overloaded,
            overloads,
            type_alias_info: None,
        })
    }
}

/// Convert a pyrefly `Class` (class definition object) to a TSP `ClassType`
/// with the `Instantiable` flag.
fn convert_class_def(cls: &Class) -> TspType {
    let declaration = make_class_declaration(cls);

    TspType::Class(TspClassType {
        declaration: Declaration::Regular(declaration),
        flags: TypeFlags::INSTANTIABLE,
        id: next_id(),
        kind: TypeKind::Class,
        literal_value: None,
        type_alias_info: None,
        type_args: None,
    })
}

/// Convert a pyrefly `Literal` to a TSP `ClassType` with `literal_value`.
fn convert_literal(lit: &pyrefly_types::literal::Literal) -> TspType {
    match &lit.value {
        Lit::Enum(e) => {
            // For enum literals, use the enum class as the declaration source
            let cls = e.class.class_object();
            let declaration = make_class_declaration(cls);
            TspType::Class(TspClassType {
                declaration: Declaration::Regular(declaration),
                flags: TypeFlags::LITERAL,
                id: next_id(),
                kind: TypeKind::Class,
                literal_value: None,
                type_alias_info: None,
                type_args: None,
            })
        }
        other => {
            let (literal_value, class_name) = match other {
                Lit::Int(i) => (
                    Some(LiteralValue::Int(i.as_i64().unwrap_or(0) as i32)),
                    "int",
                ),
                Lit::Bool(b) => (Some(LiteralValue::Bool(*b)), "bool"),
                Lit::Str(s) => (Some(LiteralValue::String(s.to_string())), "str"),
                Lit::Bytes(_) | Lit::Enum(_) => (None, ""),
            };
            if let Some(lv) = literal_value {
                TspType::Class(TspClassType {
                    declaration: Declaration::Regular(RegularDeclaration {
                        kind: DeclarationKind::Regular,
                        category: DeclarationCategory::Class,
                        name: Some(class_name.to_owned()),
                        node: Node {
                            range: zero_range(),
                            uri: String::new(),
                        },
                    }),
                    flags: TypeFlags::INSTANCE.with_literal(),
                    id: next_id(),
                    kind: TypeKind::Class,
                    literal_value: Some(lv),
                    type_alias_info: None,
                    type_args: None,
                })
            } else {
                // Bytes literal — no direct TSP LiteralValue for bytes
                TspType::Class(TspClassType {
                    declaration: Declaration::Synthesized(SynthesizedDeclaration {
                        kind: DeclarationKind::Synthesized,
                        uri: String::new(),
                    }),
                    flags: TypeFlags::INSTANCE.with_literal(),
                    id: next_id(),
                    kind: TypeKind::Class,
                    literal_value: None,
                    type_alias_info: None,
                    type_args: None,
                })
            }
        }
    }
}

/// Build a `DeclaredType` with `TypeKind::Typevar` from a `QName`.
fn make_typevar_declared(qname: &pyrefly_python::qname::QName) -> DeclaredType {
    let module_path = qname.module_path();
    let uri = path_to_uri(module_path);
    let range = qname.range();
    let lsp_range = qname.module().to_lsp_range(range);

    DeclaredType {
        declaration: Declaration::Regular(RegularDeclaration {
            category: DeclarationCategory::Typeparam,
            kind: DeclarationKind::Regular,
            name: Some(qname.id().to_string()),
            node: Node {
                range: lsp_range_to_tsp(lsp_range),
                uri,
            },
        }),
        flags: TypeFlags::NONE,
        id: next_id(),
        kind: TypeKind::Typevar,
        type_alias_info: None,
    }
}

/// Build a `RegularDeclaration` from a pyrefly `Class`.
fn make_class_declaration(cls: &Class) -> RegularDeclaration {
    let qname = cls.qname();
    let module = qname.module();
    let module_path = qname.module_path();
    let range = qname.range();

    let lsp_range = module.to_lsp_range(range);
    let uri = path_to_uri(module_path);

    RegularDeclaration {
        category: DeclarationCategory::Class,
        kind: DeclarationKind::Regular,
        name: Some(cls.name().to_string()),
        node: Node {
            range: lsp_range_to_tsp(lsp_range),
            uri,
        },
    }
}

/// Convert a `ModulePath` to a URI string, handling bundled typeshed paths.
fn path_to_uri(module_path: &pyrefly_python::module_path::ModulePath) -> String {
    if let Some(real_path) = to_real_path(module_path) {
        Url::from_file_path(&real_path).map_or_else(
            |()| real_path.to_string_lossy().to_string(),
            |u| u.to_string(),
        )
    } else {
        // Fallback for paths that can't be materialized
        module_path.as_path().to_string_lossy().to_string()
    }
}

/// Convert an `lsp_types::Range` to a TSP `Range`.
fn lsp_range_to_tsp(r: lsp_types::Range) -> TspRange {
    TspRange {
        start: TspPosition {
            line: r.start.line,
            character: r.start.character,
        },
        end: TspPosition {
            line: r.end.line,
            character: r.end.character,
        },
    }
}

/// Build a TSP zero-based range (0:0–0:0).
fn zero_range() -> TspRange {
    TspRange {
        start: TspPosition {
            line: 0,
            character: 0,
        },
        end: TspPosition {
            line: 0,
            character: 0,
        },
    }
}

/// Build a TSP `BuiltInType` with the given name.
fn builtin(name: &str) -> TspType {
    TspType::BuiltInType(BuiltInType {
        declaration: None,
        flags: TypeFlags::NONE,
        id: next_id(),
        kind: TypeKind::Builtin,
        name: name.to_owned(),
        possible_type: None,
        type_alias_info: None,
    })
}

#[cfg(test)]
mod tests {
    use pyrefly_types::types::AnyStyle;
    use pyrefly_types::types::NeverStyle;
    use pyrefly_types::types::Type as PyreflyType;
    use tsp_types::SynthesizedType;
    use tsp_types::SynthesizedTypeMetadata;

    use super::*;

    /// Build a TSP `SynthesizedType` whose stub content is the type's display
    /// string. Used only in tests.
    fn synthesized(ty: &PyreflyType) -> TspType {
        let display = ty.to_string();
        TspType::Synthesized(SynthesizedType {
            flags: TypeFlags::INSTANCE,
            id: next_id(),
            kind: TypeKind::Synthesized,
            metadata: SynthesizedTypeMetadata {
                module: TspModuleType {
                    flags: TypeFlags::NONE,
                    id: 0,
                    kind: TypeKind::Module,
                    module_name: String::new(),
                    type_alias_info: None,
                    uri: String::new(),
                },
                primary_definition_offset: 0,
            },
            stub_content: display,
            type_alias_info: None,
        })
    }

    #[test]
    fn test_convert_any() {
        let ty = PyreflyType::Any(AnyStyle::Implicit);
        let tsp = convert_type(&ty);
        match tsp {
            TspType::BuiltInType(b) => {
                assert_eq!(b.name, "any");
                assert_eq!(b.flags, TypeFlags::NONE);
                assert_eq!(b.kind, TypeKind::Builtin);
            }
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_never() {
        let ty = PyreflyType::Never(NeverStyle::Never);
        let tsp = convert_type(&ty);
        match tsp {
            TspType::BuiltInType(b) => {
                assert_eq!(b.name, "never");
            }
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_none() {
        let tsp = convert_type(&PyreflyType::None);
        match tsp {
            TspType::BuiltInType(b) => assert_eq!(b.name, "none"),
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_ellipsis() {
        let tsp = convert_type(&PyreflyType::Ellipsis);
        match tsp {
            TspType::BuiltInType(b) => assert_eq!(b.name, "ellipsis"),
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }

    #[test]
    fn test_unique_ids() {
        let a = convert_type(&PyreflyType::None);
        let b = convert_type(&PyreflyType::Ellipsis);
        let id_a = match &a {
            TspType::BuiltInType(b) => b.id,
            _ => panic!("expected BuiltInType"),
        };
        let id_b = match &b {
            TspType::BuiltInType(b) => b.id,
            _ => panic!("expected BuiltInType"),
        };
        assert_ne!(id_a, id_b, "type ids must be unique");
    }

    #[test]
    fn test_type_flags_bitwise_operations() {
        // Test BitOr
        let combined = TypeFlags::INSTANCE | TypeFlags::CALLABLE;
        assert!(combined.contains(TypeFlags::INSTANCE));
        assert!(combined.contains(TypeFlags::CALLABLE));
        assert!(!combined.contains(TypeFlags::LITERAL));

        // Test BitOrAssign
        let mut flags = TypeFlags::NONE;
        flags |= TypeFlags::INSTANTIABLE;
        assert!(flags.contains(TypeFlags::INSTANTIABLE));
        assert!(!flags.contains(TypeFlags::INSTANCE));

        // Test with_ builders
        let flags = TypeFlags::new().with_instance().with_callable();
        assert!(flags.contains(TypeFlags::INSTANCE));
        assert!(flags.contains(TypeFlags::CALLABLE));
        assert!(!flags.contains(TypeFlags::LITERAL));
    }

    #[test]
    fn test_type_flags_serialization() {
        // INSTANCE = 2
        let json = serde_json::to_value(TypeFlags::INSTANCE).unwrap();
        assert_eq!(json, serde_json::json!(2));

        // CALLABLE = 4
        let json = serde_json::to_value(TypeFlags::CALLABLE).unwrap();
        assert_eq!(json, serde_json::json!(4));

        // Combined flags (INSTANCE | CALLABLE = 6)
        let combined = TypeFlags::INSTANCE | TypeFlags::CALLABLE;
        let json = serde_json::to_value(combined).unwrap();
        assert_eq!(json, serde_json::json!(6));

        // Deserialization
        let flags: TypeFlags = serde_json::from_value(serde_json::json!(6)).unwrap();
        assert!(flags.contains(TypeFlags::INSTANCE));
        assert!(flags.contains(TypeFlags::CALLABLE));
    }

    #[test]
    fn test_synthesized_fallback() {
        let ty = PyreflyType::Ellipsis;
        let tsp = synthesized(&ty);
        match tsp {
            TspType::Synthesized(s) => {
                assert_eq!(s.flags, TypeFlags::INSTANCE);
                assert_eq!(s.kind, TypeKind::Synthesized);
            }
            other => panic!("expected SynthesizedType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_union_of_builtins() {
        // Union[None, Any] should produce a UnionType with 2 sub_types
        let union_ty = PyreflyType::union(vec![
            PyreflyType::None,
            PyreflyType::Any(AnyStyle::Explicit),
        ]);
        let tsp = convert_type(&union_ty);
        match tsp {
            TspType::Union(u) => {
                assert_eq!(u.kind, TypeKind::Union);
                assert_eq!(u.flags, TypeFlags::NONE);
                assert_eq!(u.sub_types.len(), 2);
                // First member should be BuiltIn "none"
                match &u.sub_types[0] {
                    TspType::BuiltInType(b) => assert_eq!(b.name, "none"),
                    other => panic!("expected BuiltInType for first member, got {other:?}"),
                }
                // Second member should be BuiltIn "any"
                match &u.sub_types[1] {
                    TspType::BuiltInType(b) => assert_eq!(b.name, "any"),
                    other => panic!("expected BuiltInType for second member, got {other:?}"),
                }
            }
            other => panic!("expected UnionType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_empty_union() {
        // An empty union is unusual but should not panic
        let union_ty = PyreflyType::union(vec![]);
        let tsp = convert_type(&union_ty);
        match tsp {
            TspType::Union(u) => {
                assert_eq!(u.sub_types.len(), 0);
            }
            other => panic!("expected UnionType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_nested_union() {
        // Union[None, Union[Any, Never]] — inner union is flattened by convert_type
        let inner = PyreflyType::union(vec![
            PyreflyType::Any(AnyStyle::Explicit),
            PyreflyType::Never(NeverStyle::Never),
        ]);
        let outer = PyreflyType::union(vec![PyreflyType::None, inner]);
        let tsp = convert_type(&outer);
        match tsp {
            TspType::Union(u) => {
                assert_eq!(u.sub_types.len(), 2);
                // Second member is a nested Union
                match &u.sub_types[1] {
                    TspType::Union(inner_u) => {
                        assert_eq!(inner_u.sub_types.len(), 2);
                    }
                    other => panic!("expected nested UnionType, got {other:?}"),
                }
            }
            other => panic!("expected UnionType, got {other:?}"),
        }
    }

    #[test]
    fn test_lsp_range_to_tsp_conversion() {
        let lsp_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 10,
                character: 5,
            },
            end: lsp_types::Position {
                line: 10,
                character: 15,
            },
        };
        let tsp_range = lsp_range_to_tsp(lsp_range);
        assert_eq!(tsp_range.start.line, 10);
        assert_eq!(tsp_range.start.character, 5);
        assert_eq!(tsp_range.end.line, 10);
        assert_eq!(tsp_range.end.character, 15);
    }

    #[test]
    fn test_lsp_range_to_tsp_zero_range() {
        let lsp_range = lsp_types::Range {
            start: lsp_types::Position {
                line: 0,
                character: 0,
            },
            end: lsp_types::Position {
                line: 0,
                character: 0,
            },
        };
        let tsp_range = lsp_range_to_tsp(lsp_range);
        assert_eq!(tsp_range.start.line, 0);
        assert_eq!(tsp_range.start.character, 0);
        assert_eq!(tsp_range.end.line, 0);
        assert_eq!(tsp_range.end.character, 0);
    }

    #[test]
    fn test_builtin_json_roundtrip() {
        // Serialize a BuiltInType to JSON and verify the wire format
        let tsp = builtin("any");
        let json = serde_json::to_value(&tsp).unwrap();
        let obj = json.as_object().expect("expected JSON object");
        assert_eq!(
            obj.get("kind").and_then(|v| v.as_u64()),
            Some(TypeKind::Builtin as u64)
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("any"));
        assert!(obj.contains_key("id"));
        assert!(obj.contains_key("flags"));
    }

    #[test]
    fn test_union_json_roundtrip() {
        // Serialize a Union and verify the sub_types array appears in JSON
        let union_ty = PyreflyType::union(vec![PyreflyType::None, PyreflyType::Ellipsis]);
        let tsp = convert_type(&union_ty);
        let json = serde_json::to_value(&tsp).unwrap();
        let obj = json.as_object().expect("expected JSON object");
        assert_eq!(
            obj.get("kind").and_then(|v| v.as_u64()),
            Some(TypeKind::Union as u64)
        );
        let sub_types = obj
            .get("subTypes")
            .and_then(|v| v.as_array())
            .expect("expected subTypes array");
        assert_eq!(sub_types.len(), 2);
    }

    #[test]
    fn test_synthesized_json_roundtrip() {
        // Synthesized type should serialize with stub_content
        let tsp = synthesized(&PyreflyType::Ellipsis);
        let json = serde_json::to_value(&tsp).unwrap();
        let obj = json.as_object().expect("expected JSON object");
        assert_eq!(
            obj.get("kind").and_then(|v| v.as_u64()),
            Some(TypeKind::Synthesized as u64)
        );
        assert!(
            obj.contains_key("stubContent"),
            "expected stubContent field"
        );
        assert!(obj.contains_key("metadata"), "expected metadata field");
    }

    #[test]
    fn test_convert_type_wrapper_non_class() {
        // Type(Any) should pass through since the inner isn't a ClassType
        let ty = PyreflyType::Type(Box::new(PyreflyType::Any(AnyStyle::Explicit)));
        let tsp = convert_type(&ty);
        // Any wrapped in Type() — inner is BuiltIn, not Class, so it passes through unchanged
        match tsp {
            TspType::BuiltInType(b) => assert_eq!(b.name, "any"),
            other => panic!("expected BuiltInType pass-through, got {other:?}"),
        }
    }
}
