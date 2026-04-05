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
//!  - `Function` → TSP `SynthesizedType` with stub content.
//!  - `BoundMethod` → TSP `SynthesizedType` with stub content.
//!  - `Literal` → TSP `ClassType` with `literal_value`.
//!  - `Union` → TSP `UnionType` (recursively converting members).
//!  - `Module` → TSP `ModuleType`.
//!  - `Any`, `Never`, `None`, `Ellipsis` → TSP `BuiltInType`.
//!  - Everything else → TSP `SynthesizedType` as a fallback.

use std::collections::HashSet;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use lsp_types::Url;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Params;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType as PyreflyClassType;
use pyrefly_types::literal::Lit;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Type as PyreflyType;
use tsp_types::BuiltInType;
use tsp_types::ClassType as TspClassType;
use tsp_types::Declaration;
use tsp_types::DeclarationCategory;
use tsp_types::DeclarationKind;
use tsp_types::LiteralValue;
use tsp_types::ModuleType as TspModuleType;
use tsp_types::Node;
use tsp_types::OverloadedType as TspOverloadedType;
use tsp_types::Position as TspPosition;
use tsp_types::Range as TspRange;
use tsp_types::RegularDeclaration;
use tsp_types::SynthesizedType;
use tsp_types::SynthesizedTypeMetadata;
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

/// Convert a pyrefly `Type` to a TSP protocol `Type`.
pub fn convert_type(ty: &PyreflyType) -> TspType {
    match ty {
        // --- Built-in special types ---
        PyreflyType::Any(_) => builtin("any"),
        PyreflyType::Never(_) => builtin("never"),
        PyreflyType::None => builtin("none"),
        PyreflyType::Ellipsis => builtin("ellipsis"),

        // --- Class instances (int, str, list[int], user-defined classes, etc.) ---
        PyreflyType::ClassType(ct) => convert_class_type(ct, TypeFlags::INSTANCE),

        // --- Class definitions (the class object itself, e.g. `type[int]`) ---
        PyreflyType::ClassDef(cls) => convert_class_def(cls),

        // --- Literals (Literal[42], Literal["hi"], etc.) ---
        PyreflyType::Literal(lit) => convert_literal(lit),

        // --- Functions ---
        PyreflyType::Function(func) => convert_function(&func.signature, &func.metadata.kind),

        // --- Bound methods ---
        PyreflyType::BoundMethod(bm) => match &bm.func {
            BoundMethodType::Function(f) => convert_function(&f.signature, &f.metadata.kind),
            BoundMethodType::Forall(f) => {
                convert_function(&f.body.signature, &f.body.metadata.kind)
            }
            BoundMethodType::Overload(_) => synthesized(ty),
        },

        // --- Unions ---
        PyreflyType::Union(u) => {
            let sub_types: Vec<TspType> = u.members.iter().map(convert_type).collect();
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
                synthesized(ty)
            }
        }

        // --- Overloaded functions ---
        PyreflyType::Overload(overload) => {
            let overloads: Vec<TspType> = overload
                .signatures
                .iter()
                .map(|sig| match sig {
                    pyrefly_types::types::OverloadType::Function(f) => {
                        convert_function(&f.signature, &f.metadata.kind)
                    }
                    pyrefly_types::types::OverloadType::Forall(f) => {
                        convert_function(&f.body.signature, &f.body.metadata.kind)
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

        // --- Tuples are structurally typed; emit as SynthesizedType ---
        PyreflyType::Tuple(_) => synthesized(ty),

        // --- type[X] wrapper ---
        PyreflyType::Type(inner) => {
            let inner_tsp = convert_type(inner);
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
        PyreflyType::SelfType(ct) => convert_class_type(ct, TypeFlags::INSTANCE),

        // --- Fallback: emit a SynthesizedType with the Display string ---
        _other => synthesized(ty),
    }
}

/// Convert a pyrefly `ClassType` (an instantiated class) to a TSP `ClassType`.
fn convert_class_type(ct: &PyreflyClassType, flags: TypeFlags) -> TspType {
    let cls = ct.class_object();
    let declaration = make_class_declaration(cls);
    let type_args: Option<Vec<TspType>> = {
        let args = ct.targs();
        let slice = args.as_slice();
        if slice.is_empty() {
            None
        } else {
            Some(slice.iter().map(convert_type).collect())
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
                let dummy_node = Node {
                    range: TspRange {
                        start: TspPosition {
                            line: 0,
                            character: 0,
                        },
                        end: TspPosition {
                            line: 0,
                            character: 0,
                        },
                    },
                    uri: String::new(),
                };
                TspType::Class(TspClassType {
                    declaration: Declaration::Regular(RegularDeclaration {
                        kind: DeclarationKind::Regular,
                        category: DeclarationCategory::Class,
                        name: Some(class_name.to_owned()),
                        node: dummy_node,
                    }),
                    flags: TypeFlags::INSTANCE.with_literal(),
                    id: next_id(),
                    kind: TypeKind::Class,
                    literal_value: Some(lv),
                    type_alias_info: None,
                    type_args: None,
                })
            } else {
                // Bytes or other literals without a simple TSP representation
                synthesized(&PyreflyType::Literal(Box::new(lit.clone())))
            }
        }
    }
}

/// Convert a pyrefly function to a TSP `SynthesizedType` with stub content.
///
/// Functions don't carry a TextRange in their FuncId, so we can't point
/// the client at the original source declaration. Instead, we generate a
/// self-contained Python stub that the client's SynthesizedType handler can
/// parse, bind, and evaluate to reconstruct the function type.
fn convert_function(callable: &Callable, kind: &FunctionKind) -> TspType {
    if let FunctionKind::Def(func_id) = kind {
        let func_name = func_id.name.as_str();
        let (stub_content, offset) = generate_function_stub(callable, func_name);

        let module = &func_id.module;
        let module_path = module.path();
        let module_uri = path_to_uri(module_path);
        let module_name = module.name().to_string();

        TspType::Synthesized(SynthesizedType {
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            kind: TypeKind::Synthesized,
            metadata: SynthesizedTypeMetadata {
                module: TspModuleType {
                    flags: TypeFlags::NONE,
                    id: next_id(),
                    kind: TypeKind::Module,
                    module_name,
                    type_alias_info: None,
                    uri: module_uri,
                },
                primary_definition_offset: offset,
            },
            stub_content,
            type_alias_info: None,
        })
    } else {
        // Non-Def function kinds (IsInstance, Cast, etc.)
        TspType::Synthesized(SynthesizedType {
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            kind: TypeKind::Synthesized,
            metadata: empty_synthesized_metadata(),
            stub_content: format!("{kind:?}"),
            type_alias_info: None,
        })
    }
}

/// Collect type names from a pyrefly `Type` that need local `class X: pass`
/// declarations in stub files. Stub `.pyi` files are isolated so ALL
/// referenced type names must be declared locally.
fn collect_type_names(ty: &PyreflyType, names: &mut HashSet<String>) {
    match ty {
        PyreflyType::ClassType(ct) => {
            let name = ct.class_object().name().to_string();
            names.insert(name);
            for arg in ct.targs().as_slice() {
                collect_type_names(arg, names);
            }
        }
        PyreflyType::ClassDef(cls) => {
            let name = cls.name().to_string();
            names.insert(name);
        }
        PyreflyType::Quantified(q) | PyreflyType::QuantifiedValue(q) => {
            let name = q.name.to_string();
            names.insert(name);
        }
        PyreflyType::Union(u) => {
            for member in &u.members {
                collect_type_names(member, names);
            }
        }
        PyreflyType::Type(inner) => {
            collect_type_names(inner, names);
        }
        PyreflyType::Tuple(t) => match t {
            pyrefly_types::tuple::Tuple::Concrete(elts) => {
                for elem in elts {
                    collect_type_names(elem, names);
                }
            }
            pyrefly_types::tuple::Tuple::Unbounded(elem) => {
                collect_type_names(elem.as_ref(), names);
            }
            pyrefly_types::tuple::Tuple::Unpacked(parts) => {
                for elem in &parts.0 {
                    collect_type_names(elem, names);
                }
                collect_type_names(&parts.1, names);
                for elem in &parts.2 {
                    collect_type_names(elem, names);
                }
            }
        },
        _ => {}
    }
}

/// Collect type names from a `Callable` (params + return type) for local
/// stub declarations.
fn collect_callable_type_names(callable: &Callable, names: &mut HashSet<String>) {
    match &callable.params {
        Params::List(params) => {
            for param in params.items() {
                collect_type_names(param.as_type(), names);
            }
        }
        Params::ParamSpec(prefix, _) => {
            for p in prefix.iter() {
                collect_type_names(p.ty(), names);
            }
        }
        Params::Ellipsis | Params::Materialization => {}
    }
    collect_type_names(&callable.ret, names);
}

/// Generate stub content for a function from its `Callable` signature.
///
/// The stub contains:
/// 1. `class X: pass` declarations for each type referenced in the
///    function signature.
/// 2. A `def func_name(params) -> ret: ...` definition.
///
/// Returns `(stub_content, primary_definition_offset)`.
fn generate_function_stub(callable: &Callable, func_name: &str) -> (String, i32) {
    let mut names = HashSet::new();
    collect_callable_type_names(callable, &mut names);

    // Build preamble with class declarations for ALL referenced type names.
    let mut stub = String::new();
    let mut sorted_names: Vec<&String> = names.iter().collect();
    sorted_names.sort(); // deterministic order
    for name in sorted_names {
        stub.push_str(&format!("class {name}:\n    pass\n"));
    }

    // Record offset before function definition
    let offset = stub.len() as i32;

    // Generate function definition using the Callable's Display.
    stub.push_str(&format!("def {func_name}{callable}: ...\n"));

    (stub, offset)
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

/// Build a TSP `SynthesizedType` whose stub content is the type's display
/// string.
fn synthesized(ty: &PyreflyType) -> TspType {
    let display = ty.to_string();
    TspType::Synthesized(SynthesizedType {
        flags: TypeFlags::INSTANCE,
        id: next_id(),
        kind: TypeKind::Synthesized,
        metadata: empty_synthesized_metadata(),
        stub_content: display,
        type_alias_info: None,
    })
}

/// Placeholder metadata for synthesized types.
fn empty_synthesized_metadata() -> SynthesizedTypeMetadata {
    SynthesizedTypeMetadata {
        module: TspModuleType {
            flags: TypeFlags::NONE,
            id: 0,
            kind: TypeKind::Module,
            module_name: String::new(),
            type_alias_info: None,
            uri: String::new(),
        },
        primary_definition_offset: 0,
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_types::types::AnyStyle;
    use pyrefly_types::types::NeverStyle;
    use pyrefly_types::types::Type as PyreflyType;

    use super::*;

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
