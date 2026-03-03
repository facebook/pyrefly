/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for the generated protocol types: enums, structs, Type hierarchy,
//! and serialization round-trips.

use tsp_types::*;

// ---------------------------------------------------------------------------
// Enum serialization tests
// ---------------------------------------------------------------------------

#[test]
fn test_type_kind_serialization() {
    let kind = TypeKind::Builtin;
    let json = serde_json::to_value(&kind).unwrap();
    assert_eq!(json, serde_json::json!("BuiltIn"));

    let kind = TypeKind::Function;
    let json = serde_json::to_value(&kind).unwrap();
    assert_eq!(json, serde_json::json!("Function"));
}

#[test]
fn test_type_kind_deserialization() {
    let kind: TypeKind = serde_json::from_str(r#""Class""#).unwrap();
    assert_eq!(kind, TypeKind::Class);

    let kind: TypeKind = serde_json::from_str(r#""Union""#).unwrap();
    assert_eq!(kind, TypeKind::Union);
}

#[test]
fn test_declaration_kind_round_trip() {
    for kind in [DeclarationKind::Regular, DeclarationKind::Synthesized] {
        let json = serde_json::to_string(&kind).unwrap();
        let back: DeclarationKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }
}

#[test]
fn test_declaration_category_serialization() {
    let cat = DeclarationCategory::Function;
    let json = serde_json::to_value(&cat).unwrap();
    assert_eq!(json, serde_json::json!("Function"));
}

#[test]
fn test_type_flags_serialization() {
    let flag = TypeFlags::Callable;
    let json = serde_json::to_value(&flag).unwrap();
    assert_eq!(json, serde_json::json!("Callable"));
}

#[test]
fn test_variance_round_trip() {
    for v in [
        Variance::Auto,
        Variance::Unknown,
        Variance::Invariant,
        Variance::Covariant,
        Variance::Contravariant,
    ] {
        let json = serde_json::to_string(&v).unwrap();
        let back: Variance = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }
}

#[test]
fn test_type_server_version_round_trip() {
    let v = TypeServerVersion::Current;
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!("0.4.0"));
    let back: TypeServerVersion = serde_json::from_value(json).unwrap();
    assert_eq!(back, v);
}

// ---------------------------------------------------------------------------
// Struct construction and serialization tests
// ---------------------------------------------------------------------------

/// Helper: build a minimal Position
fn pos(line: u32, character: u32) -> Position {
    Position { line, character }
}

/// Helper: build a minimal Range
fn range(sl: u32, sc: u32, el: u32, ec: u32) -> Range {
    Range {
        start: pos(sl, sc),
        end: pos(el, ec),
    }
}

/// Helper: build a minimal Node
fn node(uri: &str, sl: u32, sc: u32, el: u32, ec: u32) -> Node {
    Node {
        uri: uri.to_owned(),
        range: range(sl, sc, el, ec),
    }
}

/// Helper: build a RegularDeclaration
fn regular_decl(
    category: DeclarationCategory,
    name: Option<&str>,
    uri: &str,
) -> RegularDeclaration {
    RegularDeclaration {
        kind: "Regular".to_owned(),
        category,
        name: name.map(|s| s.to_owned()),
        node: node(uri, 0, 0, 0, 10),
    }
}

#[test]
fn test_position_serialization() {
    let p = pos(10, 5);
    let json = serde_json::to_value(&p).unwrap();
    assert_eq!(json, serde_json::json!({"line": 10, "character": 5}));

    let back: Position = serde_json::from_value(json).unwrap();
    assert_eq!(back, p);
}

#[test]
fn test_range_serialization() {
    let r = range(1, 0, 1, 20);
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 20},
        })
    );
}

#[test]
fn test_node_serialization() {
    let n = node("file:///test.py", 5, 0, 5, 15);
    let json = serde_json::to_value(&n).unwrap();
    assert_eq!(json["uri"], "file:///test.py");
    assert_eq!(json["range"]["start"]["line"], 5);
}

#[test]
fn test_module_name_serialization() {
    let mn = ModuleName {
        leading_dots: 0,
        name_parts: vec!["os".to_owned(), "path".to_owned()],
    };
    let json = serde_json::to_value(&mn).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "leadingDots": 0,
            "nameParts": ["os", "path"]
        })
    );
    let back: ModuleName = serde_json::from_value(json).unwrap();
    assert_eq!(back, mn);
}

#[test]
fn test_module_name_relative_import() {
    let mn = ModuleName {
        leading_dots: 2,
        name_parts: vec!["utils".to_owned()],
    };
    let json = serde_json::to_value(&mn).unwrap();
    assert_eq!(json["leadingDots"], 2);
}

#[test]
fn test_resolve_import_options_default() {
    let opts = ResolveImportOptions::default();
    assert_eq!(opts.allow_externally_hidden_access, Some(false));
    assert_eq!(opts.resolve_local_names, Some(false));
    assert_eq!(opts.skip_file_needed_check, Some(false));
}

#[test]
fn test_resolve_import_params_serialization() {
    let params = ResolveImportParams {
        module_descriptor: ModuleName {
            leading_dots: 0,
            name_parts: vec!["os".to_owned()],
        },
        source_uri: "file:///src/main.py".to_owned(),
        snapshot: 42,
    };
    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(json["snapshot"], 42);
    assert_eq!(json["sourceUri"], "file:///src/main.py");
    assert_eq!(json["moduleDescriptor"]["nameParts"][0], "os");

    let back: ResolveImportParams = serde_json::from_value(json).unwrap();
    assert_eq!(back, params);
}

// ---------------------------------------------------------------------------
// Declaration hierarchy (flattened base fields)
// ---------------------------------------------------------------------------

#[test]
fn test_regular_declaration_has_base_kind() {
    let decl = regular_decl(DeclarationCategory::Function, Some("foo"), "file:///a.py");
    assert_eq!(decl.kind, "Regular");
    assert_eq!(decl.category, DeclarationCategory::Function);
    assert_eq!(decl.name, Some("foo".to_owned()));
}

#[test]
fn test_regular_declaration_serialization() {
    let decl = regular_decl(DeclarationCategory::Variable, Some("x"), "file:///a.py");
    let json = serde_json::to_value(&decl).unwrap();
    assert_eq!(json["kind"], "Regular");
    assert_eq!(json["category"], "Variable");
    assert_eq!(json["name"], "x");
    assert!(json["node"].is_object());

    let back: RegularDeclaration = serde_json::from_value(json).unwrap();
    assert_eq!(back, decl);
}

#[test]
fn test_synthesized_declaration_has_base_kind() {
    let decl = SynthesizedDeclaration {
        kind: "Synthesized".to_owned(),
        uri: "file:///builtins.pyi".to_owned(),
    };
    assert_eq!(decl.kind, "Synthesized");

    let json = serde_json::to_value(&decl).unwrap();
    assert_eq!(json["kind"], "Synthesized");
    assert_eq!(json["uri"], "file:///builtins.pyi");
}

// ---------------------------------------------------------------------------
// Type hierarchy (BuiltInType with flattened TypeBase fields)
// ---------------------------------------------------------------------------

#[test]
fn test_builtin_type_has_base_fields() {
    let t = BuiltInType {
        id: 1,
        kind: "BuiltIn".to_owned(),
        flags: TypeFlags::Instance,
        type_alias_info: None,
        name: "int".to_owned(),
        declaration: None,
        possible_type: None,
    };
    // TypeBase fields are present
    assert_eq!(t.id, 1);
    assert_eq!(t.kind, "BuiltIn");
    assert_eq!(t.flags, TypeFlags::Instance);
    assert!(t.type_alias_info.is_none());
    // Own fields
    assert_eq!(t.name, "int");
}

#[test]
fn test_builtin_type_serialization_round_trip() {
    let t = BuiltInType {
        id: 42,
        kind: "BuiltIn".to_owned(),
        flags: TypeFlags::None,
        type_alias_info: None,
        name: "unknown".to_owned(),
        declaration: None,
        possible_type: None,
    };
    let json = serde_json::to_value(&t).unwrap();
    assert_eq!(json["id"], 42);
    assert_eq!(json["kind"], "BuiltIn");
    assert_eq!(json["flags"], "None");
    assert_eq!(json["name"], "unknown");

    let back: BuiltInType = serde_json::from_value(json).unwrap();
    assert_eq!(back, t);
}

#[test]
fn test_union_type_serialization() {
    // Build a union of two built-in types
    let int_type = Type::BuiltInType(BuiltInType {
        id: 1,
        kind: "BuiltIn".to_owned(),
        flags: TypeFlags::Instance,
        type_alias_info: None,
        name: "int".to_owned(),
        declaration: None,
        possible_type: None,
    });
    let str_type = Type::BuiltInType(BuiltInType {
        id: 2,
        kind: "BuiltIn".to_owned(),
        flags: TypeFlags::Instance,
        type_alias_info: None,
        name: "str".to_owned(),
        declaration: None,
        possible_type: None,
    });
    let union = UnionType {
        id: 3,
        kind: "Union".to_owned(),
        flags: TypeFlags::Instance,
        type_alias_info: None,
        sub_types: vec![int_type, str_type],
    };
    let json = serde_json::to_value(&union).unwrap();
    assert_eq!(json["kind"], "Union");
    assert_eq!(json["subTypes"].as_array().unwrap().len(), 2);

    let back: UnionType = serde_json::from_value(json).unwrap();
    assert_eq!(back.sub_types.len(), 2);
}

#[test]
fn test_module_type_serialization() {
    let m = ModuleType {
        id: 10,
        kind: "Module".to_owned(),
        flags: TypeFlags::None,
        type_alias_info: None,
        module_name: "os.path".to_owned(),
        uri: "file:///usr/lib/python3.11/posixpath.py".to_owned(),
    };
    let json = serde_json::to_value(&m).unwrap();
    assert_eq!(json["moduleName"], "os.path");
    assert_eq!(json["kind"], "Module");

    let back: ModuleType = serde_json::from_value(json).unwrap();
    assert_eq!(back, m);
}

#[test]
fn test_type_reference_type_serialization() {
    let r = TypeReferenceType {
        id: 99,
        kind: "TypeReference".to_owned(),
        flags: TypeFlags::None,
        type_alias_info: None,
        type_reference_id: 1,
    };
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["typeReferenceId"], 1);

    let back: TypeReferenceType = serde_json::from_value(json).unwrap();
    assert_eq!(back.type_reference_id, 1);
}

// ---------------------------------------------------------------------------
// Request types (beyond GetSnapshot which is already tested)
// ---------------------------------------------------------------------------

#[test]
fn test_resolve_import_request_serialization() {
    let request = ResolveImportRequest {
        method: TSPRequestMethods::TypeServerResolveImport,
        id: LSPId::Int(1),
        params: ResolveImportParams {
            module_descriptor: ModuleName {
                leading_dots: 0,
                name_parts: vec!["os".to_owned(), "path".to_owned()],
            },
            source_uri: "file:///test.py".to_owned(),
            snapshot: 1,
        },
    };
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["method"], "typeServer/resolveImport");
    assert_eq!(json["params"]["moduleDescriptor"]["nameParts"][1], "path");

    let back: ResolveImportRequest = serde_json::from_value(json).unwrap();
    assert_eq!(back.method, TSPRequestMethods::TypeServerResolveImport);
    assert_eq!(back.params.module_descriptor.name_parts.len(), 2);
}

#[test]
fn test_get_python_search_paths_request_serialization() {
    let request = GetPythonSearchPathsRequest {
        method: TSPRequestMethods::TypeServerGetPythonSearchPaths,
        id: LSPId::String("req-42".to_owned()),
        params: GetPythonSearchPathsParams {
            from_uri: "file:///project".to_owned(),
            snapshot: 5,
        },
    };
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["method"], "typeServer/getPythonSearchPaths");
    assert_eq!(json["params"]["fromUri"], "file:///project");
    assert_eq!(json["params"]["snapshot"], 5);
    assert_eq!(json["id"], "req-42");
}

#[test]
fn test_get_supported_protocol_version_request() {
    let request = GetSupportedProtocolVersionRequest {
        method: TSPRequestMethods::TypeServerGetSupportedProtocolVersion,
        id: LSPId::Int(1),
        params: None,
    };
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["method"], "typeServer/getSupportedProtocolVersion");
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

#[test]
fn test_snapshot_changed_notification_serialization() {
    let notif = SnapshotChangedNotification {
        jsonrpc: "2.0".to_owned(),
        method: TSPNotificationMethods::TypeServerSnapshotChanged,
        params: None,
    };
    let json = serde_json::to_value(&notif).unwrap();
    assert_eq!(json["jsonrpc"], "2.0");
    assert_eq!(json["method"], "typeServer/snapshotChanged");
}

// ---------------------------------------------------------------------------
// LiteralValue variants
// ---------------------------------------------------------------------------

#[test]
fn test_literal_value_int() {
    let v = LiteralValue::Int(42);
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!(42));
}

#[test]
fn test_literal_value_string() {
    let v = LiteralValue::String("hello".to_owned());
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!("hello"));
}

#[test]
fn test_literal_value_bool() {
    let v = LiteralValue::Bool(true);
    let json = serde_json::to_value(&v).unwrap();
    assert_eq!(json, serde_json::json!(true));
}

#[test]
fn test_enum_literal_serialization() {
    let int_type = Type::BuiltInType(BuiltInType {
        id: 1,
        kind: "BuiltIn".to_owned(),
        flags: TypeFlags::Instance,
        type_alias_info: None,
        name: "int".to_owned(),
        declaration: None,
        possible_type: None,
    });
    let e = EnumLiteral {
        class_name: "Color".to_owned(),
        item_name: "RED".to_owned(),
        item_type: Box::new(int_type),
    };
    let json = serde_json::to_value(&e).unwrap();
    assert_eq!(json["className"], "Color");
    assert_eq!(json["itemName"], "RED");
}

// ---------------------------------------------------------------------------
// TSPRequests discriminated union
// ---------------------------------------------------------------------------

#[test]
fn test_tsp_requests_enum_deserialization() {
    // GetSnapshotRequest has no params
    let json = serde_json::json!({
        "method": "typeServer/getSnapshot",
        "id": 1
    });
    let req: TSPRequests = serde_json::from_value(json).unwrap();
    match req {
        TSPRequests::GetSnapshotRequest { id } => {
            assert_eq!(id, serde_json::json!(1));
        }
        _ => panic!("Expected GetSnapshotRequest variant"),
    }
}

#[test]
fn test_tsp_requests_enum_resolve_import() {
    let json = serde_json::json!({
        "method": "typeServer/resolveImport",
        "id": 2,
        "params": {
            "moduleDescriptor": {
                "leadingDots": 0,
                "nameParts": ["os"]
            },
            "sourceUri": "file:///main.py",
            "snapshot": 1
        }
    });
    let req: TSPRequests = serde_json::from_value(json).unwrap();
    match req {
        TSPRequests::ResolveImportRequest { id, params } => {
            assert_eq!(id, serde_json::json!(2));
            assert_eq!(params.module_descriptor.name_parts, vec!["os"]);
        }
        _ => panic!("Expected ResolveImportRequest variant"),
    }
}
