#![allow(clippy::all)]
#![allow(dead_code)]

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ****** THIS IS A GENERATED FILE, DO NOT EDIT. ******
// Steps to generate:
// 1. Create tsp.json and tsp.schema.json from typeServerProtocol.ts
// 2. Install lsprotocol generator: `pip install git+https://github.com/microsoft/lsprotocol.git`
// 3. Run: `python generate_protocol.py`

use serde::Deserialize;
use serde::Serialize;

/// This type allows extending any string enum to support custom values.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum CustomStringEnum<T> {
    /// The value is one of the known enum values.
    Known(T),
    /// The value is custom.
    Custom(String),
}

/// This type allows extending any integer enum to support custom values.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum CustomIntEnum<T> {
    /// The value is one of the known enum values.
    Known(T),
    /// The value is custom.
    Custom(i32),
}

/// This allows a field to have two types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR2<T, U> {
    T(T),
    U(U),
}

/// This allows a field to have three types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR3<T, U, V> {
    T(T),
    U(U),
    V(V),
}

/// This allows a field to have four types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR4<T, U, V, W> {
    T(T),
    U(U),
    V(V),
    W(W),
}

/// This allows a field to have five types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR5<T, U, V, W, X> {
    T(T),
    U(U),
    V(V),
    W(W),
    X(X),
}

/// This allows a field to have six types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR6<T, U, V, W, X, Y> {
    T(T),
    U(U),
    V(V),
    W(W),
    X(X),
    Y(Y),
}

/// This allows a field to have seven types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum OR7<T, U, V, W, X, Y, Z> {
    T(T),
    U(U),
    V(V),
    W(W),
    X(X),
    Y(Y),
    Z(Z),
}

/// This allows a field to always have null or empty value.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LSPNull {
    None,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TSPRequestMethods {
    #[serde(rename = "typeServer/getSnapshot")]
    TypeServerGetSnapshot,
    #[serde(rename = "typeServer/getSupportedProtocolVersion")]
    TypeServerGetSupportedProtocolVersion,
    #[serde(rename = "typeServer/getDiagnostics")]
    TypeServerGetDiagnostics,
    #[serde(rename = "typeServer/getDiagnosticsVersion")]
    TypeServerGetDiagnosticsVersion,
    #[serde(rename = "typeServer/getType")]
    TypeServerGetType,
    #[serde(rename = "typeServer/getBuiltinType")]
    TypeServerGetBuiltinType,
    #[serde(rename = "typeServer/getTypeArgs")]
    TypeServerGetTypeArgs,
    #[serde(rename = "typeServer/getSymbolsForType")]
    TypeServerGetSymbolsForType,
    #[serde(rename = "typeServer/getSymbolsForNode")]
    TypeServerGetSymbolsForNode,
    #[serde(rename = "typeServer/getOverloads")]
    TypeServerGetOverloads,
    #[serde(rename = "typeServer/getMatchingOverloads")]
    TypeServerGetMatchingOverloads,
    #[serde(rename = "typeServer/getMetaclass")]
    TypeServerGetMetaclass,
    #[serde(rename = "typeServer/getTypeOfDeclaration")]
    TypeServerGetTypeOfDeclaration,
    #[serde(rename = "typeServer/getRepr")]
    TypeServerGetRepr,
    #[serde(rename = "typeServer/getDocString")]
    TypeServerGetDocstring,
    #[serde(rename = "typeServer/resolveImportDeclaration")]
    TypeServerResolveImportDeclaration,
    #[serde(rename = "typeServer/resolveImport")]
    TypeServerResolveImport,
    #[serde(rename = "typeServer/getTypeAliasInfo")]
    TypeServerGetTypeAliasInfo,
    #[serde(rename = "typeServer/combineTypes")]
    TypeServerCombineTypes,
    #[serde(rename = "typeServer/createInstanceType")]
    TypeServerCreateInstanceType,
    #[serde(rename = "typeServer/getPythonSearchPaths")]
    TypeServerGetPythonSearchPaths,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(tag = "method")]
pub enum TSPRequests {
    #[serde(rename = "typeServer/getSnapshot")]
    GetSnapshotRequest { id: serde_json::Value },
    #[serde(rename = "typeServer/getSupportedProtocolVersion")]
    GetSupportedProtocolVersionRequest { id: serde_json::Value },
    #[serde(rename = "typeServer/getDiagnostics")]
    GetDiagnosticsRequest {
        id: serde_json::Value,
        params: GetDiagnosticsParams,
    },
    #[serde(rename = "typeServer/getDiagnosticsVersion")]
    GetDiagnosticsVersionRequest {
        id: serde_json::Value,
        params: GetDiagnosticsVersionParams,
    },
    #[serde(rename = "typeServer/getType")]
    GetTypeRequest {
        id: serde_json::Value,
        params: GetTypeParams,
    },
    #[serde(rename = "typeServer/getBuiltinType")]
    GetBuiltinTypeRequest {
        id: serde_json::Value,
        params: GetBuiltinTypeParams,
    },
    #[serde(rename = "typeServer/getTypeArgs")]
    GetTypeArgsRequest {
        id: serde_json::Value,
        params: GetTypeArgsParams,
    },
    #[serde(rename = "typeServer/getSymbolsForType")]
    GetSymbolsForTypeRequest {
        id: serde_json::Value,
        params: GetSymbolsForTypeParams,
    },
    #[serde(rename = "typeServer/getSymbolsForNode")]
    GetSymbolsForNodeRequest {
        id: serde_json::Value,
        params: GetSymbolsForNodeParams,
    },
    #[serde(rename = "typeServer/getOverloads")]
    GetOverloadsRequest {
        id: serde_json::Value,
        params: GetOverloadsParams,
    },
    #[serde(rename = "typeServer/getMatchingOverloads")]
    GetMatchingOverloadsRequest {
        id: serde_json::Value,
        params: GetMatchingOverloadsParams,
    },
    #[serde(rename = "typeServer/getMetaclass")]
    GetMetaclassRequest {
        id: serde_json::Value,
        params: GetMetaclassParams,
    },
    #[serde(rename = "typeServer/getTypeOfDeclaration")]
    GetTypeOfDeclarationRequest {
        id: serde_json::Value,
        params: GetTypeOfDeclarationParams,
    },
    #[serde(rename = "typeServer/getRepr")]
    GetReprRequest {
        id: serde_json::Value,
        params: GetReprParams,
    },
    #[serde(rename = "typeServer/getDocString")]
    GetDocstringRequest {
        id: serde_json::Value,
        params: GetDocstringParams,
    },
    #[serde(rename = "typeServer/resolveImportDeclaration")]
    ResolveImportDeclarationRequest {
        id: serde_json::Value,
        params: ResolveImportDeclarationParams,
    },
    #[serde(rename = "typeServer/resolveImport")]
    ResolveImportRequest {
        id: serde_json::Value,
        params: ResolveImportParams,
    },
    #[serde(rename = "typeServer/getTypeAliasInfo")]
    GetTypeAliasInfoRequest {
        id: serde_json::Value,
        params: GetTypeAliasInfoParams,
    },
    #[serde(rename = "typeServer/combineTypes")]
    CombineTypesRequest {
        id: serde_json::Value,
        params: CombineTypesParams,
    },
    #[serde(rename = "typeServer/createInstanceType")]
    CreateInstanceTypeRequest {
        id: serde_json::Value,
        params: CreateInstanceTypeParams,
    },
    #[serde(rename = "typeServer/getPythonSearchPaths")]
    GetPythonSearchPathsRequest {
        id: serde_json::Value,
        params: GetPythonSearchPathsParams,
    },
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TSPNotificationMethods {
    #[serde(rename = "typeServer/snapshotChanged")]
    TypeServerSnapshotChanged,
    #[serde(rename = "typeServer/diagnosticsChanged")]
    TypeServerDiagnosticsChanged,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum MessageDirection {
    #[serde(rename = "clientToServer")]
    ClientToServer,
    #[serde(rename = "serverToClient")]
    ServerToClient,
}

/// Represents a category of a type, such as class, function, variable, etc.
#[derive(PartialEq, Debug, Eq, Clone)]
pub enum TypeCategory {
    /// Type can be anything
    Any = 0,

    /// Callable type
    Function = 1,

    /// Functions defined with @overload decorator
    Overloaded = 2,

    /// Class definition
    Class = 3,

    /// Module instance
    Module = 4,

    /// Union of two or more other types
    Union = 5,

    /// Type variable
    TypeVar = 6,
}
impl Serialize for TypeCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            TypeCategory::Any => serializer.serialize_i32(0),
            TypeCategory::Function => serializer.serialize_i32(1),
            TypeCategory::Overloaded => serializer.serialize_i32(2),
            TypeCategory::Class => serializer.serialize_i32(3),
            TypeCategory::Module => serializer.serialize_i32(4),
            TypeCategory::Union => serializer.serialize_i32(5),
            TypeCategory::TypeVar => serializer.serialize_i32(6),
        }
    }
}
impl<'de> Deserialize<'de> for TypeCategory {
    fn deserialize<D>(deserializer: D) -> Result<TypeCategory, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        match value {
            0 => Ok(TypeCategory::Any),
            1 => Ok(TypeCategory::Function),
            2 => Ok(TypeCategory::Overloaded),
            3 => Ok(TypeCategory::Class),
            4 => Ok(TypeCategory::Module),
            5 => Ok(TypeCategory::Union),
            6 => Ok(TypeCategory::TypeVar),
            _ => Err(serde::de::Error::custom("Unexpected value")),
        }
    }
}

/// Flags that describe the characteristics of a type. These flags can be combined using bitwise operations.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct TypeFlags(pub i32);
impl TypeFlags {
    pub const NONE: TypeFlags = TypeFlags(0);
    pub const INSTANTIABLE: TypeFlags = TypeFlags(1);
    pub const INSTANCE: TypeFlags = TypeFlags(2);
    pub const CALLABLE: TypeFlags = TypeFlags(4);
    pub const LITERAL: TypeFlags = TypeFlags(8);
    pub const INTERFACE: TypeFlags = TypeFlags(16);
    pub const GENERIC: TypeFlags = TypeFlags(32);
    pub const FROM_ALIAS: TypeFlags = TypeFlags(64);
    pub const UNPACKED: TypeFlags = TypeFlags(128);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_instantiable(self) -> Self {
        TypeFlags(self.0 | TypeFlags::INSTANTIABLE.0)
    }
    #[inline]
    pub fn with_instance(self) -> Self {
        TypeFlags(self.0 | TypeFlags::INSTANCE.0)
    }
    #[inline]
    pub fn with_callable(self) -> Self {
        TypeFlags(self.0 | TypeFlags::CALLABLE.0)
    }
    #[inline]
    pub fn with_literal(self) -> Self {
        TypeFlags(self.0 | TypeFlags::LITERAL.0)
    }
    #[inline]
    pub fn with_interface(self) -> Self {
        TypeFlags(self.0 | TypeFlags::INTERFACE.0)
    }
    #[inline]
    pub fn with_generic(self) -> Self {
        TypeFlags(self.0 | TypeFlags::GENERIC.0)
    }
    #[inline]
    pub fn with_from_alias(self) -> Self {
        TypeFlags(self.0 | TypeFlags::FROM_ALIAS.0)
    }
    #[inline]
    pub fn with_unpacked(self) -> Self {
        TypeFlags(self.0 | TypeFlags::UNPACKED.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for TypeFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for TypeFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(TypeFlags(v))
    }
}
impl std::ops::BitOr for TypeFlags {
    type Output = TypeFlags;
    fn bitor(self, rhs: TypeFlags) -> TypeFlags {
        TypeFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for TypeFlags {
    fn bitor_assign(&mut self, rhs: TypeFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for TypeFlags {
    type Output = TypeFlags;
    fn bitand(self, rhs: TypeFlags) -> TypeFlags {
        TypeFlags(self.0 & rhs.0)
    }
}

/// Flags that describe the characteristics of a function or method. These flags can be combined using bitwise operations.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct FunctionFlags(pub i32);
impl FunctionFlags {
    pub const NONE: FunctionFlags = FunctionFlags(0);
    pub const ASYNC: FunctionFlags = FunctionFlags(1);
    pub const GENERATOR: FunctionFlags = FunctionFlags(2);
    pub const ABSTRACT: FunctionFlags = FunctionFlags(4);
    pub const STATIC: FunctionFlags = FunctionFlags(8);
    pub const GRADUAL_CALLABLE_FORM: FunctionFlags = FunctionFlags(16);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_async(self) -> Self {
        FunctionFlags(self.0 | FunctionFlags::ASYNC.0)
    }
    #[inline]
    pub fn with_generator(self) -> Self {
        FunctionFlags(self.0 | FunctionFlags::GENERATOR.0)
    }
    #[inline]
    pub fn with_abstract(self) -> Self {
        FunctionFlags(self.0 | FunctionFlags::ABSTRACT.0)
    }
    #[inline]
    pub fn with_static(self) -> Self {
        FunctionFlags(self.0 | FunctionFlags::STATIC.0)
    }
    #[inline]
    pub fn with_gradual_callable_form(self) -> Self {
        FunctionFlags(self.0 | FunctionFlags::GRADUAL_CALLABLE_FORM.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for FunctionFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for FunctionFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(FunctionFlags(v))
    }
}
impl std::ops::BitOr for FunctionFlags {
    type Output = FunctionFlags;
    fn bitor(self, rhs: FunctionFlags) -> FunctionFlags {
        FunctionFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for FunctionFlags {
    fn bitor_assign(&mut self, rhs: FunctionFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for FunctionFlags {
    type Output = FunctionFlags;
    fn bitand(self, rhs: FunctionFlags) -> FunctionFlags {
        FunctionFlags(self.0 & rhs.0)
    }
}

/// Flags that describe the characteristics of a class. These flags can be combined using bitwise operations.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ClassFlags(pub i32);
impl ClassFlags {
    pub const NONE: ClassFlags = ClassFlags(0);
    pub const ENUM: ClassFlags = ClassFlags(1);
    pub const TYPED_DICT: ClassFlags = ClassFlags(2);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_enum(self) -> Self {
        ClassFlags(self.0 | ClassFlags::ENUM.0)
    }
    #[inline]
    pub fn with_typed_dict(self) -> Self {
        ClassFlags(self.0 | ClassFlags::TYPED_DICT.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for ClassFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for ClassFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(ClassFlags(v))
    }
}
impl std::ops::BitOr for ClassFlags {
    type Output = ClassFlags;
    fn bitor(self, rhs: ClassFlags) -> ClassFlags {
        ClassFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for ClassFlags {
    fn bitor_assign(&mut self, rhs: ClassFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for ClassFlags {
    type Output = ClassFlags;
    fn bitand(self, rhs: ClassFlags) -> ClassFlags {
        ClassFlags(self.0 & rhs.0)
    }
}

/// Flags that describe the characteristics of a type variable. These flags can be combined using bitwise operations.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct TypeVarFlags(pub i32);
impl TypeVarFlags {
    pub const NONE: TypeVarFlags = TypeVarFlags(0);
    pub const IS_PARAM_SPEC: TypeVarFlags = TypeVarFlags(1);
    pub const IS_TYPE_VAR_TUPLE: TypeVarFlags = TypeVarFlags(2);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_is_param_spec(self) -> Self {
        TypeVarFlags(self.0 | TypeVarFlags::IS_PARAM_SPEC.0)
    }
    #[inline]
    pub fn with_is_type_var_tuple(self) -> Self {
        TypeVarFlags(self.0 | TypeVarFlags::IS_TYPE_VAR_TUPLE.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for TypeVarFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for TypeVarFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(TypeVarFlags(v))
    }
}
impl std::ops::BitOr for TypeVarFlags {
    type Output = TypeVarFlags;
    fn bitor(self, rhs: TypeVarFlags) -> TypeVarFlags {
        TypeVarFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for TypeVarFlags {
    fn bitor_assign(&mut self, rhs: TypeVarFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for TypeVarFlags {
    type Output = TypeVarFlags;
    fn bitand(self, rhs: TypeVarFlags) -> TypeVarFlags {
        TypeVarFlags(self.0 & rhs.0)
    }
}

/// Represents the category of a declaration in the type system.
#[derive(PartialEq, Debug, Eq, Clone)]
pub enum DeclarationCategory {
    /// An intrinsic refers to a symbol that has no actual declaration in the source code, such as built-in types or functions.
    Intrinsic = 0,

    /// A variable is a named storage location that can hold a value.
    Variable = 1,

    /// A parameter is a variable that is passed to a function or method.
    Param = 2,

    /// This is for PEP 695 type parameters.
    TypeParam = 3,

    /// This is for PEP 695 type aliases.
    TypeAlias = 4,

    /// A function is any construct that begins with the `def` keyword and has a body, which can be called with arguments.
    Function = 5,

    /// A class is any construct that begins with the `class` keyword and has a body, which can be instantiated.
    Class = 6,

    /// An import declaration, which is a reference to another module.
    Import = 7,
}
impl Serialize for DeclarationCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            DeclarationCategory::Intrinsic => serializer.serialize_i32(0),
            DeclarationCategory::Variable => serializer.serialize_i32(1),
            DeclarationCategory::Param => serializer.serialize_i32(2),
            DeclarationCategory::TypeParam => serializer.serialize_i32(3),
            DeclarationCategory::TypeAlias => serializer.serialize_i32(4),
            DeclarationCategory::Function => serializer.serialize_i32(5),
            DeclarationCategory::Class => serializer.serialize_i32(6),
            DeclarationCategory::Import => serializer.serialize_i32(7),
        }
    }
}
impl<'de> Deserialize<'de> for DeclarationCategory {
    fn deserialize<D>(deserializer: D) -> Result<DeclarationCategory, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        match value {
            0 => Ok(DeclarationCategory::Intrinsic),
            1 => Ok(DeclarationCategory::Variable),
            2 => Ok(DeclarationCategory::Param),
            3 => Ok(DeclarationCategory::TypeParam),
            4 => Ok(DeclarationCategory::TypeAlias),
            5 => Ok(DeclarationCategory::Function),
            6 => Ok(DeclarationCategory::Class),
            7 => Ok(DeclarationCategory::Import),
            _ => Err(serde::de::Error::custom("Unexpected value")),
        }
    }
}

/// Flags that describe extra information about a declaration.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct DeclarationFlags(pub i32);
impl DeclarationFlags {
    pub const NONE: DeclarationFlags = DeclarationFlags(0);
    pub const CLASS_MEMBER: DeclarationFlags = DeclarationFlags(1);
    pub const CONSTANT: DeclarationFlags = DeclarationFlags(2);
    pub const FINAL: DeclarationFlags = DeclarationFlags(4);
    pub const IS_DEFINED_BY_SLOTS: DeclarationFlags = DeclarationFlags(8);
    pub const USES_LOCAL_NAME: DeclarationFlags = DeclarationFlags(16);
    pub const UNRESOLVED_IMPORT: DeclarationFlags = DeclarationFlags(32);
    pub const SIMPLE_PARAM: DeclarationFlags = DeclarationFlags(64);
    pub const ARGS_LIST_PARAM: DeclarationFlags = DeclarationFlags(128);
    pub const KWARGS_DICT_PARAM: DeclarationFlags = DeclarationFlags(256);
    pub const POSITIONAL_PARAM: DeclarationFlags = DeclarationFlags(512);
    pub const STANDARD_PARAM: DeclarationFlags = DeclarationFlags(1024);
    pub const KEYWORD_PARAM: DeclarationFlags = DeclarationFlags(2048);
    pub const EXPANDED_ARGS_PARAM: DeclarationFlags = DeclarationFlags(4096);
    pub const RETURN_TYPE: DeclarationFlags = DeclarationFlags(8192);
    pub const ENUM_MEMBER: DeclarationFlags = DeclarationFlags(16384);
    pub const TYPE_DECLARED: DeclarationFlags = DeclarationFlags(32768);
    pub const SPECIALIZED_TYPE: DeclarationFlags = DeclarationFlags(65536);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_class_member(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::CLASS_MEMBER.0)
    }
    #[inline]
    pub fn with_constant(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::CONSTANT.0)
    }
    #[inline]
    pub fn with_final(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::FINAL.0)
    }
    #[inline]
    pub fn with_is_defined_by_slots(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::IS_DEFINED_BY_SLOTS.0)
    }
    #[inline]
    pub fn with_uses_local_name(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::USES_LOCAL_NAME.0)
    }
    #[inline]
    pub fn with_unresolved_import(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::UNRESOLVED_IMPORT.0)
    }
    #[inline]
    pub fn with_simple_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::SIMPLE_PARAM.0)
    }
    #[inline]
    pub fn with_args_list_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::ARGS_LIST_PARAM.0)
    }
    #[inline]
    pub fn with_kwargs_dict_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::KWARGS_DICT_PARAM.0)
    }
    #[inline]
    pub fn with_positional_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::POSITIONAL_PARAM.0)
    }
    #[inline]
    pub fn with_standard_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::STANDARD_PARAM.0)
    }
    #[inline]
    pub fn with_keyword_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::KEYWORD_PARAM.0)
    }
    #[inline]
    pub fn with_expanded_args_param(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::EXPANDED_ARGS_PARAM.0)
    }
    #[inline]
    pub fn with_return_type(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::RETURN_TYPE.0)
    }
    #[inline]
    pub fn with_enum_member(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::ENUM_MEMBER.0)
    }
    #[inline]
    pub fn with_type_declared(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::TYPE_DECLARED.0)
    }
    #[inline]
    pub fn with_specialized_type(self) -> Self {
        DeclarationFlags(self.0 | DeclarationFlags::SPECIALIZED_TYPE.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for DeclarationFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for DeclarationFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(DeclarationFlags(v))
    }
}
impl std::ops::BitOr for DeclarationFlags {
    type Output = DeclarationFlags;
    fn bitor(self, rhs: DeclarationFlags) -> DeclarationFlags {
        DeclarationFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for DeclarationFlags {
    fn bitor_assign(&mut self, rhs: DeclarationFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for DeclarationFlags {
    type Output = DeclarationFlags;
    fn bitand(self, rhs: DeclarationFlags) -> DeclarationFlags {
        DeclarationFlags(self.0 & rhs.0)
    }
}

/// Flags that are used for searching for symbols.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct SymbolSearchFlags(pub i32);
impl SymbolSearchFlags {
    pub const NONE: SymbolSearchFlags = SymbolSearchFlags(0);
    pub const SKIP_INSTANCE_ATTRIBUTES: SymbolSearchFlags = SymbolSearchFlags(1);
    pub const SKIP_TYPE_BASE_CLASS: SymbolSearchFlags = SymbolSearchFlags(2);
    pub const SKIP_ATTRIBUTE_ACCESS_OVERRIDES: SymbolSearchFlags = SymbolSearchFlags(4);
    pub const GET_BOUND_ATTRIBUTES: SymbolSearchFlags = SymbolSearchFlags(8);
    pub const SKIP_UNREACHABLE_CODE: SymbolSearchFlags = SymbolSearchFlags(16);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_skip_instance_attributes(self) -> Self {
        SymbolSearchFlags(self.0 | SymbolSearchFlags::SKIP_INSTANCE_ATTRIBUTES.0)
    }
    #[inline]
    pub fn with_skip_type_base_class(self) -> Self {
        SymbolSearchFlags(self.0 | SymbolSearchFlags::SKIP_TYPE_BASE_CLASS.0)
    }
    #[inline]
    pub fn with_skip_attribute_access_overrides(self) -> Self {
        SymbolSearchFlags(self.0 | SymbolSearchFlags::SKIP_ATTRIBUTE_ACCESS_OVERRIDES.0)
    }
    #[inline]
    pub fn with_get_bound_attributes(self) -> Self {
        SymbolSearchFlags(self.0 | SymbolSearchFlags::GET_BOUND_ATTRIBUTES.0)
    }
    #[inline]
    pub fn with_skip_unreachable_code(self) -> Self {
        SymbolSearchFlags(self.0 | SymbolSearchFlags::SKIP_UNREACHABLE_CODE.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for SymbolSearchFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for SymbolSearchFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(SymbolSearchFlags(v))
    }
}
impl std::ops::BitOr for SymbolSearchFlags {
    type Output = SymbolSearchFlags;
    fn bitor(self, rhs: SymbolSearchFlags) -> SymbolSearchFlags {
        SymbolSearchFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for SymbolSearchFlags {
    fn bitor_assign(&mut self, rhs: SymbolSearchFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for SymbolSearchFlags {
    type Output = SymbolSearchFlags;
    fn bitand(self, rhs: SymbolSearchFlags) -> SymbolSearchFlags {
        SymbolSearchFlags(self.0 & rhs.0)
    }
}

/// Flags that give more information about a symbol.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct SymbolFlags(pub i32);
impl SymbolFlags {
    pub const NONE: SymbolFlags = SymbolFlags(0);
    pub const SYNTHESIZED_NAME: SymbolFlags = SymbolFlags(1);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_synthesized_name(self) -> Self {
        SymbolFlags(self.0 | SymbolFlags::SYNTHESIZED_NAME.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for SymbolFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for SymbolFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(SymbolFlags(v))
    }
}
impl std::ops::BitOr for SymbolFlags {
    type Output = SymbolFlags;
    fn bitor(self, rhs: SymbolFlags) -> SymbolFlags {
        SymbolFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for SymbolFlags {
    fn bitor_assign(&mut self, rhs: SymbolFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for SymbolFlags {
    type Output = SymbolFlags;
    fn bitand(self, rhs: SymbolFlags) -> SymbolFlags {
        SymbolFlags(self.0 & rhs.0)
    }
}

/// Flags that control how type representations are formatted.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct TypeReprFlags(pub i32);
impl TypeReprFlags {
    pub const NONE: TypeReprFlags = TypeReprFlags(0);
    pub const EXPAND_TYPE_ALIASES: TypeReprFlags = TypeReprFlags(1);
    pub const PRINT_TYPE_VAR_VARIANCE: TypeReprFlags = TypeReprFlags(2);
    pub const CONVERT_TO_INSTANCE_TYPE: TypeReprFlags = TypeReprFlags(4);
    pub const PYTHON_SYNTAX: TypeReprFlags = TypeReprFlags(8);
    #[inline]
    pub fn new() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn with_expand_type_aliases(self) -> Self {
        TypeReprFlags(self.0 | TypeReprFlags::EXPAND_TYPE_ALIASES.0)
    }
    #[inline]
    pub fn with_print_type_var_variance(self) -> Self {
        TypeReprFlags(self.0 | TypeReprFlags::PRINT_TYPE_VAR_VARIANCE.0)
    }
    #[inline]
    pub fn with_convert_to_instance_type(self) -> Self {
        TypeReprFlags(self.0 | TypeReprFlags::CONVERT_TO_INSTANCE_TYPE.0)
    }
    #[inline]
    pub fn with_python_syntax(self) -> Self {
        TypeReprFlags(self.0 | TypeReprFlags::PYTHON_SYNTAX.0)
    }
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}
impl Serialize for TypeReprFlags {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_i32(self.0)
    }
}
impl<'de> Deserialize<'de> for TypeReprFlags {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = i32::deserialize(d)?;
        Ok(TypeReprFlags(v))
    }
}
impl std::ops::BitOr for TypeReprFlags {
    type Output = TypeReprFlags;
    fn bitor(self, rhs: TypeReprFlags) -> TypeReprFlags {
        TypeReprFlags(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for TypeReprFlags {
    fn bitor_assign(&mut self, rhs: TypeReprFlags) {
        self.0 |= rhs.0;
    }
}
impl std::ops::BitAnd for TypeReprFlags {
    type Output = TypeReprFlags;
    fn bitand(self, rhs: TypeReprFlags) -> TypeReprFlags {
        TypeReprFlags(self.0 & rhs.0)
    }
}

/// Unique identifier for a type definition within the snapshot. A handle doesn't need to exist beyond the lifetime of the snapshot.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum TypeHandle {
    String(String),
    Int(i32),
}

/// Unique identifier for a declaration within the session.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum DeclarationHandle {
    String(String),
    Int(i32),
}

/// Position in a text document expressed as zero-based line and character offset.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Position {
    /// Character offset on a line in a document (zero-based).
    pub character: u32,

    /// Line position in a document (zero-based).
    pub line: u32,
}

/// A range in a text document expressed as (zero-based) start and end positions.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Range {
    /// The range's end position.
    pub end: Position,

    /// The range's start position.
    pub start: Position,
}

/// Represents a diagnostic, such as a compiler error or warning.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Diagnostic {
    /// The diagnostic's code.
    pub code: Option<OR2<i32, String>>,

    /// The diagnostic's message.
    pub message: String,

    /// The range at which the message applies.
    pub range: Range,

    /// The diagnostic's severity.
    pub severity: Option<i32>,

    /// A human-readable string describing the source of this diagnostic.
    pub source: Option<String>,
}

/// Represents a node in an AST (Abstract Syntax Tree) or similar structure.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Node {
    /// The range of the node in the source file. This is a zero-based range, meaning the start and end positions are both zero-based. The range uses character offsets the same way the LSP does.
    pub range: Range,

    /// URI of the source file containing this node.
    pub uri: String,
}

/// Represents a module name with optional leading dots for relative imports.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModuleName {
    /// The leading dots in the module name. This is used to determine the relative import level.
    pub leading_dots: i32,

    /// The parts of the module name, split by dots. For example, for `my_module.sub_module`, this would be `['my_module', 'sub_module']`.
    pub name_parts: Vec<String>,
}

/// Represents a type in the type system.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Type {
    /// The typing module defines aliases for builtin types (e.g. Tuple, List, Dict). This field holds the alias name.
    pub alias_name: Option<String>,

    /// Essential classification of the Type.
    pub category: TypeCategory,

    /// Flags specific to the category. For example, for a class type, this would be ClassFlags. For a function type, this would be FunctionFlags.
    pub category_flags: i32,

    /// Declaration of the type, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decl: Option<Declaration>,

    /// Flags describing the type.
    pub flags: TypeFlags,

    /// Unique identifier for the type definition within the snapshot. A handle doesn't need to exist beyond the lifetime of the snapshot.
    pub handle: TypeHandle,

    /// Name of the module the type comes from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_name: Option<ModuleName>,

    /// Simple name of the type. For example, for a class `MyClass` in module `my_module`, this would be `MyClass`.
    pub name: String,
}

/// Represents a symbol declaration in the type system.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Declaration {
    /// Category of this symbol (function, variable, etc.).
    pub category: DeclarationCategory,

    /// Extra information about the declaration.
    pub flags: DeclarationFlags,

    /// Unique identifier for the declaration within the session.
    pub handle: DeclarationHandle,

    /// The dot-separated import name for the file that contains the declaration.
    pub module_name: ModuleName,

    /// The symbol name for the declaration (as the user sees it)
    pub name: String,

    /// Parse node associated with the declaration
    pub node: Option<Node>,

    /// The file that contains the declaration. Unless this is an import declaration, then the uri refers to the file the import is referring to.
    pub uri: String,
}

/// Symbol information for a node
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Symbol {
    /// The declarations for the symbol. This can include multiple declarations for the same symbol, such as when a symbol is defined in multiple files.
    pub decls: Vec<Declaration>,

    /// Flags giving more information about the symbol.
    pub flags: SymbolFlags,

    /// The name of the symbol found.
    pub name: String,

    /// The type that is the semantic parent of this symbol. For example if the symbol is for a parameter, the parent would be the function or method that contains the parameter. If the symbol is for a class member, the parent would be the class that contains the member.
    pub parent: Option<Type>,

    /// The type of the symbol found.
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Options for resolving an import declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportOptions {
    /// Whether to allow access to members that are hidden by external modules.
    pub allow_externally_hidden_access: Option<bool>,

    /// Whether to resolve local names in the import declaration.
    pub resolve_local_names: Option<bool>,

    /// Whether to skip checking if the file is needed for the import resolution.
    pub skip_file_needed_check: Option<bool>,
}

/// Parameters for resolving an import
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportParams {
    /// The descriptor of the imported module.
    pub module_descriptor: ModuleName,

    /// The snapshot version.
    pub snapshot: i32,

    /// The URI of the source file where the import is referenced.
    pub source_uri: String,
}

/// Parameters for getting symbols for a type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSymbolsForTypeParams {
    /// Flags used to do the search
    pub flags: SymbolSearchFlags,

    /// The name to search for. If undefined, returns all the symbols for the type or node.
    pub name: Option<String>,

    /// The location where the symbols are being requested. This can help search for symbols.
    pub node: Option<Node>,

    /// The snapshot version of the type server state.
    pub snapshot: i32,

    /// The type for which the symbols are being requested. If this is a class, the symbols are based on the members of the class. If this is a function, the symbols are the parameters and the return value.
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting symbols for a node.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSymbolsForNodeParams {
    /// Flags used to do the search
    pub flags: SymbolSearchFlags,

    /// The name to search for. If undefined, returns all the symbols for the type or node.
    pub name: Option<String>,

    /// The location to search for symbols from. This node is essentially used to scope the symbol search. It can be a Module node in order to search for top level symbols.
    pub node: Node,

    /// The snapshot version of the type server state.
    pub snapshot: i32,
}

/// Parameters for getting builtin type information.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetBuiltinTypeParams {
    /// The name of the builtin type being requested.
    pub name: String,

    /// The node that is used to scope the builtin type. Every module may have a different set of builtins based on where the module is located.
    pub scoping_node: Node,

    /// The snapshot version of the type server state.
    pub snapshot: i32,
}

/// Information about a type alias.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypeAliasInfo {
    /// The original name of the alias.
    pub name: String,

    /// The arguments for the type alias, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_args: Option<Vec<Type>>,
}

/// Parameters for getting diagnostics for a file.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDiagnosticsParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The URI of the file to get diagnostics for
    pub uri: String,
}

/// Parameters for getting diagnostics version for a file.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDiagnosticsVersionParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The URI of the file
    pub uri: String,
}

/// Parameters for getting type information for a node.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeParams {
    /// The node to get type information for
    pub node: Node,

    /// The snapshot version
    pub snapshot: i32,
}

/// Parameters for getting type arguments.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeArgsParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The type to get arguments for
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting function overloads.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetOverloadsParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The type to get overloads for
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting matching overloads for a call.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetMatchingOverloadsParams {
    /// The call node to get matching overloads for
    pub call_node: Node,

    /// The snapshot version
    pub snapshot: i32,
}

/// Parameters for getting metaclass of a type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetMetaclassParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The type to get metaclass for
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting type of a declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeOfDeclarationParams {
    /// The declaration to get type for
    pub decl: Declaration,

    /// The snapshot version
    pub snapshot: i32,
}

/// Parameters for getting type representation.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetReprParams {
    /// Formatting flags
    pub flags: TypeReprFlags,

    /// The snapshot version
    pub snapshot: i32,

    /// The type to get representation for
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting documentation string.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDocstringParams {
    /// The bound object or class type
    pub bound_object_or_class: Option<Type>,

    /// The declaration to get documentation for
    pub decl: Declaration,

    /// The snapshot version
    pub snapshot: i32,

    /// The type context
    #[serde(rename = "type")]
    pub type_: Option<Type>,
}

/// Parameters for resolving import declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportDeclarationParams {
    /// The import declaration to resolve
    pub decl: Declaration,

    /// Resolution options
    pub options: ResolveImportOptions,

    /// The snapshot version
    pub snapshot: i32,
}

/// Parameters for getting type alias information.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeAliasInfoParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The type to get alias info for
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for combining types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CombineTypesParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The types to combine
    pub types: Vec<Type>,
}

/// Parameters for creating instance type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateInstanceTypeParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The type to create instance from
    #[serde(rename = "type")]
    pub type_: Type,
}

/// Parameters for getting Python search paths.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetPythonSearchPathsParams {
    /// The URI to get search paths from
    pub from_uri: String,

    /// The snapshot version
    pub snapshot: i32,
}

/// Parameters for snapshot changed notification.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotChangedParams {
    /// The new snapshot version
    pub new: i32,

    /// The old snapshot version
    pub old: i32,
}

/// Parameters for diagnostics changed notification.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticsChangedParams {
    /// The snapshot version
    pub snapshot: i32,

    /// The URI of the file with changed diagnostics
    pub uri: String,

    /// The diagnostics version
    pub version: i32,
}

/// Notification sent by the server to indicate any outstanding snapshots are invalid.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotChangedNotification {
    /// The version of the JSON RPC protocol.
    pub jsonrpc: String,

    /// The method to be invoked.
    pub method: TSPNotificationMethods,

    pub params: SnapshotChangedParams,
}

/// Notification sent by the server to indicate that diagnostics have changed and the client should re-request diagnostics for the file.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticsChangedNotification {
    /// The version of the JSON RPC protocol.
    pub jsonrpc: String,

    /// The method to be invoked.
    pub method: TSPNotificationMethods,

    pub params: DiagnosticsChangedParams,
}

/// An identifier to denote a specific request.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LSPId {
    Int(i32),
    String(String),
}

/// An identifier to denote a specific response.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LSPIdOptional {
    Int(i32),
    String(String),
    None,
}

/// Request from client to get the current snapshot of the type server. A snapshot is a point-in-time representation of the type server's state, including all loaded files and their types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSnapshotRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: Option<LSPNull>,
}

/// Response to the [GetSnapshotRequest].
pub type GetSnapshotResponse = i32;

/// Request to get the version of the protocol the type server supports. Returns a string representation of the protocol version (should be semver format).
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSupportedProtocolVersionRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: Option<LSPNull>,
}

/// Response to the [GetSupportedProtocolVersionRequest].
pub type GetSupportedProtocolVersionResponse = String;

/// Request to get diagnostics for a specific file.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDiagnosticsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetDiagnosticsParams,
}

/// Response to the [GetDiagnosticsRequest].
pub type GetDiagnosticsResponse = Vec<Diagnostic>;

/// Request to get the version of diagnostics for a specific file.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDiagnosticsVersionRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetDiagnosticsVersionParams,
}

/// Response to the [GetDiagnosticsVersionRequest].
pub type GetDiagnosticsVersionResponse = i32;

/// Request to get the type information for a specific node.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetTypeParams,
}

/// Response to the [GetTypeRequest].
pub type GetTypeResponse = Type;

/// Request to get the type information for a specific builtin type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetBuiltinTypeRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetBuiltinTypeParams,
}

/// Response to the [GetBuiltinTypeRequest].
pub type GetBuiltinTypeResponse = Type;

/// Request to get the collection of subtypes that make up a union type or the types that makes up a generic type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeArgsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetTypeArgsParams,
}

/// Response to the [GetTypeArgsRequest].
pub type GetTypeArgsResponse = Vec<Type>;

/// Request to find symbols from a type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSymbolsForTypeRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetSymbolsForTypeParams,
}

/// Response to the [GetSymbolsForTypeRequest].
pub type GetSymbolsForTypeResponse = Vec<Symbol>;

/// Request to find symbols from a node.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetSymbolsForNodeRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetSymbolsForNodeParams,
}

/// Response to the [GetSymbolsForNodeRequest].
pub type GetSymbolsForNodeResponse = Vec<Symbol>;

/// Request to get all overloads of a function or method. The returned value doesn't include the implementation signature.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetOverloadsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetOverloadsParams,
}

/// Response to the [GetOverloadsRequest].
pub type GetOverloadsResponse = Vec<Type>;

/// Request to get the overloads that a call node matches.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetMatchingOverloadsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetMatchingOverloadsParams,
}

/// Response to the [GetMatchingOverloadsRequest].
pub type GetMatchingOverloadsResponse = Vec<Type>;

/// Request to get the meta class of a type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetMetaclassRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetMetaclassParams,
}

/// Response to the [GetMetaclassRequest].
pub type GetMetaclassResponse = Type;

/// Request to get the type of a declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeOfDeclarationRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetTypeOfDeclarationParams,
}

/// Response to the [GetTypeOfDeclarationRequest].
pub type GetTypeOfDeclarationResponse = Type;

/// Request to get the string representation of a type in a human-readable format. This may or may not be the same as the type's "name".
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetReprRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetReprParams,
}

/// Response to the [GetReprRequest].
pub type GetReprResponse = String;

/// Request to get the docstring for a specific declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDocstringRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetDocstringParams,
}

/// Response to the [GetDocstringRequest].
pub type GetDocstringResponse = String;

/// Request to resolve an import declaration.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportDeclarationRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: ResolveImportDeclarationParams,
}

/// Response to the [ResolveImportDeclarationRequest].
pub type ResolveImportDeclarationResponse = Declaration;

/// Request to resolve an import. This is used to resolve the import name to its location in the file system.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: ResolveImportParams,
}

/// Response to the [ResolveImportRequest].
pub type ResolveImportResponse = String;

/// Get information about a type alias.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetTypeAliasInfoRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetTypeAliasInfoParams,
}

/// Response to the [GetTypeAliasInfoRequest].
pub type GetTypeAliasInfoResponse = TypeAliasInfo;

/// Request to combine types. This is used to combine multiple types into a single type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CombineTypesRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: CombineTypesParams,
}

/// Response to the [CombineTypesRequest].
pub type CombineTypesResponse = Type;

/// Request to generate an instance type representation for the provided type.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateInstanceTypeRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: CreateInstanceTypeParams,
}

/// Response to the [CreateInstanceTypeRequest].
pub type CreateInstanceTypeResponse = Type;

/// Request to get the search paths that the type server uses for Python modules.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetPythonSearchPathsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetPythonSearchPathsParams,
}

/// Response to the [GetPythonSearchPathsRequest].
pub type GetPythonSearchPathsResponse = Vec<String>;

// Type Server Protocol Constants (idiomatic Rust)

/// The version of the Type Server Protocol
pub const TYPE_SERVER_VERSION: &str = "0.2.0";
/// Represents an invalid handle value
pub const INVALID_HANDLE: i32 = -1;
