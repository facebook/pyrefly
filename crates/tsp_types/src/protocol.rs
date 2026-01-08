#![allow(clippy::all)]
#![allow(dead_code)]

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ****** THIS IS A GENERATED FILE, DO NOT EDIT. ******
// Steps to generate:
// 1. Create tsp.json and tsp.schema.json from typeServerProtocol.ts
// 2. Install lsprotocol generator: `pip install git+https://github.com/microsoft/lsprotocol.git`
// 3. Run: `python generate_protocol.py`

use serde::{Serialize, Deserialize};

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
pub enum TSPRequestMethods{
    #[serde(rename = "typeServer/getComputedType")]
    TypeServerGetComputedType,
    #[serde(rename = "typeServer/getDeclaredType")]
    TypeServerGetDeclaredType,
    #[serde(rename = "typeServer/getExpectedType")]
    TypeServerGetExpectedType,
    #[serde(rename = "typeServer/getPythonSearchPaths")]
    TypeServerGetPythonSearchPaths,
    #[serde(rename = "typeServer/resolveImport")]
    TypeServerResolveImport,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(tag = "method")]pub enum TSPRequests {
    #[serde(rename = "typeServer/getComputedType")]    GetComputedTypeRequest{
        id: serde_json::Value,
        params: None,
    },
    #[serde(rename = "typeServer/getDeclaredType")]    GetDeclaredTypeRequest{
        id: serde_json::Value,
        params: None,
    },
    #[serde(rename = "typeServer/getExpectedType")]    GetExpectedTypeRequest{
        id: serde_json::Value,
        params: None,
    },
    #[serde(rename = "typeServer/getPythonSearchPaths")]    GetPythonSearchPathsRequest{
        id: serde_json::Value,
        params: GetPythonSearchPathsParams,
    },
    #[serde(rename = "typeServer/resolveImport")]    ResolveImportRequest{
        id: serde_json::Value,
        params: ResolveImportParams,
    },
}



#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TSPNotificationMethods{
    #[serde(rename = "typeServer/snapshotChanged")]
    TypeServerSnapshotChanged,
}


#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum MessageDirection{
    #[serde(rename = "clientToServer")]
    ClientToServer,
    #[serde(rename = "serverToClient")]
    ServerToClient,
}


/// Notification sent by the server to indicate any outstanding snapshots are invalid.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotChangedNotification
{
    /// The version of the JSON RPC protocol.
    pub jsonrpc: String,
    
    /// The method to be invoked.
    pub method: TSPNotificationMethods,
    
    pub params: Option<serde_json::Value>,
    
}



/// An identifier to denote a specific request.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LSPId
{
    Int(i32),
    String(String),
}



/// An identifier to denote a specific response.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LSPIdOptional
{
    Int(i32),
    String(String),
    None,
}



/// Requests and notifications for the type server protocol. Request for the computed type of a declaration or node. Computed type is the type that is inferred based on the code flow. Example: def foo(a: int | str): if instanceof(a, int): b = a + 1  # Computed type of 'b' is 'int'
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetComputedTypeRequest
{
    
    /// The method to be invoked.
    pub method: TSPRequestMethods,
    
    /// The request id.
    pub id: LSPId,
    
    pub params: Option<serde_json::Value>,
    
}



/// Response to the [GetComputedTypeRequest].
pub type GetComputedTypeResponse = Type;





/// Request for the declared type of a declaration or node. Declared type is the type that is explicitly declared in the source code. Example: def foo(a: int | str): # Declared type of parameter 'a' is 'int | str' pass
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetDeclaredTypeRequest
{
    
    /// The method to be invoked.
    pub method: TSPRequestMethods,
    
    /// The request id.
    pub id: LSPId,
    
    pub params: Option<serde_json::Value>,
    
}



/// Response to the [GetDeclaredTypeRequest].
pub type GetDeclaredTypeResponse = Type;





/// Request for the expected type of a declaration or node. Expected type is the type that the context expects. Example: def foo(a: int | str): pass foo(4)  # Expected type of argument 'a' is 'int | str'
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetExpectedTypeRequest
{
    
    /// The method to be invoked.
    pub method: TSPRequestMethods,
    
    /// The request id.
    pub id: LSPId,
    
    pub params: Option<serde_json::Value>,
    
}



/// Response to the [GetExpectedTypeRequest].
pub type GetExpectedTypeResponse = Type;





/// Request to get the search paths that the type server uses for Python modules.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetPythonSearchPathsRequest
{
    
    /// The method to be invoked.
    pub method: TSPRequestMethods,
    
    /// The request id.
    pub id: LSPId,
    
    pub params: Option<serde_json::Value>,
    
}



/// Response to the [GetPythonSearchPathsRequest].
pub type GetPythonSearchPathsResponse = Vec<String>;





/// Request to resolve an import. This is used to resolve the import name to its location in the file system.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportRequest
{
    
    /// The method to be invoked.
    pub method: TSPRequestMethods,
    
    /// The request id.
    pub id: LSPId,
    
    pub params: Option<serde_json::Value>,
    
}



/// Response to the [ResolveImportRequest].
pub type ResolveImportResponse = String;




