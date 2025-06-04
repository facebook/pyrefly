/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @generated
 * Regenerate with glean/schema/gen/Glean/Schema/Gen/Rust.hs
 *  buck2 run glean/schema/gen:gen-schema -- --dir glean/schema/source --rust pyrefly/pyrefly/lib/report/glean
 */

#![allow(warnings)]
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::report::glean::schema::*;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaName {
    pub id: u64,
    pub key: Box<XRefsViaName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaNameByFile {
    pub id: u64,
    pub key: Box<XRefsViaNameByFile_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDeclaration {
    pub id: u64,
    pub key: Box<VariableDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: VariableBySName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Type {
    pub id: u64,
    pub key: Box<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct StringLiteral {
    pub id: u64,
    pub key: Box<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByName {
    pub id: u64,
    pub key: Box<SearchVariableByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchVariableByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByName {
    pub id: u64,
    pub key: Box<SearchModuleByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchModuleByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByName {
    pub id: u64,
    pub key: Box<SearchMethodByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchMethodByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByName {
    pub id: u64,
    pub key: Box<SearchFunctionByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchFunctionByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByName {
    pub id: u64,
    pub key: Box<SearchFieldByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchFieldByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByName {
    pub id: u64,
    pub key: Box<SearchClassByName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchClassByLowerCaseName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SNameToName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: SNameToName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SName {
    pub id: u64,
    pub key: Box<SName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ResolveOriginalName {
    pub id: u64,
    pub key: Box<ResolveOriginalName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct NameToSName {
    pub id: u64,
    pub key: Box<Name>,
    pub value: NameToSName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Name {
    pub id: u64,
    pub key: Box<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModuleDefinition {
    pub id: u64,
    pub key: Box<ModuleDefinition_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModuleBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: ModuleBySName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Module {
    pub id: u64,
    pub key: Box<Module_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverrides {
    pub id: u64,
    pub key: Box<MethodOverrides_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverriden {
    pub id: u64,
    pub key: Box<MethodOverriden_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodByLocalNameStr {
    pub id: u64,
    pub key: Box<MethodByLocalNameStr_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsSName {
    pub id: u64,
    pub key: Box<ImportStatementByAsSName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsName {
    pub id: u64,
    pub key: Box<ImportStatementByAsName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatement {
    pub id: u64,
    pub key: Box<ImportStatement_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarsByFile {
    pub id: u64,
    pub key: Box<ImportStarsByFile_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarStatement {
    pub id: u64,
    pub key: Box<ImportStarStatement_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarLocation {
    pub id: u64,
    pub key: Box<ImportStarLocation_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDeclaration {
    pub id: u64,
    pub key: Box<FunctionDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: FunctionBySName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DerivedClassToBase {
    pub id: u64,
    pub key: Box<DerivedClassToBase_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionLocation {
    pub id: u64,
    pub key: Box<DefinitionLocation_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionsByFile {
    pub id: u64,
    pub key: Box<DefinitionsByFile_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsTopLevelDefinition {
    pub id: u64,
    pub key: Box<Definition>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDefinition {
    pub id: u64,
    pub key: Box<FunctionDefinition_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDefinition {
    pub id: u64,
    pub key: Box<VariableDefinition_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDefinition {
    pub id: u64,
    pub key: Box<DeclarationDefinition_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDocstring {
    pub id: u64,
    pub key: Box<DeclarationDocstring_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationLocation {
    pub id: u64,
    pub key: Box<DeclarationLocation_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationReference {
    pub id: u64,
    pub key: Box<DeclarationReference_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationToName {
    pub id: u64,
    pub key: Box<Declaration>,
    pub value: DeclarationToName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationUses {
    pub id: u64,
    pub key: Box<DeclarationUses_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithLocalName {
    pub id: u64,
    pub key: Box<DeclarationWithLocalName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithName {
    pub id: u64,
    pub key: Box<DeclarationWithName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithSName {
    pub id: u64,
    pub key: Box<DeclarationWithSName_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationsByFile {
    pub id: u64,
    pub key: Box<DeclarationsByFile_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionDeclaration {
    pub id: u64,
    pub key: Box<DefinitionDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DirectXRefsByFile {
    pub id: u64,
    pub key: Box<DirectXRefsByFile_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsAbstract {
    pub id: u64,
    pub key: Box<Declaration>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsTopLevelDeclaration {
    pub id: u64,
    pub key: Box<Declaration>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct NonImportDeclaration {
    pub id: u64,
    pub key: Box<Declaration>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SNameWithDeclaration {
    pub id: u64,
    pub key: Box<SNameWithDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Contains {
    pub id: u64,
    pub key: Box<Contains_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainingTopLevelDeclaration {
    pub id: u64,
    pub key: Box<ContainingTopLevelDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedByTopLevelDeclaration {
    pub id: u64,
    pub key: Box<ContainedByTopLevelDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedBy {
    pub id: u64,
    pub key: Box<ContainedBy_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDefinition {
    pub id: u64,
    pub key: Box<ClassDefinition_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDeclaration {
    pub id: u64,
    pub key: Box<ClassDeclaration_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: ClassBySName_value,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CalleeToCaller {
    pub id: u64,
    pub key: Box<CalleeToCaller_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct BaseClassToDerived {
    pub id: u64,
    pub key: Box<BaseClassToDerived_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileCall {
    pub id: u64,
    pub key: Box<FileCall_key>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaName_key {
    pub xref: Name,
    pub file: src::File,
    pub span: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefViaName {
    pub target: Name,
    pub source: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaNameByFile_key {
    pub file: src::File,
    pub xrefs: Vec<XRefViaName>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDeclaration_key {
    pub name: Name,
}

pub type VariableBySName_value = VariableDeclaration;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TypeInfo {
    pub displayType: Type,
    pub xrefs: Vec<XRefViaName>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: VariableDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: Module,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: FunctionDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: FunctionDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: VariableDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByName_key {
    pub name: String,
    pub parent: Option<SName>,
    pub decl: ClassDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByLowerCaseName_key {
    pub name_lowercase: String,
    pub name: String,
}

pub type SNameToName_value = Name;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SName_key {
    pub local_name: Name,
    pub parent: Option<SName>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ResolveOriginalName_key {
    pub name: Name,
    pub original_name: Name,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Parameter {
    pub name: Name,
    pub typeInfo: Option<TypeInfo>,
    pub value: Option<String>,
}

pub type NameToSName_value = SName;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModuleDefinition_key {
    pub module: Module,
}

pub type ModuleBySName_value = Module;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Module_key {
    pub name: Name,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverrides_key {
    pub derived: FunctionDeclaration,
    pub base: FunctionDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverriden_key {
    pub base: FunctionDeclaration,
    pub derived: FunctionDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodByLocalNameStr_key {
    pub cls: ClassDeclaration,
    pub method_local_name: String,
    pub method: FunctionDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsSName_key {
    pub sname: SName,
    pub import_: ImportStatement,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsName_key {
    pub name: Name,
    pub import_: ImportStatement,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatement_key {
    pub from_name: Name,
    pub as_name: Name,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarsByFile_key {
    pub file: src::File,
    pub span: src::ByteSpan,
    pub declaration: ImportStarStatement,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarStatement_key {
    pub from_name: Name,
    pub into_module: Module,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarLocation_key {
    pub import_star: ImportStarStatement,
    pub file: src::File,
    pub span: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDeclaration_key {
    pub name: Name,
}

pub type FunctionBySName_value = FunctionDeclaration;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DerivedClassToBase_key {
    pub derived: ClassDeclaration,
    pub base: ClassDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Definition {
    cls(ClassDefinition),
    func(FunctionDefinition),
    variable(VariableDefinition),
    module(ModuleDefinition),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionLocation_key {
    pub definition: Definition,
    pub file: src::File,
    pub span: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionsByFile_key {
    pub file: src::File,
    pub span: src::ByteSpan,
    pub definition: Definition,
}

pub type Decorator = String;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum DeclarationContainer {
    module(Module),
    cls(ClassDeclaration),
    func(FunctionDeclaration),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDefinition_key {
    pub declaration: FunctionDeclaration,
    pub is_async: bool,
    pub returnsInfo: Option<TypeInfo>,
    pub params: Vec<Parameter>,
    pub posonly_params: Option<Vec<Parameter>>,
    pub kwonly_params: Option<Vec<Parameter>>,
    pub star_arg: Option<Parameter>,
    pub star_kwarg: Option<Parameter>,
    pub decorators: Option<Vec<Decorator>>,
    pub container: Option<DeclarationContainer>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDefinition_key {
    pub declaration: VariableDeclaration,
    pub typeInfo: Option<TypeInfo>,
    pub container: Option<DeclarationContainer>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Declaration {
    cls(ClassDeclaration),
    func(FunctionDeclaration),
    variable(VariableDeclaration),
    imp(ImportStatement),
    module(Module),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDefinition_key {
    pub declaration: Declaration,
    pub definition: Definition,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDocstring_key {
    pub declaration: Declaration,
    pub location: src::ByteSpan,
    pub pretty_text: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationLocation_key {
    pub declaration: Declaration,
    pub file: src::File,
    pub span: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationReference_key {
    pub target: Declaration,
    pub source: Declaration,
}

pub type DeclarationToName_value = Name;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationUses_key {
    pub declaration: Declaration,
    pub file: src::File,
    pub span: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithLocalName_key {
    pub local_name: Name,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithName_key {
    pub name: Name,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithSName_key {
    pub sname: SName,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationsByFile_key {
    pub file: src::File,
    pub span: src::ByteSpan,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionDeclaration_key {
    pub definition: Definition,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DirectXRef {
    pub target: Declaration,
    pub source: src::ByteSpan,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DirectXRefsByFile_key {
    pub file: src::File,
    pub xref: DirectXRef,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SNameWithDeclaration_key {
    pub declaration: Declaration,
    pub sname: SName,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Contains_key {
    pub container: Declaration,
    pub containee: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainingTopLevelDeclaration_key {
    pub declaration: Declaration,
    pub container: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedByTopLevelDeclaration_key {
    pub container: Declaration,
    pub declaration: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedBy_key {
    pub containee: Declaration,
    pub container: Declaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDefinition_key {
    pub declaration: ClassDeclaration,
    pub bases: Option<Vec<ClassDeclaration>>,
    pub keywords: Option<Vec<Parameter>>,
    pub decorators: Option<Vec<Decorator>>,
    pub container: Option<DeclarationContainer>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDeclaration_key {
    pub name: Name,
    pub bases: Option<Vec<Name>>,
}

pub type ClassBySName_value = ClassDeclaration;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CalleeToCaller_key {
    pub callee: Name,
    pub caller: Name,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct BaseClassToDerived_key {
    pub base: ClassDeclaration,
    pub derived: ClassDeclaration,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum Argument {
    lit(StringLiteral),
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CallArgument {
    pub label: Option<Name>,
    pub span: src::ByteSpan,
    pub argument: Option<Argument>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileCall_key {
    pub file: src::File,
    pub callee_span: src::ByteSpan,
    pub call_args: Vec<CallArgument>,
}
