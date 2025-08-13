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
use serde_repr::*;

use crate::report::glean::schema::*;

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaNameByTarget {
    pub id: u64,
    pub key: Box<XRefsViaNameByTarget_key>,
}

impl XRefsViaNameByTarget {
    pub fn GLEAN_name() -> String {
        String::from("python.XRefsViaNameByTarget.4")
    }

    pub fn new(target: Name, file: src::File, spans: Vec<src::ByteSpan>) -> Self {
        XRefsViaNameByTarget {
            id: 0,
            key: Box::new(XRefsViaNameByTarget_key {
                target,
                file,
                spans
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaNameByFile {
    pub id: u64,
    pub key: Box<XRefsViaNameByFile_key>,
}

impl XRefsViaNameByFile {
    pub fn GLEAN_name() -> String {
        String::from("python.XRefsViaNameByFile.4")
    }

    pub fn new(file: src::File, xrefs: Vec<XRefViaName>) -> Self {
        XRefsViaNameByFile {
            id: 0,
            key: Box::new(XRefsViaNameByFile_key {
                file,
                xrefs
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDeclaration {
    pub id: u64,
    pub key: Box<VariableDeclaration_key>,
}

impl VariableDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.VariableDeclaration.4")
    }

    pub fn new(name: Name) -> Self {
        VariableDeclaration {
            id: 0,
            key: Box::new(VariableDeclaration_key {
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: VariableBySName_value,
}

impl VariableBySName {
    pub fn GLEAN_name() -> String {
        String::from("python.VariableBySName.4")
    }

    pub fn new(key: SName, value: VariableBySName_value) -> Self {
        VariableBySName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Type {
    pub id: u64,
    pub key: Box<String>,
}

impl Type {
    pub fn GLEAN_name() -> String {
        String::from("python.Type.4")
    }

    pub fn new(key: String) -> Self {
        Type {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct StringLiteral {
    pub id: u64,
    pub key: Box<String>,
}

impl StringLiteral {
    pub fn GLEAN_name() -> String {
        String::from("python.StringLiteral.4")
    }

    pub fn new(key: String) -> Self {
        StringLiteral {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByName {
    pub id: u64,
    pub key: Box<SearchVariableByName_key>,
}

impl SearchVariableByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchVariableByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: VariableDeclaration) -> Self {
        SearchVariableByName {
            id: 0,
            key: Box::new(SearchVariableByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchVariableByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchVariableByLowerCaseName_key>,
}

impl SearchVariableByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchVariableByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchVariableByLowerCaseName {
            id: 0,
            key: Box::new(SearchVariableByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByName {
    pub id: u64,
    pub key: Box<SearchModuleByName_key>,
}

impl SearchModuleByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchModuleByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: Module) -> Self {
        SearchModuleByName {
            id: 0,
            key: Box::new(SearchModuleByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchModuleByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchModuleByLowerCaseName_key>,
}

impl SearchModuleByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchModuleByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchModuleByLowerCaseName {
            id: 0,
            key: Box::new(SearchModuleByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByName {
    pub id: u64,
    pub key: Box<SearchMethodByName_key>,
}

impl SearchMethodByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchMethodByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: FunctionDeclaration) -> Self {
        SearchMethodByName {
            id: 0,
            key: Box::new(SearchMethodByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchMethodByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchMethodByLowerCaseName_key>,
}

impl SearchMethodByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchMethodByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchMethodByLowerCaseName {
            id: 0,
            key: Box::new(SearchMethodByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByName {
    pub id: u64,
    pub key: Box<SearchFunctionByName_key>,
}

impl SearchFunctionByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchFunctionByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: FunctionDeclaration) -> Self {
        SearchFunctionByName {
            id: 0,
            key: Box::new(SearchFunctionByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFunctionByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchFunctionByLowerCaseName_key>,
}

impl SearchFunctionByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchFunctionByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchFunctionByLowerCaseName {
            id: 0,
            key: Box::new(SearchFunctionByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByName {
    pub id: u64,
    pub key: Box<SearchFieldByName_key>,
}

impl SearchFieldByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchFieldByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: VariableDeclaration) -> Self {
        SearchFieldByName {
            id: 0,
            key: Box::new(SearchFieldByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchFieldByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchFieldByLowerCaseName_key>,
}

impl SearchFieldByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchFieldByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchFieldByLowerCaseName {
            id: 0,
            key: Box::new(SearchFieldByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByName {
    pub id: u64,
    pub key: Box<SearchClassByName_key>,
}

impl SearchClassByName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchClassByName.4")
    }

    pub fn new(name: String, parent: Option<SName>, decl: ClassDeclaration) -> Self {
        SearchClassByName {
            id: 0,
            key: Box::new(SearchClassByName_key {
                name,
                parent,
                decl
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SearchClassByLowerCaseName {
    pub id: u64,
    pub key: Box<SearchClassByLowerCaseName_key>,
}

impl SearchClassByLowerCaseName {
    pub fn GLEAN_name() -> String {
        String::from("python.SearchClassByLowerCaseName.4")
    }

    pub fn new(name_lowercase: String, name: String) -> Self {
        SearchClassByLowerCaseName {
            id: 0,
            key: Box::new(SearchClassByLowerCaseName_key {
                name_lowercase,
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SNameToName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: SNameToName_value,
}

impl SNameToName {
    pub fn GLEAN_name() -> String {
        String::from("python.SNameToName.4")
    }

    pub fn new(key: SName, value: SNameToName_value) -> Self {
        SNameToName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SName {
    pub id: u64,
    pub key: Box<SName_key>,
}

impl SName {
    pub fn GLEAN_name() -> String {
        String::from("python.SName.4")
    }

    pub fn new(local_name: Name, parent: Option<SName>) -> Self {
        SName {
            id: 0,
            key: Box::new(SName_key {
                local_name,
                parent
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ResolveOriginalName {
    pub id: u64,
    pub key: Box<ResolveOriginalName_key>,
}

impl ResolveOriginalName {
    pub fn GLEAN_name() -> String {
        String::from("python.ResolveOriginalName.4")
    }

    pub fn new(name: Name, original_name: Name) -> Self {
        ResolveOriginalName {
            id: 0,
            key: Box::new(ResolveOriginalName_key {
                name,
                original_name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct NameToSName {
    pub id: u64,
    pub key: Box<Name>,
    pub value: NameToSName_value,
}

impl NameToSName {
    pub fn GLEAN_name() -> String {
        String::from("python.NameToSName.4")
    }

    pub fn new(key: Name, value: NameToSName_value) -> Self {
        NameToSName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Name {
    pub id: u64,
    pub key: Box<String>,
}

impl Name {
    pub fn GLEAN_name() -> String {
        String::from("python.Name.4")
    }

    pub fn new(key: String) -> Self {
        Name {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModuleDefinition {
    pub id: u64,
    pub key: Box<ModuleDefinition_key>,
}

impl ModuleDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.ModuleDefinition.4")
    }

    pub fn new(module: Module) -> Self {
        ModuleDefinition {
            id: 0,
            key: Box::new(ModuleDefinition_key {
                module
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModuleBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: ModuleBySName_value,
}

impl ModuleBySName {
    pub fn GLEAN_name() -> String {
        String::from("python.ModuleBySName.4")
    }

    pub fn new(key: SName, value: ModuleBySName_value) -> Self {
        ModuleBySName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Module {
    pub id: u64,
    pub key: Box<Module_key>,
}

impl Module {
    pub fn GLEAN_name() -> String {
        String::from("python.Module.4")
    }

    pub fn new(name: Name) -> Self {
        Module {
            id: 0,
            key: Box::new(Module_key {
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverrides {
    pub id: u64,
    pub key: Box<MethodOverrides_key>,
}

impl MethodOverrides {
    pub fn GLEAN_name() -> String {
        String::from("python.MethodOverrides.4")
    }

    pub fn new(derived: FunctionDeclaration, base: FunctionDeclaration) -> Self {
        MethodOverrides {
            id: 0,
            key: Box::new(MethodOverrides_key {
                derived,
                base
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodOverridden {
    pub id: u64,
    pub key: Box<MethodOverridden_key>,
}

impl MethodOverridden {
    pub fn GLEAN_name() -> String {
        String::from("python.MethodOverridden.4")
    }

    pub fn new(base: FunctionDeclaration, derived: FunctionDeclaration) -> Self {
        MethodOverridden {
            id: 0,
            key: Box::new(MethodOverridden_key {
                base,
                derived
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct MethodByLocalNameStr {
    pub id: u64,
    pub key: Box<MethodByLocalNameStr_key>,
}

impl MethodByLocalNameStr {
    pub fn GLEAN_name() -> String {
        String::from("python.MethodByLocalNameStr.4")
    }

    pub fn new(cls: ClassDeclaration, method_local_name: String, method: FunctionDeclaration) -> Self {
        MethodByLocalNameStr {
            id: 0,
            key: Box::new(MethodByLocalNameStr_key {
                cls,
                method_local_name,
                method
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsSName {
    pub id: u64,
    pub key: Box<ImportStatementByAsSName_key>,
}

impl ImportStatementByAsSName {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStatementByAsSName.4")
    }

    pub fn new(sname: SName, import_: ImportStatement) -> Self {
        ImportStatementByAsSName {
            id: 0,
            key: Box::new(ImportStatementByAsSName_key {
                sname,
                import_
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatementByAsName {
    pub id: u64,
    pub key: Box<ImportStatementByAsName_key>,
}

impl ImportStatementByAsName {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStatementByAsName.4")
    }

    pub fn new(name: Name, import_: ImportStatement) -> Self {
        ImportStatementByAsName {
            id: 0,
            key: Box::new(ImportStatementByAsName_key {
                name,
                import_
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStatement {
    pub id: u64,
    pub key: Box<ImportStatement_key>,
}

impl ImportStatement {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStatement.4")
    }

    pub fn new(from_name: Name, as_name: Name) -> Self {
        ImportStatement {
            id: 0,
            key: Box::new(ImportStatement_key {
                from_name,
                as_name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarsByFile {
    pub id: u64,
    pub key: Box<ImportStarsByFile_key>,
}

impl ImportStarsByFile {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStarsByFile.4")
    }

    pub fn new(file: src::File, span: src::ByteSpan, declaration: ImportStarStatement) -> Self {
        ImportStarsByFile {
            id: 0,
            key: Box::new(ImportStarsByFile_key {
                file,
                span,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarStatement {
    pub id: u64,
    pub key: Box<ImportStarStatement_key>,
}

impl ImportStarStatement {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStarStatement.4")
    }

    pub fn new(from_name: Name, into_module: Module) -> Self {
        ImportStarStatement {
            id: 0,
            key: Box::new(ImportStarStatement_key {
                from_name,
                into_module
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImportStarLocation {
    pub id: u64,
    pub key: Box<ImportStarLocation_key>,
}

impl ImportStarLocation {
    pub fn GLEAN_name() -> String {
        String::from("python.ImportStarLocation.4")
    }

    pub fn new(import_star: ImportStarStatement, file: src::File, span: src::ByteSpan) -> Self {
        ImportStarLocation {
            id: 0,
            key: Box::new(ImportStarLocation_key {
                import_star,
                file,
                span
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDeclaration {
    pub id: u64,
    pub key: Box<FunctionDeclaration_key>,
}

impl FunctionDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.FunctionDeclaration.4")
    }

    pub fn new(name: Name) -> Self {
        FunctionDeclaration {
            id: 0,
            key: Box::new(FunctionDeclaration_key {
                name
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: FunctionBySName_value,
}

impl FunctionBySName {
    pub fn GLEAN_name() -> String {
        String::from("python.FunctionBySName.4")
    }

    pub fn new(key: SName, value: FunctionBySName_value) -> Self {
        FunctionBySName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DerivedClassToBase {
    pub id: u64,
    pub key: Box<DerivedClassToBase_key>,
}

impl DerivedClassToBase {
    pub fn GLEAN_name() -> String {
        String::from("python.DerivedClassToBase.4")
    }

    pub fn new(derived: ClassDeclaration, base: ClassDeclaration) -> Self {
        DerivedClassToBase {
            id: 0,
            key: Box::new(DerivedClassToBase_key {
                derived,
                base
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionLocation {
    pub id: u64,
    pub key: Box<DefinitionLocation_key>,
}

impl DefinitionLocation {
    pub fn GLEAN_name() -> String {
        String::from("python.DefinitionLocation.4")
    }

    pub fn new(definition: Definition, file: src::File, span: src::ByteSpan) -> Self {
        DefinitionLocation {
            id: 0,
            key: Box::new(DefinitionLocation_key {
                definition,
                file,
                span
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionsByFile {
    pub id: u64,
    pub key: Box<DefinitionsByFile_key>,
}

impl DefinitionsByFile {
    pub fn GLEAN_name() -> String {
        String::from("python.DefinitionsByFile.4")
    }

    pub fn new(file: src::File, span: src::ByteSpan, definition: Definition) -> Self {
        DefinitionsByFile {
            id: 0,
            key: Box::new(DefinitionsByFile_key {
                file,
                span,
                definition
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsTopLevelDefinition {
    pub id: u64,
    pub key: Box<Definition>,
}

impl IsTopLevelDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.IsTopLevelDefinition.4")
    }

    pub fn new(key: Definition) -> Self {
        IsTopLevelDefinition {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FunctionDefinition {
    pub id: u64,
    pub key: Box<FunctionDefinition_key>,
}

impl FunctionDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.FunctionDefinition.4")
    }

    pub fn new(declaration: FunctionDeclaration, is_async: bool, returnsInfo: Option<TypeInfo>, params: Vec<Parameter>, posonly_params: Option<Vec<Parameter>>, kwonly_params: Option<Vec<Parameter>>, star_arg: Option<Parameter>, star_kwarg: Option<Parameter>, decorators: Option<Vec<Decorator>>, container: Option<DeclarationContainer>) -> Self {
        FunctionDefinition {
            id: 0,
            key: Box::new(FunctionDefinition_key {
                declaration,
                is_async,
                returnsInfo,
                params,
                posonly_params,
                kwonly_params,
                star_arg,
                star_kwarg,
                decorators,
                container
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDefinition {
    pub id: u64,
    pub key: Box<VariableDefinition_key>,
}

impl VariableDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.VariableDefinition.4")
    }

    pub fn new(declaration: VariableDeclaration, typeInfo: Option<TypeInfo>, container: Option<DeclarationContainer>) -> Self {
        VariableDefinition {
            id: 0,
            key: Box::new(VariableDefinition_key {
                declaration,
                typeInfo,
                container
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDefinition {
    pub id: u64,
    pub key: Box<DeclarationDefinition_key>,
}

impl DeclarationDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationDefinition.4")
    }

    pub fn new(declaration: Declaration, definition: Definition) -> Self {
        DeclarationDefinition {
            id: 0,
            key: Box::new(DeclarationDefinition_key {
                declaration,
                definition
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationDocstring {
    pub id: u64,
    pub key: Box<DeclarationDocstring_key>,
}

impl DeclarationDocstring {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationDocstring.4")
    }

    pub fn new(declaration: Declaration, location: src::ByteSpan, pretty_text: String) -> Self {
        DeclarationDocstring {
            id: 0,
            key: Box::new(DeclarationDocstring_key {
                declaration,
                location,
                pretty_text
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationLocation {
    pub id: u64,
    pub key: Box<DeclarationLocation_key>,
}

impl DeclarationLocation {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationLocation.4")
    }

    pub fn new(declaration: Declaration, file: src::File, span: src::ByteSpan) -> Self {
        DeclarationLocation {
            id: 0,
            key: Box::new(DeclarationLocation_key {
                declaration,
                file,
                span
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationReference {
    pub id: u64,
    pub key: Box<DeclarationReference_key>,
}

impl DeclarationReference {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationReference.4")
    }

    pub fn new(target: Declaration, source: Declaration) -> Self {
        DeclarationReference {
            id: 0,
            key: Box::new(DeclarationReference_key {
                target,
                source
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationToName {
    pub id: u64,
    pub key: Box<Declaration>,
    pub value: DeclarationToName_value,
}

impl DeclarationToName {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationToName.4")
    }

    pub fn new(key: Declaration, value: DeclarationToName_value) -> Self {
        DeclarationToName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationUses {
    pub id: u64,
    pub key: Box<DeclarationUses_key>,
}

impl DeclarationUses {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationUses.4")
    }

    pub fn new(declaration: Declaration, file: src::File, span: src::ByteSpan) -> Self {
        DeclarationUses {
            id: 0,
            key: Box::new(DeclarationUses_key {
                declaration,
                file,
                span
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithLocalName {
    pub id: u64,
    pub key: Box<DeclarationWithLocalName_key>,
}

impl DeclarationWithLocalName {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationWithLocalName.4")
    }

    pub fn new(local_name: Name, declaration: Declaration) -> Self {
        DeclarationWithLocalName {
            id: 0,
            key: Box::new(DeclarationWithLocalName_key {
                local_name,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithName {
    pub id: u64,
    pub key: Box<DeclarationWithName_key>,
}

impl DeclarationWithName {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationWithName.4")
    }

    pub fn new(name: Name, declaration: Declaration) -> Self {
        DeclarationWithName {
            id: 0,
            key: Box::new(DeclarationWithName_key {
                name,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationWithSName {
    pub id: u64,
    pub key: Box<DeclarationWithSName_key>,
}

impl DeclarationWithSName {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationWithSName.4")
    }

    pub fn new(sname: SName, declaration: Declaration) -> Self {
        DeclarationWithSName {
            id: 0,
            key: Box::new(DeclarationWithSName_key {
                sname,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeclarationsByFile {
    pub id: u64,
    pub key: Box<DeclarationsByFile_key>,
}

impl DeclarationsByFile {
    pub fn GLEAN_name() -> String {
        String::from("python.DeclarationsByFile.4")
    }

    pub fn new(file: src::File, span: src::ByteSpan, declaration: Declaration) -> Self {
        DeclarationsByFile {
            id: 0,
            key: Box::new(DeclarationsByFile_key {
                file,
                span,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DefinitionDeclaration {
    pub id: u64,
    pub key: Box<DefinitionDeclaration_key>,
}

impl DefinitionDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.DefinitionDeclaration.4")
    }

    pub fn new(definition: Definition, declaration: Declaration) -> Self {
        DefinitionDeclaration {
            id: 0,
            key: Box::new(DefinitionDeclaration_key {
                definition,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DirectXRefsByFile {
    pub id: u64,
    pub key: Box<DirectXRefsByFile_key>,
}

impl DirectXRefsByFile {
    pub fn GLEAN_name() -> String {
        String::from("python.DirectXRefsByFile.4")
    }

    pub fn new(file: src::File, xref: DirectXRef) -> Self {
        DirectXRefsByFile {
            id: 0,
            key: Box::new(DirectXRefsByFile_key {
                file,
                xref
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsAbstract {
    pub id: u64,
    pub key: Box<Declaration>,
}

impl IsAbstract {
    pub fn GLEAN_name() -> String {
        String::from("python.IsAbstract.4")
    }

    pub fn new(key: Declaration) -> Self {
        IsAbstract {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IsTopLevelDeclaration {
    pub id: u64,
    pub key: Box<Declaration>,
}

impl IsTopLevelDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.IsTopLevelDeclaration.4")
    }

    pub fn new(key: Declaration) -> Self {
        IsTopLevelDeclaration {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct NonImportDeclaration {
    pub id: u64,
    pub key: Box<Declaration>,
}

impl NonImportDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.NonImportDeclaration.4")
    }

    pub fn new(key: Declaration) -> Self {
        NonImportDeclaration {
            id: 0,
            key: Box::new(key),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SNameWithDeclaration {
    pub id: u64,
    pub key: Box<SNameWithDeclaration_key>,
}

impl SNameWithDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.SNameWithDeclaration.4")
    }

    pub fn new(declaration: Declaration, sname: SName) -> Self {
        SNameWithDeclaration {
            id: 0,
            key: Box::new(SNameWithDeclaration_key {
                declaration,
                sname
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct Contains {
    pub id: u64,
    pub key: Box<Contains_key>,
}

impl Contains {
    pub fn GLEAN_name() -> String {
        String::from("python.Contains.4")
    }

    pub fn new(container: Declaration, containee: Declaration) -> Self {
        Contains {
            id: 0,
            key: Box::new(Contains_key {
                container,
                containee
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainingTopLevelDeclaration {
    pub id: u64,
    pub key: Box<ContainingTopLevelDeclaration_key>,
}

impl ContainingTopLevelDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.ContainingTopLevelDeclaration.4")
    }

    pub fn new(declaration: Declaration, container: Declaration) -> Self {
        ContainingTopLevelDeclaration {
            id: 0,
            key: Box::new(ContainingTopLevelDeclaration_key {
                declaration,
                container
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedByTopLevelDeclaration {
    pub id: u64,
    pub key: Box<ContainedByTopLevelDeclaration_key>,
}

impl ContainedByTopLevelDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.ContainedByTopLevelDeclaration.4")
    }

    pub fn new(container: Declaration, declaration: Declaration) -> Self {
        ContainedByTopLevelDeclaration {
            id: 0,
            key: Box::new(ContainedByTopLevelDeclaration_key {
                container,
                declaration
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ContainedBy {
    pub id: u64,
    pub key: Box<ContainedBy_key>,
}

impl ContainedBy {
    pub fn GLEAN_name() -> String {
        String::from("python.ContainedBy.4")
    }

    pub fn new(containee: Declaration, container: Declaration) -> Self {
        ContainedBy {
            id: 0,
            key: Box::new(ContainedBy_key {
                containee,
                container
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDefinition {
    pub id: u64,
    pub key: Box<ClassDefinition_key>,
}

impl ClassDefinition {
    pub fn GLEAN_name() -> String {
        String::from("python.ClassDefinition.4")
    }

    pub fn new(declaration: ClassDeclaration, bases: Option<Vec<ClassDeclaration>>, keywords: Option<Vec<Parameter>>, decorators: Option<Vec<Decorator>>, container: Option<DeclarationContainer>) -> Self {
        ClassDefinition {
            id: 0,
            key: Box::new(ClassDefinition_key {
                declaration,
                bases,
                keywords,
                decorators,
                container
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDeclaration {
    pub id: u64,
    pub key: Box<ClassDeclaration_key>,
}

impl ClassDeclaration {
    pub fn GLEAN_name() -> String {
        String::from("python.ClassDeclaration.4")
    }

    pub fn new(name: Name, bases: Option<Vec<Name>>) -> Self {
        ClassDeclaration {
            id: 0,
            key: Box::new(ClassDeclaration_key {
                name,
                bases
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassBySName {
    pub id: u64,
    pub key: Box<SName>,
    pub value: ClassBySName_value,
}

impl ClassBySName {
    pub fn GLEAN_name() -> String {
        String::from("python.ClassBySName.4")
    }

    pub fn new(key: SName, value: ClassBySName_value) -> Self {
        ClassBySName {
            id: 0,
            key: Box::new(key),
            value: value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CalleeToCaller {
    pub id: u64,
    pub key: Box<CalleeToCaller_key>,
}

impl CalleeToCaller {
    pub fn GLEAN_name() -> String {
        String::from("python.CalleeToCaller.4")
    }

    pub fn new(callee: Name, caller: Name) -> Self {
        CalleeToCaller {
            id: 0,
            key: Box::new(CalleeToCaller_key {
                callee,
                caller
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct BaseClassToDerived {
    pub id: u64,
    pub key: Box<BaseClassToDerived_key>,
}

impl BaseClassToDerived {
    pub fn GLEAN_name() -> String {
        String::from("python.BaseClassToDerived.4")
    }

    pub fn new(base: ClassDeclaration, derived: ClassDeclaration) -> Self {
        BaseClassToDerived {
            id: 0,
            key: Box::new(BaseClassToDerived_key {
                base,
                derived
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileCall {
    pub id: u64,
    pub key: Box<FileCall_key>,
}

impl FileCall {
    pub fn GLEAN_name() -> String {
        String::from("python.FileCall.4")
    }

    pub fn new(file: src::File, callee_span: src::ByteSpan, call_args: Vec<CallArgument>) -> Self {
        FileCall {
            id: 0,
            key: Box::new(FileCall_key {
                file,
                callee_span,
                call_args
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct XRefsViaNameByTarget_key {
    pub target: Name,
    pub file: src::File,
    pub spans: Vec<src::ByteSpan>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typeInfo: Option<TypeInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
pub struct MethodOverridden_key {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returnsInfo: Option<TypeInfo>,
    pub params: Vec<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub posonly_params: Option<Vec<Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kwonly_params: Option<Vec<Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub star_arg: Option<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub star_kwarg: Option<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decorators: Option<Vec<Decorator>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<DeclarationContainer>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VariableDefinition_key {
    pub declaration: VariableDeclaration,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typeInfo: Option<TypeInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bases: Option<Vec<ClassDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decorators: Option<Vec<Decorator>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<DeclarationContainer>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ClassDeclaration_key {
    pub name: Name,
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<Name>,
    pub span: src::ByteSpan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argument: Option<Argument>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FileCall_key {
    pub file: src::File,
    pub callee_span: src::ByteSpan,
    pub call_args: Vec<CallArgument>,
}
