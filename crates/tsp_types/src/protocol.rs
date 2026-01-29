/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    #[serde(rename = "typeServer/getComputedType")]
    TypeServerGetComputedType,
    #[serde(rename = "typeServer/getDeclaredType")]
    TypeServerGetDeclaredType,
    #[serde(rename = "typeServer/getExpectedType")]
    TypeServerGetExpectedType,
    #[serde(rename = "typeServer/getPythonSearchPaths")]
    TypeServerGetPythonSearchPaths,
    #[serde(rename = "typeServer/getSnapshot")]
    TypeServerGetSnapshot,
    #[serde(rename = "typeServer/getSupportedProtocolVersion")]
    TypeServerGetSupportedProtocolVersion,
    #[serde(rename = "typeServer/resolveImport")]
    TypeServerResolveImport,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(tag = "method")]
pub enum TSPRequests {
    #[serde(rename = "typeServer/getComputedType")]
    GetComputedTypeRequest {
        id: serde_json::Value,
        params: serde_json::Value,
    },
    #[serde(rename = "typeServer/getDeclaredType")]
    GetDeclaredTypeRequest {
        id: serde_json::Value,
        params: serde_json::Value,
    },
    #[serde(rename = "typeServer/getExpectedType")]
    GetExpectedTypeRequest {
        id: serde_json::Value,
        params: serde_json::Value,
    },
    #[serde(rename = "typeServer/getPythonSearchPaths")]
    GetPythonSearchPathsRequest {
        id: serde_json::Value,
        params: GetPythonSearchPathsParams,
    },
    #[serde(rename = "typeServer/getSnapshot")]
    GetSnapshotRequest { id: serde_json::Value },
    #[serde(rename = "typeServer/getSupportedProtocolVersion")]
    GetSupportedProtocolVersionRequest { id: serde_json::Value },
    #[serde(rename = "typeServer/resolveImport")]
    ResolveImportRequest {
        id: serde_json::Value,
        params: ResolveImportParams,
    },
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TSPNotificationMethods {
    #[serde(rename = "typeServer/snapshotChanged")]
    TypeServerSnapshotChanged,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum MessageDirection {
    #[serde(rename = "clientToServer")]
    ClientToServer,
    #[serde(rename = "serverToClient")]
    ServerToClient,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TypeServerVersion {
    /// Initial protocol version
    #[serde(rename = "0.1.0")]
    V010,

    /// Added new request types and fields
    #[serde(rename = "0.2.0")]
    V020,

    /// Switch to more complex types
    #[serde(rename = "0.3.0")]
    V030,

    /// Switch to Type union and using stubs
    #[serde(rename = "0.4.0")]
    Current,
}

/// Flags that describe the characteristics of a type. These flags can be combined using bitwise operations.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TypeFlags {
    #[serde(rename = "None")]
    None,

    /// Indicates if the type can be instantiated.
    #[serde(rename = "Instantiable")]
    Instantiable,

    /// Indicates if the type represents an instance (as opposed to a class or type itself).
    #[serde(rename = "Instance")]
    Instance,

    /// Indicates if an instance of the type can be called like a function. (It has a `__call__` method).
    #[serde(rename = "Callable")]
    Callable,

    /// Indicates if the instance is a literal (like `42`, `"hello"`, etc.).
    #[serde(rename = "Literal")]
    Literal,

    /// Indicates if the type is an interface (a type that defines a set of methods and properties). In Python this would be a Protocol.
    #[serde(rename = "Interface")]
    Interface,

    /// Indicates if the type is a generic type (a type that can be parameterized with other types).
    #[serde(rename = "Generic")]
    Generic,

    /// Indicates if the type came from an alias (a type that refers to another type).
    #[serde(rename = "FromAlias")]
    Fromalias,

    /// Indicates if the type is unpacked (used with TypeVarTuple).
    #[serde(rename = "Unpacked")]
    Unpacked,

    /// Indicates if the type is optional (used with Tuple type arguments).
    #[serde(rename = "Optional")]
    Optional,

    /// Indicates if the type is unbound (used with *args in tuple type arguments).
    #[serde(rename = "Unbound")]
    Unbound,
}

/// Flags that describe the characteristics of a type variable. These flags can be combined using bitwise operations.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TypeVarFlags {
    #[serde(rename = "None")]
    None,

    /// Indicates if the type variable is a ParamSpec (as defined in PEP 612).
    #[serde(rename = "IsParamSpec")]
    Isparamspec,

    /// Indicates if the type variable is a TypeVarTuple (as defined in PEP 646).
    #[serde(rename = "IsTypeVarTuple")]
    Istypevartuple,
}

/// Represents the category of a declaration in the type system. This is used to classify declarations such as variables, functions, classes, etc.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum DeclarationCategory {
    /// An intrinsic refers to a symbol that has no actual declaration in the source code, such as built-in types or functions. One such example is a '__class__' declaration.
    #[serde(rename = "Intrinsic")]
    Intrinsic,

    /// A variable is a named storage location that can hold a value.
    #[serde(rename = "Variable")]
    Variable,

    /// A parameter is a variable that is passed to a function or method.
    #[serde(rename = "Param")]
    Param,

    /// This is for PEP 695 type parameters.
    #[serde(rename = "TypeParam")]
    Typeparam,

    /// This is for PEP 695 type aliases.
    #[serde(rename = "TypeAlias")]
    Typealias,

    /// A function is any construct that begins with the `def` keyword and has a body, which can be called with arguments.
    #[serde(rename = "Function")]
    Function,

    /// A class is any construct that begins with the `class` keyword and has a body, which can be instantiated.
    #[serde(rename = "Class")]
    Class,

    /// An import declaration, which is a reference to another module.
    #[serde(rename = "Import")]
    Import,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum ClassTypeFlags {
    #[serde(rename = "None")]
    None,

    #[serde(rename = "BuiltIn")]
    Builtin,

    #[serde(rename = "SpecialBuiltIn")]
    Specialbuiltin,

    #[serde(rename = "TypedDictClass")]
    Typeddictclass,

    #[serde(rename = "TypedDictMarkedClosed")]
    Typeddictmarkedclosed,

    #[serde(rename = "TypedDictEffectivelyClosed")]
    Typeddicteffectivelyclosed,

    #[serde(rename = "CanOmitDictValues")]
    Canomitdictvalues,

    #[serde(rename = "SupportsAbstractMethods")]
    Supportsabstractmethods,

    #[serde(rename = "PropertyClass")]
    Propertyclass,

    #[serde(rename = "Final")]
    Final,

    #[serde(rename = "ProtocolClass")]
    Protocolclass,

    #[serde(rename = "PseudoGenericClass")]
    Pseudogenericclass,

    #[serde(rename = "RuntimeCheckable")]
    Runtimecheckable,

    #[serde(rename = "TypingExtensionClass")]
    Typingextensionclass,

    #[serde(rename = "HasCustomClassGetItem")]
    Hascustomclassgetitem,

    #[serde(rename = "TupleClass")]
    Tupleclass,

    #[serde(rename = "EnumClass")]
    Enumclass,

    #[serde(rename = "ClassProperty")]
    Classproperty,

    #[serde(rename = "DefinedInStub")]
    Definedinstub,

    #[serde(rename = "TypeCheckOnly")]
    Typecheckonly,

    #[serde(rename = "NewTypeClass")]
    Newtypeclass,

    #[serde(rename = "ValidTypeAliasClass")]
    Validtypealiasclass,

    #[serde(rename = "SpecialFormClass")]
    Specialformclass,

    #[serde(rename = "IllegalIsinstanceClass")]
    Illegalisinstanceclass,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum ParamCategory {
    /// Regular parameter: def foo(x: int)
    #[serde(rename = "Simple")]
    Simple,

    /// Variadic positional: def foo(*args: int)
    #[serde(rename = "ArgsList")]
    Argslist,

    /// Variadic keyword: def foo(**kwargs: str)
    #[serde(rename = "KwargsDict")]
    Kwargsdict,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum FunctionParamFlags {
    #[serde(rename = "None")]
    None,

    #[serde(rename = "NameSynthesized")]
    Namesynthesized,

    #[serde(rename = "TypeDeclared")]
    Typedeclared,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum TypeKind {
    /// unknown, any, never, etc.
    #[serde(rename = "BuiltIn")]
    Builtin,

    /// Base for source-declared types (rarely used directly)
    #[serde(rename = "Declared")]
    Declared,

    /// Functions and methods from def statements
    #[serde(rename = "Function")]
    Function,

    /// Classes from class statements
    #[serde(rename = "Class")]
    Class,

    /// int | str | None
    #[serde(rename = "Union")]
    Union,

    /// import os -> os is ModuleType
    #[serde(rename = "Module")]
    Module,

    /// T, P, Ts in generics
    #[serde(rename = "TypeVar")]
    Typevar,

    /// Functions with multiple @overload signatures
    #[serde(rename = "Overloaded")]
    Overloaded,

    /// Types that are synthesized by the type checker
    #[serde(rename = "Synthesized")]
    Synthesized,

    /// Reference by ID for deduplication
    #[serde(rename = "TypeReference")]
    Typereference,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum DeclarationKind {
    /// Declaration exists in source code with AST node
    #[serde(rename = "Regular")]
    Regular,

    /// Declaration created by type checker (no source node)
    #[serde(rename = "Synthesized")]
    Synthesized,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
pub enum Variance {
    /// Variance not yet determined, will be inferred
    #[serde(rename = "Auto")]
    Auto,

    /// Variance cannot be determined
    #[serde(rename = "Unknown")]
    Unknown,

    /// No subtyping relationship (default for mutable types)
    #[serde(rename = "Invariant")]
    Invariant,

    /// Preserves subtyping: Generic[Child] <: Generic[Parent]
    #[serde(rename = "Covariant")]
    Covariant,

    /// Reverses subtyping: Generic[Parent] <: Generic[Child]
    #[serde(rename = "Contravariant")]
    Contravariant,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum LiteralValue {
    Int(i32),
    Bool(bool),
    String(String),
    Enum(EnumLiteral),
    Sentinel(SentinelLiteral),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum Declaration {
    Regular(RegularDeclaration),
    Synthesized(SynthesizedDeclaration),
}

pub type TypeVarType = DeclaredType;

#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(untagged)]
pub enum Type {
    BuiltInType(BuiltInType),
    Declared(DeclaredType),
    Function(FunctionType),
    Class(ClassType),
    Union(UnionType),
    Module(ModuleType),
    Var(TypeVarType),
    Overloaded(OverloadedType),
    Synthesized(SynthesizedType),
    Reference(TypeReferenceType),
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

/// Base interface for all type variants. Contains common fields shared by all types.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypeBase {
    /// Flags that describe the characteristics of this type.
    pub flags: Option<Vec<TypeFlags>>,

    /// The kind/category of this type.
    pub kind: TypeKind,
}

/// Base interface for all declaration variants.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeclarationBase {
    /// The category of this declaration.
    pub category: DeclarationCategory,

    /// The module where this declaration is defined.
    pub module_name: String,

    /// The name of the declaration.
    pub name: String,
}

/// A type that is declared in source code.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeclaredType {
    /// The declaration that defines this type.
    pub declaration: Option<Declaration>,

    /// Type arguments if this is a generic type.
    pub type_args: Option<Vec<Type>>,
}

/// Represents a location in source code (a node in the AST). Used to point to specific declarations, expressions, or statements in Python source files. Used for: - Pointing to where a type is declared - Identifying the location of expressions for type inference - Error reporting and diagnostics - Linking types back to their source definitions Examples: - For `def foo():`, the node points to the function declaration - For a variable `x = 42`, the node points to the assignment - For default parameter values in functions
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Node {
    /// The range of the node in the source file. This is a zero-based range, meaning the start and end positions are both zero-based The range uses character offsets the same way the LSP does.
    pub range: Range,

    /// URI of the source file containing this node.
    pub uri: String,
}

/// Represents a Python module name, handling both absolute and relative imports. Used for: - Import statement resolution - Tracking module dependencies - Resolving relative imports (from . import, from .. import) Examples: - `import os.path`: leadingDots=0, nameParts=['os', 'path'] - `from . import utils`: leadingDots=1, nameParts=['utils'] - `from ...parent import module`: leadingDots=3, nameParts=['parent', 'module'] - `import mymodule`: leadingDots=0, nameParts=['mymodule']
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModuleName {
    /// The leading dots in the module name. This is used to determine the relative import level.
    pub leading_dots: i32,

    /// The parts of the module name, split by dots. For example, for `my_module.sub_module`, this would be `['my_module', 'sub_module']`.
    pub name_parts: Vec<String>,
}

/// Options for customizing import resolution behavior. Controls how the type server resolves import statements and accesses imported symbols. Used for: - Fine-tuning import resolution during type checking - Controlling access to private/hidden module members - Optimizing resolution by skipping file checks TODO: See if we can remove this as these are pretty specific to Pyright at the moment. Examples: ```python # resolveLocalNames affects whether local assignments are resolved: from module import name name = something_else  # Does 'name' refer to import or local assignment? # allowExternallyHiddenAccess affects access to _private names: from module import _internal_function  # Normally hidden from external access ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportOptions {
    /// Whether to allow access to members that are hidden by external modules. When true, permits access to symbols marked as private (e.g., _private or not in __all__).
    pub allow_externally_hidden_access: Option<bool>,

    /// Whether to resolve local names in the import declaration. When true, considers local variable assignments that shadow imports.
    pub resolve_local_names: Option<bool>,

    /// Whether to skip checking if the file is needed for the import resolution. When true, optimizes by not verifying file existence/validity.
    pub skip_file_needed_check: Option<bool>,
}

/// Parameters for the ResolveImportRequest. Provides the context needed to resolve a Python import statement to its file location. Used when: - Resolving `import` or `from...import` statements - Finding the file that contains an imported module - Navigating to imported symbols Examples: ```python # In file.py: from os.path import join  # sourceUri = file.py, moduleDescriptor = os.path import mymodule          # sourceUri = file.py, moduleDescriptor = mymodule from . import utils      # sourceUri = file.py, moduleDescriptor = .utils (relative) ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResolveImportParams {
    /// The descriptor of the imported module. Contains the module name parts and leading dots for relative imports.
    pub module_descriptor: ModuleName,

    /// Snapshot version of the type server. Type server should throw a ServerCanceled exception if this snapshot is no longer current.
    pub snapshot: i32,

    /// The URI of the source file where the import is referenced. Used to resolve relative imports and determine the import context.
    pub source_uri: String,
}

/// Parameters for the GetPythonSearchPathsRequest. Requests the list of directories that Python searches for modules and packages. The search paths include: - Standard library directories - Site-packages directories (third-party packages) - Virtual environment paths (if active) - Project-specific paths (PYTHONPATH, src directories) Used for: - Resolving import statements to find module files - Auto-import suggestions - Determining which packages are available Example search paths: ``` [ "/usr/lib/python3.11",              # Standard library "/venv/lib/python3.11/site-packages",  # Virtual env packages "/project/src"                       # Project source ] ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetPythonSearchPathsParams {
    /// Root folder to get search paths from. Determines the Python environment and project context for path resolution.
    pub from_uri: String,

    /// Snapshot version of the type server. Type server should throw a ServerCanceled exception if this snapshot is no longer current.
    pub snapshot: i32,
}

/// Represents a single field in a TypedDict. Contains the type and metadata for one key-value pair in a TypedDict definition. Fields: - valueType: The type of the value for this key - isRequired: Whether this key must be present (vs. optional using NotRequired[]) - isReadOnly: Whether this key cannot be modified (using ReadOnly[]) - isProvided: Whether this key has been provided in a partial TypedDict construction Examples: ```python class Movie(TypedDict): name: str              # isRequired=true, isReadOnly=false year: int              # isRequired=true, isReadOnly=false rating: NotRequired[float]  # isRequired=false id: ReadOnly[int]      # isRequired=true, isReadOnly=true ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypedDictEntry {
    /// True if this key has been provided during partial TypedDict construction. Used for tracking incremental TypedDict building.
    pub is_provided: bool,

    /// True if this key cannot be modified after creation. Set by ReadOnly[] annotation. Example: `id: ReadOnly[int]` has isReadOnly=true.
    pub is_read_only: bool,

    /// True if this key must be present in the dictionary. False for NotRequired[] fields. Example: `year: NotRequired[int]` has isRequired=false.
    pub is_required: bool,

    /// The type of values for this TypedDict key. Example: For `name: str`, valueType is the str type.
    pub value_type: Box<Type>,
}

/// Represents all fields in a TypedDict class. Contains both explicitly declared fields and optional extra items. Fields: - knownItems: Dictionary mapping field names to their TypedDictEntry definitions - extraItems: Optional type for additional keys (when using extra_items parameter) Examples: ```python # Standard TypedDict class Movie(TypedDict): name: str year: int # knownItems = {"name": TypedDictEntry(...), "year": TypedDictEntry(...)} # extraItems = undefined # TypedDict with extra_items (closed TypedDict) class Config(TypedDict, extra_items=str): host: str port: int # knownItems = {"host": ..., "port": ...} # extraItems = TypedDictEntry for str type ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypedDictEntries {}

/// Represents a single type argument in a tuple type. Used for tuple[...] generic type arguments, handling both fixed and variadic elements. Fields: - type: The type of this tuple element - isUnbounded: True for *args-style variadic elements (zero or more of this type) - isOptional: True if this element can be omitted (has a default) Examples: ```python # Fixed tuple x: tuple[int, str, bool] # Three TupleTypeArg: [int, str, bool], all with isUnbounded=false # Unbounded tuple y: tuple[int, ...] # One TupleTypeArg: int with isUnbounded=true # Callable with optional params def foo(a: int, b: str = "") -> None: ... # Captured as tuple[int, str] where str has isOptional=true ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TupleTypeArg {
    /// For tuples captured from a callable, this indicates the corresponding positional parameter has a default argument and can therefore be omitted. Example: For `def foo(a: int, b: str = "")`, b's type has isOptional=true.
    pub is_optional: Option<bool>,

    /// Does the type argument represent a single value or an "unbounded" (zero or more) arguments? True for `tuple[int, ...]` (variadic), false for `tuple[int, str]` (fixed).
    pub is_unbounded: bool,

    /// The type of this tuple element. Example: In `tuple[int, str, bool]`, each position has a different type.
    #[serde(rename = "type")]
    pub type_: Box<Type>,
}

/// Represents information about a property method (getter, setter, or deleter). Used to track the implementation of @property decorated methods. Fields: - methodType: The type of the decorated function (fget, fset, or fdel) - classType: The class that declared this method Examples: ```python class Person: @property def name(self) -> str:  # fgetInfo contains this method's type return self._name @name.setter def name(self, value: str) -> None:  # fsetInfo contains this method's type self._name = value @name.deleter def name(self) -> None:  # fdelInfo contains this method's type del self._name ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PropertyMethodInfo {
    /// The class that declared this property method. Used to track which class in the inheritance hierarchy defined the method. Example: If a property is inherited, classType points to the declaring class.
    pub class_type: Option<Box<Type>>,

    /// The type of the decorated function (fget, fset, or fdel). Contains the function signature including parameters and return type. Example: For `@property def name(self) -> str`, this is the function type.
    pub method_type: Box<Type>,
}

/// Represents specialized (concrete) types for a generic function's parameters and return type. Used when generic type parameters are substituted with actual types. Fields: - parameterTypes: Concrete types for each parameter after type variable substitution - parameterDefaultTypes: Specialized types for default values (if different from declared) - returnType: Specialized return type after type variable substitution Examples: ```python # Generic function def identity[T](x: T) -> T: return x # When called as identity[int](42): # - parameterTypes = [int] (T substituted with int) # - returnType = int (T substituted with int) # For list.append bound to list[str]: # - parameterTypes = [str] (specialized from generic T) ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SpecializedFunctionTypes {
    /// Specialized types of default arguments for each parameter in the "parameters" array. If an entry is undefined or the entire array is missing, there is no specialized type, and the original "defaultType" should be used. Example: For a generic default value that depends on T, this contains the specialized version.
    pub parameter_default_types: Option<Vec<Option<Type>>>,

    /// Specialized types for each of the parameters in the "parameters" array. Array matches the parameters array, with type variables replaced by concrete types. Example: For `def foo[T](x: T)` specialized to `T=int`, parameterTypes=[int].
    pub parameter_types: Vec<Type>,

    /// Specialized type of the declared return type. Undefined if there is no declared return type. Example: For `def foo[T](x: T) -> T` specialized to `T=int`, returnType=int.
    pub return_type: Box<Type>,
}

/// Represents a literal value from an Enum. Used to track specific enum members as literal types. Fields: - className: Name of the enum class - itemName: Name of the specific enum member - itemType: Type of the enum member's value Examples: ```python from enum import Enum class Color(Enum): RED = 1 GREEN = 2 BLUE = 3 # Color.RED is an EnumLiteral: # className="Color", itemName="RED", itemType=int (for value 1) def process(color: Literal[Color.RED]) -> None: pass  # EnumLiteral tracks that it's specifically Color.RED ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EnumLiteral {
    /// Name of the enum class. Example: "Color" for the Color enum.
    pub class_name: String,

    /// Name of the specific enum member. Example: "RED" for Color.RED.
    pub item_name: String,

    /// Type of the enum member's value. Example: int type if the enum values are integers.
    pub item_type: Box<Type>,
}

/// Represents a sentinel value (a unique object used as a marker). Used for special singleton values that act as sentinels in APIs. Fields: - classNode: AST node where the sentinel class is defined - moduleName: Module containing the sentinel - className: Name of the sentinel class Examples: ```python # Common sentinel pattern class _Sentinel: pass MISSING = _Sentinel() def get_value(key: str, default: int | _Sentinel = MISSING) -> int: ... # MISSING is a SentinelLiteral pointing to the _Sentinel class instance # Used in standard library (e.g., dataclasses.MISSING) from dataclasses import field, MISSING # MISSING is tracked as a SentinelLiteral ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SentinelLiteral {
    /// Name of the sentinel class. Example: "_MISSING_TYPE" for the class of dataclasses.MISSING.
    pub class_name: String,

    /// AST node pointing to the sentinel class definition. Used to locate the class in source code.
    pub class_node: Node,

    /// Fully qualified module name where the sentinel is defined. Example: "dataclasses" for dataclasses.MISSING.
    pub module_name: String,
}

/// Represents a declaration that exists in source code. Points to the actual AST node where a symbol is declared. Fields: - category: Type of declaration (Variable, Function, Class, etc.) - node: AST node pointing to the declaration location - name: Name of the declared symbol (undefined for anonymous/implicit declarations) Examples: ```python def my_function(x: int) -> str:  # Function declaration return str(x) class MyClass:  # Class declaration x: int      # Variable declaration T = TypeVar('T')  # TypeParam declaration ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegularDeclaration {
    /// Category of the declaration (Variable, Function, Class, etc.). Determines how the declaration should be interpreted. Example: DeclarationCategory.Function for `def foo():`.
    pub category: DeclarationCategory,

    /// Name of the declared symbol, or undefined for anonymous declarations. Example: "foo" for `def foo():`, undefined for lambda functions.
    pub name: String,

    /// AST node pointing to the declaration location in source code. Contains file URI and range information. Example: Points to the `def` keyword and function name for function declarations.
    pub node: Node,
}

/// Represents a synthesized declaration (not in source code). Used for implicitly created symbols like built-in types or decorator-generated members. Fields: - uri: The file URI where this is conceptually declared (often the module using it) Examples: ```python # Built-in functions have synthesized declarations len([1, 2, 3])  # len is synthesized, not from source # @dataclass generates __init__, __eq__, etc. - synthesized declarations @dataclass class Point: x: int y: int # Point.__init__ is synthesized ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SynthesizedDeclaration {
    /// URI of the file where this symbol is conceptually declared. For built-ins, this might be a special URI; for decorator-generated code, it's the file containing the decorator. Example: File URI of a @dataclass-decorated class for synthesized __init__.
    pub uri: String,
}

/// Contains metadata about a type alias. Used when a type is created through a type alias statement (PEP 613) or traditional assignment. Fields: - name: Short name of the alias - fullName: Fully qualified name including module path - moduleName: Module where the alias is defined - fileUri: File location of the alias definition - scopeId: Scope identifier for the alias (for scoped type variables) - isTypeAliasType: True if this uses the `type` keyword (PEP 695) - typeParams: Generic type parameters declared by the alias - typeArgs: Concrete type arguments when the alias is specialized - computedVariance: Inferred variance for type parameters Examples: ```python # PEP 695 style (isTypeAliasType=true) type IntList = list[int] # Traditional style (isTypeAliasType=false) IntList = list[int] # Generic alias with type parameters type Pair[T] = tuple[T, T] # typeParams=[T], can be specialized to Pair[int] # Using typing.TypeAlias from typing import TypeAlias UserId: TypeAlias = int ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypeAliasInfo {
    /// Computed variance for each type parameter. Inferred based on how type parameters are used in the alias definition. Example: [Covariant] if the type parameter only appears in return positions.
    pub computed_variance: Option<Vec<Variance>>,

    /// URI of the file containing the type alias definition. Example: "file:///path/to/mymodule.py".
    pub file_uri: String,

    /// Fully qualified name including module path. Example: "mymodule.IntList".
    pub full_name: String,

    /// True if this alias uses the `type` keyword (PEP 695), false for traditional assignment. Example: true for `type X = int`, false for `X = int`.
    pub is_type_alias_type: bool,

    /// Module where the type alias is defined. Example: "mymodule" for a type defined in mymodule.py.
    pub module_name: String,

    /// Short name of the type alias. Example: "IntList" for `type IntList = list[int]`.
    pub name: String,

    /// Scope identifier for type variables used in this alias. Ensures type variables are scoped to this alias definition. Example: Different aliases can use the same type variable name 'T' without conflict.
    pub scope_id: String,

    /// Concrete type arguments when this alias is specialized. Example: [int] when `Pair[int]` is used (specializing Pair[T]).
    pub type_args: Option<Vec<Type>>,

    /// Generic type parameters declared by this alias. Example: [T] for `type Pair[T] = tuple[T, T]`.
    pub type_params: Option<Vec<Type>>,
}

/// Represents special built-in types that are fundamental to Python's type system. These are not regular classes but represent special semantic meanings. Used for: - Type inference failures (unknown) - Gradual typing (any) - Uninitialized variables (unbound) - Special literals (ellipsis for ...) - Non-returning functions (never/noreturn) Examples: - `unknown`: `x` in `def foo(x):` with no type hints and no usage to infer from - `any`: Explicit `Any` annotation or from untyped imports - `unbound`: Variable declared but not yet assigned: `x: int` (before assignment) - `ellipsis`: The `...` in `def foo(...): ...` or `Tuple[int, ...]` - `never`: `def raise_error() -> Never:` or function with only raise statements
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BuiltInType {
    /// Optional declaration information for built-in types (usually undefined for true built-ins). Example: Some built-ins like __class__ have synthesized declarations.
    pub declaration: Option<Declaration>,

    /// The name of the built-in type. Limited to specific known built-in types. 'unknown': Type cannot be determined 'any': Accepts any value (gradual typing) 'unbound': Variable not yet bound to a value 'ellipsis': The ... literal 'never': Type that never occurs (e.g., function that always raises) 'noreturn': Function that doesn't return (alias for never)
    pub name: String,

    /// For 'unknown' types, this may contain a possible type based on context. Used when type inference has partial information but can't fully determine the type. Example: In `if isinstance(x, int): ...` the possibleType of unknown x might be int
    pub possible_type: Option<Box<Type>>,
}

/// Represents a function or method that has a declaration in the source code. Used for functions parsed from actual `def` statements. Uses TypeKind.Function for discrimination from ClassType and other types. Binding behavior: - boundToType: Contains the class/instance the method is bound to. Used for: - User-defined functions with `def` statements - Methods declared in source classes - Lambda functions (though simple ones) Not used for: - Built-in functions like `len`, `print` (use SynthesizedType) - Synthesized methods from decorators like @dataclass (use SynthesizedType) Example: ```python def calculate(x: int, y: int) -> int: return x + y class MyClass: def method(self, value: str) -> None: pass ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FunctionType {
    /// The class or object instance that this method is bound to. Example: In `obj.method`, boundToType is the type of `obj`.
    pub bound_to_type: Option<Box<Type>>,

    /// The return type annotation of the function. Example: In `def foo() -> int:`, returnType is the int type.
    pub return_type: Option<Box<Type>>,

    /// Specialized versions of parameter types and return type when the function has type parameters. Contains concrete types substituted for generic type variables. Example: When calling `list[int].append(1)`, the self parameter is specialized to list[int].
    pub specialized_types: Option<SpecializedFunctionTypes>,
}

/// Represents a class or class instance that has a declaration in the source code. Used for classes parsed from actual `class` statements. Uses TypeKind.Class for discrimination from FunctionType and other types. Used for: - User-defined classes with `class` statements - Class instances (instances of user-defined classes) - Specialized generic classes (e.g., `MyClass[int]`) - Literal instances (e.g., the number `42` is an instance of `int`) Not used for: - Built-in classes like `int`, `str`, `list` (use SynthesizedType) - Classes synthesized by decorators (use SynthesizedType) Example: ```python class Point: x: int y: int class Container[T]: value: T # point has ClassType (instance of Point) point = Point() # container has ClassType with typeArgs=[int] container: Container[int] = Container() ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ClassType {
    /// The literal value if this class represents a literal (e.g., int literal 42, str literal "hello"). Can be a primitive value, enum member, or sentinel object. Example: For the literal `42`, literalValue = 42.
    pub literal_value: Option<LiteralValue>,

    /// Type arguments when this class is a specialized generic type. Example: For `list[int]`, typeArgs = [int].
    pub type_args: Option<Vec<Type>>,
}

/// Represents a union of multiple types (Type1 | Type2 | ...). Used when a value can be one of several different types. Used for: - Explicit union type annotations using `|` or `Union[...]` - Optional types (which are unions with None) - Type narrowing results (e.g., after isinstance checks) - Inferred types from multiple branches Examples: ```python # Explicit union annotation def process(value: int | str) -> None: pass # Optional (union with None) def find(key: str) -> str | None: return None # Inferred union from branches if condition: x = 42        # int else: x = "hello"  # str # x has type int | str ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UnionType {
    /// Array of types that make up this union. Example: For `int | str | None`, subTypes = [int, str, None].
    pub sub_types: Vec<Type>,
}

/// Represents a Python module as a type. Used when a module object itself is referenced (not its contents). Used for: - Module imports: `import os` makes `os` a ModuleType - Module attributes accessed via __file__, __name__, etc. - Submodule references: `os.path` is also a ModuleType The loaderFields contain all the symbols exported by the module that would be accessible via attribute access (module.symbol_name). Examples: ```python import os import os.path as path from typing import Protocol # `os` has ModuleType with loaderFields containing {"path": ..., "getcwd": ..., etc.} # `path` has ModuleType for the os.path module # In type stubs, Protocol is a module symbol that gets loaded ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModuleType {
    /// Fully qualified name of the module. Example: "os.path" for the os.path module.
    pub module_name: String,

    /// URI of the module's source file. Example: "file:///path/to/module.py" or "<builtin>" for built-in modules.
    pub uri: String,
}

/// Represents an overloaded function with multiple signatures. Used when a function has multiple `@overload` decorators defining different call signatures. Used for: - Functions with @overload decorators - Built-in functions with multiple signatures (e.g., `range(stop)` vs `range(start, stop, step)`) - Methods with different signatures for different argument types The `overloads` array contains all the @overload signatures, and `implementation` contains the actual implementation (if present). Examples: ```python from typing import overload @overload def process(value: int) -> str: ... @overload def process(value: str) -> int: ... def process(value: int | str) -> int | str: if isinstance(value, int): return str(value) return len(value) # The type of `process` is OverloadedType with: # - overloads = [signature for (int)->str, signature for (str)->int] # - implementation = signature for (int|str)->(int|str) ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OverloadedType {
    /// The implementation signature (if present). This is the actual function body, as opposed to the @overload declarations. Example: The non-decorated function definition after all @overload decorators.
    pub implementation: Option<Box<Type>>,

    /// List of overload signatures for this overloaded function. Each overload represents a different way the function can be called. Example: For a function with @overload decorators, each overload is in this array.
    pub overloads: Vec<Type>,
}

/// Metadata about a synthesized type that provides additional context. This information is used by the client to enhance IntelliSense and type checking.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SynthesizedTypeMetadata {
    pub module: ModuleType,

    pub primary_definition_offset: i32,
}

/// Represents synthesized/generated types. When the type server generates its own types that do not directly correspond to source code declarations, it uses this handle. The stub content should be a complete, valid Python stub (.pyi) that includes: 1. All necessary imports (typing module, collections.abc, etc.) 2. TypeVar and ParamSpec declarations used in the type 3. Type aliases or class definitions 4. Function signatures with full parameter and return type annotations This approach is particularly useful for: - Synthesized methods from decorators like @dataclass.__init__ - NewType declarations - Generic type specializations Examples: # Example 1: Synthesized dataclass __init__ from dataclasses import dataclass @dataclass class Point: x: int y: int # Stub content for Point.__init__: """ def __init__(self, x: int, y: int) -> None: '''Initialize Point''' """ # metadata: { primaryDefinitionOffset: 0 } # Example 2: Generic function with TypeVar from typing import TypeVar T = TypeVar('T') def identity(x: T) -> T: return x # Stub content: """ from typing import TypeVar T = TypeVar('T') def identity(x: T) -> T: ... """ # metadata: { primaryDefinitionOffset: 45 } (offset points to 'def') # Example 3: ParamSpec function from typing import ParamSpec, Callable P = ParamSpec('P') def wrapper(func: Callable[P, int]) -> Callable[P, str]: ... # Stub content: """ from typing import ParamSpec, Callable P = ParamSpec('P') def wrapper(func: Callable[P, int]) -> Callable[P, str]: ... """ # metadata: { primaryDefinitionOffset: 67 } # Example 4: NewType from typing import NewType UserId = NewType('UserId', int) # Stub content: """ from typing import NewType UserId = NewType('UserId', int) """ # metadata: { primaryDefinitionOffset: 25 } (offset points to 'UserId') # Example 5: Complex generic specialization with ParamSpec class Wrapper[P, R]: func: Callable[P, R] def example(x: int, y: str) -> bool: ... w: Wrapper[(x: int, y: str), bool] = Wrapper() # Stub content for Wrapper[(x: int, y: str), bool]: """ from typing import ParamSpec, TypeVar, Generic, Callable P = ParamSpec('P') R = TypeVar('R') class Wrapper(Generic[P, R]): func: Callable[P, R] """ # metadata: { primaryDefinitionOffset: 87 } (offset points to 'class Wrapper') ``` Important: The stub content is used to reconstruct the type on the client side by: 1. Parsing the stub as a Python type stub file 2. Evaluating the type expressions within the stub 3. Extracting the resulting type for use in type checking and IntelliSense
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SynthesizedType {
    pub metadata: SynthesizedTypeMetadata,

    pub stub_content: String,
}

/// Represents a reference to another type by its ID. Used to avoid duplicating large type structures and to handle forward references. Used for: - Deduplication: When the same type appears multiple times, subsequent occurrences can reference the first occurrence instead of duplicating all fields - Cyclic references: Breaking cycles in recursive type definitions - Large types: Reducing payload size for complex types used repeatedly This is an optimization mechanism in the protocol to keep type handles compact when transmitting over the wire. Examples: ```python # Recursive type definition class Node: value: int next: Node | None  # 'Node' references back to itself # When serializing the type of 'next', the second occurrence of Node # uses TypeReferenceType pointing to the first Node's ID # Repeated complex type def process_lists( list1: list[dict[str, int]], list2: list[dict[str, int]],  # Can reference the type from list1 list3: list[dict[str, int]]   # Can reference the type from list1 ) -> None: pass ```
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypeReferenceType {
    /// Identifier that references another Type by its id. Used to avoid duplicating large type structures and handle forward references. Example: When a type appears multiple times, later occurrences use TypeReference pointing to the first occurrence's id.
    pub type_reference_id: i32,
}

/// Notification sent by the server to indicate any outstanding snapshots are invalid.
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SnapshotChangedNotification {
    /// The version of the JSON RPC protocol.
    pub jsonrpc: String,

    /// The method to be invoked.
    pub method: TSPNotificationMethods,

    pub params: Option<serde_json::Value>,
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

/// Requests and notifications for the type server protocol. Request for the computed type of a declaration or node. Computed type is the type that is inferred based on the code flow. Example: def foo(a: int | str): if instanceof(a, int): b = a + 1  # Computed type of 'b' is 'int'
#[derive(Serialize, Deserialize, PartialEq, Debug, Eq, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GetComputedTypeRequest {
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
pub struct GetDeclaredTypeRequest {
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
pub struct GetExpectedTypeRequest {
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
pub struct GetPythonSearchPathsRequest {
    /// The method to be invoked.
    pub method: TSPRequestMethods,

    /// The request id.
    pub id: LSPId,

    pub params: GetPythonSearchPathsParams,
}

/// Response to the [GetPythonSearchPathsRequest].
pub type GetPythonSearchPathsResponse = Vec<String>;

/// Request from client to get the current snapshot of the type server. A snapshot is a point-in-time representation of the type server's state, including all loaded files and their types. A type server should change its snapshot whenever any type it might have returned is no longer valid. Meaning types are only usable for the snapshot they were returned with. Snapshots are not meant to survive any changes that would make the type server throw away its internal cache. They are merely an identifier to indicate to the client that the type server will accept requests for types from that snapshot.
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

/// Request to get the version of the protocol the type server supports. Returns a string representation of the protocol version (should be semver format)
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
