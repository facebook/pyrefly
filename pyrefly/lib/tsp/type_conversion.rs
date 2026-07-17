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
//!  - `Any`, `Never`, `Ellipsis` → TSP `BuiltInType`.
//!  - Solver-internal types → TSP `BuiltInType` with a representative name.
//!
//! Note: `None` is emitted as a `NoneType` `ClassType`, not as a `BuiltInType`,
//! because `BuiltInType.name` is restricted to protocol sentinel names. See
//! [`convert_type_with_resolvers`] for how the version-correct class is sourced.
//! All `Type` variants are explicitly handled; no types fall through to a
//! generic `SynthesizedType` stub.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use lsp_types::Url;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::FuncId;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Params;
use pyrefly_types::callable_residual::CallableResidualKind;
use pyrefly_types::class::Class;
use pyrefly_types::class::ClassType as PyreflyClassType;
use pyrefly_types::literal::Lit;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedOrigin;
use pyrefly_types::sentinel::Sentinel;
use pyrefly_types::type_alias::TypeAliasData;
use pyrefly_types::type_alias::TypeAliasRef;
use pyrefly_types::types::BoundMethodType;
use pyrefly_types::types::Forallable;
use pyrefly_types::types::Type as PyreflyType;
use ruff_python_ast::name::Name;
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
use tsp_types::SentinelLiteral;
use tsp_types::SpecializedFunctionTypes;
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

/// Callback that resolves a module name (e.g. `pkg.subpkg`) to a canonical
/// filesystem path for that module (preferably package `__init__.py[i]` for
/// packages).
pub type ModulePathResolver<'a> =
    dyn Fn(&pyrefly_types::module::ModuleType) -> Option<PathBuf> + 'a;

/// Callback that resolves an exported symbol (by defining module and name) to
/// the `ModulePath` and `lsp_types::Range` of its original definition,
/// following re-exports. Used to give real source locations to special forms,
/// `typing` classes, and functions whose `FuncId` lacks a `def_index` (e.g.
/// imported user functions and special functions like `typing.overload`).
pub type ExportLocationResolver<'a> =
    dyn Fn(ModuleName, &Name) -> Option<(ModulePath, lsp_types::Range)> + 'a;

/// The stdlib classes used to encode pyrefly types that would otherwise be
/// emitted as off-spec `BuiltInType` sentinels: the protocol restricts
/// `BuiltInType.name` to a fixed set that excludes names like `none`, `bool`,
/// and `int`. Passing the real classes (rather than re-deriving them by name)
/// keeps each declaration version-correct — e.g. `NoneType` is sourced from
/// `types` on Python 3.10+ and `_typeshed` before — and identical to writing
/// the annotation explicitly.
#[derive(Clone, Copy)]
pub struct StdlibClasses<'a> {
    /// `NoneType`, encoding `None`.
    pub none_type: &'a PyreflyClassType,
    /// `bool`, encoding `TypeGuard`/`TypeIs` (their runtime type).
    pub bool_type: &'a PyreflyClassType,
    /// `int`, encoding `Size`/`Dim` (integer tensor dimensions).
    pub int_type: &'a PyreflyClassType,
}

/// Convert a pyrefly `Type` to a TSP protocol `Type` using optional
/// source-range and module-URI resolvers, plus the stdlib classes used to
/// encode sentinel-like types (see [`StdlibClasses`]).
pub fn convert_type_with_resolvers<'a>(
    ty: &PyreflyType,
    func_range_resolver: Option<&'a FuncRangeResolver<'a>>,
    module_path_resolver: Option<&'a ModulePathResolver<'a>>,
    export_location_resolver: Option<&'a ExportLocationResolver<'a>>,
    stdlib: StdlibClasses<'a>,
) -> TspType {
    TypeConverter {
        resolve_func_range: func_range_resolver,
        resolve_module_path: module_path_resolver,
        resolve_export: export_location_resolver,
        stdlib,
    }
    .convert(ty)
}

/// Convert a pyrefly `Type` to a TSP protocol `Type`.
///
/// Function declarations will have zero-range nodes since no binding
/// resolver is available. Use [`convert_type_with_resolvers`] when source
/// locations are needed.
#[cfg(test)]
pub fn convert_type(ty: &PyreflyType) -> TspType {
    let stdlib = TestStdlib::new();
    convert_type_with_resolvers(ty, None, None, None, stdlib.classes())
}

/// Stand-in for the real `Stdlib` classes used by the resolver-free tests,
/// which have no stdlib. Each class mirrors its real counterpart closely enough
/// to exercise the production path: a top-level class in its bundled module.
#[cfg(test)]
struct TestStdlib {
    none_type: PyreflyClassType,
    bool_type: PyreflyClassType,
    int_type: PyreflyClassType,
}

#[cfg(test)]
impl TestStdlib {
    fn new() -> Self {
        Self {
            none_type: test_class(ModuleName::types(), "NoneType"),
            bool_type: test_class(ModuleName::builtins(), "bool"),
            int_type: test_class(ModuleName::builtins(), "int"),
        }
    }

    fn classes(&self) -> StdlibClasses<'_> {
        StdlibClasses {
            none_type: &self.none_type,
            bool_type: &self.bool_type,
            int_type: &self.int_type,
        }
    }
}

/// Build a top-level `name` class in bundled `module_name` (`<module>.pyi`).
#[cfg(test)]
fn test_class(module_name: ModuleName, name: &str) -> PyreflyClassType {
    use pyrefly_python::module::Module;
    use pyrefly_python::nesting_context::NestingContext;
    use pyrefly_types::class::ClassDefIndex;
    use pyrefly_types::types::TArgs;
    use ruff_python_ast::Identifier;

    let module = Module::new(
        module_name,
        ModulePath::bundled_typeshed(PathBuf::from(format!("{module_name}.pyi"))),
        Arc::new(String::new()),
    );
    let class = Class::new(
        ClassDefIndex(0),
        Identifier::new(Name::new(name), TextRange::default()),
        NestingContext::toplevel(),
        module,
        None,
    );
    PyreflyClassType::new(class, TArgs::default())
}

/// Holds an optional range resolver and drives recursive type conversion.
struct TypeConverter<'a> {
    resolve_func_range: Option<&'a FuncRangeResolver<'a>>,
    resolve_module_path: Option<&'a ModulePathResolver<'a>>,
    resolve_export: Option<&'a ExportLocationResolver<'a>>,
    /// Stdlib classes used to encode sentinel-like types; see [`StdlibClasses`].
    stdlib: StdlibClasses<'a>,
}

impl TypeConverter<'_> {
    /// Convert a pyrefly `Type` to a TSP protocol `Type`.
    fn convert(&self, ty: &PyreflyType) -> TspType {
        match ty {
            // --- Built-in special types ---
            PyreflyType::Any(_) => builtin("any"),
            PyreflyType::Never(_) => builtin("never"),
            // `None` → the stdlib's real `NoneType` class (see `stdlib`).
            PyreflyType::None => {
                self.convert_class_type(self.stdlib.none_type, TypeFlags::INSTANCE)
            }
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
            PyreflyType::CallableResidual(residual) => match &residual.kind {
                CallableResidualKind::Generic { quantified } => {
                    self.convert(&quantified.as_gradual_type())
                }
                CallableResidualKind::Overload { .. } => self.convert(&PyreflyType::any_implicit()),
            },

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
            PyreflyType::Module(m) => {
                let module_name = m.to_string();
                let uri = self
                    .resolve_module_path
                    .and_then(|resolve| resolve(m))
                    .map(|path| path_buf_to_uri(&path))
                    .unwrap_or_default();
                TspType::Module(TspModuleType {
                    flags: TypeFlags::NONE,
                    id: next_id(),
                    kind: TypeKind::Module,
                    module_name,
                    type_alias_info: None,
                    uri,
                })
            }

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
                    // Anonymous TypedDict — fall back to the typing.TypedDict class
                    self.typing_class("TypedDict", TypeFlags::INSTANTIABLE)
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
                Forallable::TypeAlias(ta) => self.convert_type_alias_data(ta),
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
            // `type[X]` is the class object (instantiable), not an instance,
            // whatever TSP shape `X` converted to: a `Class` for `type[C]`, a
            // `Var` for `type[T]`, a `Union` for `type[A | B]`, etc.
            PyreflyType::Type(inner) => mark_instantiable(self.convert(inner)),

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
            // These are TypeVar-like solver-internal placeholders. Emit them as
            // TSP TypeVar so consumers don't see them as malformed BuiltIns.
            PyreflyType::Quantified(q) | PyreflyType::QuantifiedValue(q) => {
                self.convert_quantified(q)
            }

            // --- LiteralString → typing.LiteralString class ---
            PyreflyType::LiteralString(_) => {
                self.typing_class("LiteralString", TypeFlags::INSTANCE)
            }

            // --- Annotated[X, ...] → unwrap to X ---
            PyreflyType::Annotated(inner, _) => self.convert(inner),

            // --- TypeGuard[X] / TypeIs[X] → the stdlib `bool` class (their runtime type) ---
            // Emitted as the real class, not `builtin("bool")`: the protocol
            // restricts `BuiltInType.name` to a fixed sentinel set that excludes
            // `bool`, so a bare builtin surfaces as Unknown on the consumer.
            PyreflyType::TypeGuard(_) | PyreflyType::TypeIs(_) => {
                self.convert_class_type(self.stdlib.bool_type, TypeFlags::INSTANCE)
            }

            // --- SuperInstance → convert as the class type ---
            PyreflyType::SuperInstance(si) => self.convert_class_type(&si.0, TypeFlags::INSTANCE),

            // --- Tensor → ClassType from base_class ---
            PyreflyType::ShapedArray(t) => {
                self.convert_class_type(&t.base_class, TypeFlags::INSTANCE)
            }

            PyreflyType::IntTuple(_) => builtin("tuple"),

            // --- NNModule → ClassType from class ---
            PyreflyType::NNModule(m) => self.convert_class_type(&m.class, TypeFlags::INSTANCE),

            // --- DataFrame → convert the underlying instance type ---
            PyreflyType::DataFrame(schema) => self.convert(&schema.underlying_type()),

            // --- TypeAlias → unwrap to the aliased type, or typing class for refs ---
            PyreflyType::TypeAlias(ta) | PyreflyType::UntypedAlias(ta) => {
                self.convert_type_alias_data(ta.as_ref())
            }

            // --- SpecialForm → typing.<form-name> class ---
            // The TSP protocol restricts BuiltInType.name to a fixed set of
            // sentinel names (unknown/any/never/etc.), so emitting forms like
            // `Literal`/`Final`/`ClassVar` as BuiltIn was off-spec and surfaced
            // as Unknown on the consumer side. Emit them as ClassType
            // referencing the typing module instead.
            PyreflyType::SpecialForm(sf) => {
                self.typing_class(&sf.to_string(), TypeFlags::INSTANTIABLE)
            }

            // --- Unpack(X) → convert inner ---
            PyreflyType::Unpack(inner) => self.convert(inner),

            // --- TypeForm(X) → convert inner ---
            PyreflyType::TypeForm(inner) => self.convert(inner),

            // --- Intersect → convert the fallback type ---
            PyreflyType::Intersect(pair) => self.convert(&pair.1),

            // --- ElementOfTypeVarTuple → TypeVar ---
            PyreflyType::ElementOfTypeVarTuple(q) => synthesized_typevar(q.name.as_str()),

            // --- ParamSpec-related internal types → TypeVar ---
            PyreflyType::Args(q)
            | PyreflyType::Kwargs(q)
            | PyreflyType::ArgsValue(q)
            | PyreflyType::KwargsValue(q) => synthesized_typevar(q.name.as_str()),

            // --- ParamSpecValue → typing.ParamSpec ---
            PyreflyType::ParamSpecValue(_) => {
                self.typing_class("ParamSpec", TypeFlags::INSTANTIABLE)
            }

            // --- Concatenate → typing.Concatenate ---
            PyreflyType::Concatenate(..) => {
                self.typing_class("Concatenate", TypeFlags::INSTANTIABLE)
            }

            // --- KwCall → convert the return type ---
            PyreflyType::KwCall(kw) => self.convert(&kw.return_ty),

            // --- Int → the stdlib `int` class (symbolic integers represent dimensions) ---
            // Emitted as the real class, not `builtin("int")`: the protocol
            // restricts `BuiltInType.name` to a fixed sentinel set that excludes
            // `int`, so a bare builtin surfaces as Unknown on the consumer.
            PyreflyType::Int(_) => {
                self.convert_class_type(self.stdlib.int_type, TypeFlags::INSTANCE)
            }

            // --- Solver-internal variable → built-in unknown ---
            PyreflyType::Var(_) => builtin("unknown"),

            // --- Materialization is a solver artifact ---
            PyreflyType::Materialization => builtin("unknown"),

            // --- Sentinel → a `ClassType` carrying a `SentinelLiteral` ---
            // The protocol has a dedicated sentinel literal (class name plus its
            // defining location), so emit that rather than an off-spec
            // `sentinel` `BuiltInType` that surfaces as Unknown. The location
            // comes from the sentinel's own `QName`, so no stdlib class is
            // needed.
            PyreflyType::Sentinel(s) => convert_sentinel(s),
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
    ///
    /// Like [`convert_function`](Self::convert_function), the parameter and
    /// return types are carried in `specialized_types` so the consumer can
    /// reconstruct the signature; a synthesized callable has no source
    /// declaration, so this is the only channel for its parameter types.
    fn convert_callable(&self, callable: &Callable) -> TspType {
        let ret = self.convert(&callable.ret);
        let specialized_types = self.specialized_types(callable, &ret);
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
            specialized_types,
            type_alias_info: None,
        })
    }

    /// Convert the body of a type alias. A value alias unwraps to its aliased
    /// type; a `Ref` (a bare, not-yet-expanded reference, e.g. the recursive
    /// `X` in `type X = int | list[X]`) has no backing class, so it is emitted
    /// as a resolvable class handle pointing at the alias's own definition.
    /// Shared by the direct `TypeAlias` arm and the `Forall`-wrapped one so
    /// they stay in sync.
    fn convert_type_alias_data(&self, ta: &TypeAliasData) -> TspType {
        match ta {
            TypeAliasData::Value(alias) => self.convert(&alias.as_type()),
            TypeAliasData::Ref(r) => self.alias_ref_class(r),
        }
    }

    /// Build a TSP `ClassType` for a type-alias reference, resolving its
    /// declaration against the alias's *own* defining module (`r.module_name`)
    /// rather than assuming `typing`. The export resolver pins the exact
    /// definition range; when it is unavailable we fall back to a zero range in
    /// the alias's real module file (`r.module_path`) — still the correct file,
    /// just an imprecise position — never a bare builtin or the wrong module.
    fn alias_ref_class(&self, r: &TypeAliasRef) -> TspType {
        let (uri, range) = self
            .resolve_export
            .and_then(|resolve| resolve(r.module_name, &r.name))
            .map_or_else(
                || (path_to_uri(&r.module_path), zero_range()),
                |(module_path, lsp_range)| (path_to_uri(&module_path), lsp_range_to_tsp(lsp_range)),
            );
        TspType::Class(TspClassType {
            declaration: Declaration::Regular(RegularDeclaration {
                kind: DeclarationKind::Regular,
                category: DeclarationCategory::Class,
                name: Some(r.name.to_string()),
                node: Node { range, uri },
            }),
            flags: TypeFlags::INSTANTIABLE,
            id: next_id(),
            kind: TypeKind::Class,
            literal_value: None,
            type_alias_info: None,
            type_args: None,
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
        let declaration = self.function_declaration(kind);
        let specialized_types = self.specialized_types(callable, &ret);

        TspType::Function(TspFunctionType {
            bound_to_type,
            declaration,
            flags: TypeFlags::CALLABLE,
            id: next_id(),
            kind: TypeKind::Function,
            return_type: Some(Box::new(ret)),
            specialized_types,
            type_alias_info: None,
        })
    }

    /// Build `SpecializedFunctionTypes` carrying the converted parameter and
    /// return types.
    ///
    /// Pylance rebuilds a function's parameter *names* by parsing the source
    /// declaration, but it cannot evaluate the parameter/return *types* of an
    /// external type server's file (its in-process evaluator has not parsed
    /// the typeshed/workspace those declarations live in), so they degrade to
    /// `Unknown`. Sending the already-converted types here lets Pylance
    /// overlay real types onto the source-derived parameter list.
    fn specialized_types(
        &self,
        callable: &Callable,
        ret: &TspType,
    ) -> Option<SpecializedFunctionTypes> {
        let Params::List(params) = &callable.params else {
            return None;
        };

        let parameter_types: Vec<TspType> = params
            .items()
            .iter()
            .map(|param| self.convert(param.as_type()))
            .collect();

        Some(SpecializedFunctionTypes {
            parameter_default_types: None,
            parameter_types,
            return_type: Some(Box::new(ret.clone())),
        })
    }

    /// Convert a `Quantified` (solver-internal TypeVar placeholder) to a TSP
    /// `TypeVar`.
    ///
    /// A `Quantified` carries a `QuantifiedIdentity` pinning the source module
    /// and range where its TypeVar was declared. When the export-location
    /// resolver can map that `(module, name)` back to a real definition we
    /// build a `RegularDeclaration` with the true source location, so Pylance
    /// resolves the declaration and renders the TypeVar's name instead of
    /// `Unknown`. Otherwise we fall back to a synthesized (locationless)
    /// declaration.
    fn convert_quantified(&self, q: &Quantified) -> TspType {
        if let Some(resolve) = self.resolve_export {
            let identity = q.identity();
            if let Some((module_path, lsp_range)) = resolve(identity.module, &q.name) {
                // A PEP 695 type parameter (`def f[T]()`) is a real type-param
                // declaration; a legacy `T = TypeVar("T")` resolves to a module-
                // level *variable* in the consumer. Use the matching category so
                // Pylance's declaration lookup succeeds.
                let category = match identity.origin {
                    QuantifiedOrigin::Pep695 => DeclarationCategory::Typeparam,
                    _ => DeclarationCategory::Variable,
                };
                return TspType::Var(DeclaredType {
                    declaration: Declaration::Regular(RegularDeclaration {
                        category,
                        kind: DeclarationKind::Regular,
                        name: Some(q.name.to_string()),
                        node: Node {
                            range: lsp_range_to_tsp(lsp_range),
                            uri: path_to_uri(&module_path),
                        },
                    }),
                    flags: TypeFlags::NONE,
                    id: next_id(),
                    kind: TypeKind::Typevar,
                    type_alias_info: None,
                });
            }
        }

        synthesized_typevar(q.name.as_str())
    }

    /// Build a declaration for a function described by `kind`.
    ///
    /// Resolution order:
    ///  1. A `Def` whose `FuncId` carries a `def_index`: use the binding-table
    ///     range via `resolve_func_range`.
    ///  2. Otherwise, resolve the function by `(module, name)` through the
    ///     export-location resolver. This covers imported user functions whose
    ///     `FuncId` lacks a `def_index`, and special functions that are not
    ///     `Def` at all (e.g. `typing.overload`).
    ///  3. Fall back to a zero range pointing at the defining module (for
    ///     `Def`), or a synthesized declaration when even the module is unknown.
    fn function_declaration(&self, kind: &FunctionKind) -> Declaration {
        if let FunctionKind::Def(func_id) = kind
            && let Some(range) = self.resolve_func_range.and_then(|resolve| resolve(func_id))
        {
            let lsp_range = func_id.module.to_lsp_range(range);
            return Declaration::Regular(RegularDeclaration {
                category: DeclarationCategory::Function,
                kind: DeclarationKind::Regular,
                name: Some(func_id.name.to_string()),
                node: Node {
                    range: lsp_range_to_tsp(lsp_range),
                    uri: path_to_uri(func_id.module.path()),
                },
            });
        }

        let name = kind.function_name();
        if let Some((module_path, lsp_range)) = self
            .resolve_export
            .and_then(|resolve| resolve(kind.module_name(), name.as_ref()))
        {
            return Declaration::Regular(RegularDeclaration {
                category: DeclarationCategory::Function,
                kind: DeclarationKind::Regular,
                name: Some(name.to_string()),
                node: Node {
                    range: lsp_range_to_tsp(lsp_range),
                    uri: path_to_uri(&module_path),
                },
            });
        }

        if let FunctionKind::Def(func_id) = kind {
            return Declaration::Regular(RegularDeclaration {
                category: DeclarationCategory::Function,
                kind: DeclarationKind::Regular,
                name: Some(func_id.name.to_string()),
                node: Node {
                    range: zero_range(),
                    uri: path_to_uri(func_id.module.path()),
                },
            });
        }

        Declaration::Synthesized(SynthesizedDeclaration {
            kind: DeclarationKind::Synthesized,
            uri: String::new(),
        })
    }

    /// Build a TSP `ClassType` whose declaration points at `typing.<name>`,
    /// resolving the real definition range from the typeshed when possible.
    /// Used for `SpecialForm`, anonymous `TypedDict`, `LiteralString`,
    /// `Concatenate`, `ParamSpec`, and `TypeAlias::Ref` where pyrefly does not
    /// have an explicit `Class` backing but the consumer needs a typed handle
    /// it can render and resolve via the typeshed.
    fn typing_class(&self, name: &str, flags: TypeFlags) -> TspType {
        TspType::Class(TspClassType {
            declaration: Declaration::Regular(self.typing_class_declaration(name)),
            flags,
            id: next_id(),
            kind: TypeKind::Class,
            literal_value: None,
            type_alias_info: None,
            type_args: None,
        })
    }

    /// Build a class declaration for `typing.<name>`. Resolves the real source
    /// range via the export-location resolver; falls back to a zero range
    /// pointing at bundled `typing.pyi` when unavailable.
    fn typing_class_declaration(&self, name: &str) -> RegularDeclaration {
        let symbol = Name::new(name);
        if let Some((module_path, lsp_range)) = self
            .resolve_export
            .and_then(|resolve| resolve(ModuleName::typing(), &symbol))
        {
            return RegularDeclaration {
                kind: DeclarationKind::Regular,
                category: DeclarationCategory::Class,
                name: Some(name.to_owned()),
                node: Node {
                    range: lsp_range_to_tsp(lsp_range),
                    uri: path_to_uri(&module_path),
                },
            };
        }
        make_typing_class_declaration(name)
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

/// Force the `INSTANTIABLE` flag on any TSP type variant, overwriting other
/// flags (e.g. clearing `INSTANCE`). Used for `type[X]`, whose inner type may
/// convert to any TSP shape but always denotes a class object. Exhaustive so
/// the compiler flags new variants.
fn mark_instantiable(mut ty: TspType) -> TspType {
    match &mut ty {
        TspType::BuiltInType(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Declared(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Function(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Class(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Union(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Module(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Var(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Overloaded(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Synthesized(t) => t.flags = TypeFlags::INSTANTIABLE,
        TspType::Reference(t) => t.flags = TypeFlags::INSTANTIABLE,
    }
    ty
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
                    declaration: Declaration::Regular(make_builtin_class_declaration(class_name)),
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

/// Convert a `Sentinel` to a TSP `ClassType` carrying a `SentinelLiteral`. The
/// sentinel's `QName` supplies both the enclosing class declaration and the
/// literal's class name and defining location.
fn convert_sentinel(sentinel: &Sentinel) -> TspType {
    let qname = sentinel.qname();
    let node = Node {
        range: lsp_range_to_tsp(qname.module().to_lsp_range(qname.range())),
        uri: path_to_uri(qname.module_path()),
    };
    TspType::Class(TspClassType {
        declaration: Declaration::Regular(RegularDeclaration {
            category: DeclarationCategory::Class,
            kind: DeclarationKind::Regular,
            name: Some(qname.id().to_string()),
            node: node.clone(),
        }),
        flags: TypeFlags::INSTANCE.with_literal(),
        id: next_id(),
        kind: TypeKind::Class,
        literal_value: Some(LiteralValue::Sentinel(SentinelLiteral {
            class_name: qname.id().to_string(),
            class_node: node,
            module_name: qname.module_name().to_string(),
        })),
        type_alias_info: None,
        type_args: None,
    })
}

/// Build a declaration for a class in `builtins.pyi`.
fn make_builtin_class_declaration(name: &str) -> RegularDeclaration {
    let module_path =
        pyrefly_python::module_path::ModulePath::bundled_typeshed(PathBuf::from("builtins.pyi"));
    RegularDeclaration {
        kind: DeclarationKind::Regular,
        category: DeclarationCategory::Class,
        name: Some(name.to_owned()),
        node: Node {
            range: zero_range(),
            uri: path_to_uri(&module_path),
        },
    }
}

/// Build a declaration for a class in `typing.pyi`.
fn make_typing_class_declaration(name: &str) -> RegularDeclaration {
    let module_path =
        pyrefly_python::module_path::ModulePath::bundled_typeshed(PathBuf::from("typing.pyi"));
    RegularDeclaration {
        kind: DeclarationKind::Regular,
        category: DeclarationCategory::Class,
        name: Some(name.to_owned()),
        node: Node {
            range: zero_range(),
            uri: path_to_uri(&module_path),
        },
    }
}

/// Build a TSP `TypeVar` with a synthesized declaration. Used for
/// solver-internal TypeVar-like placeholders (Quantified, Args, Kwargs,
/// ElementOfTypeVarTuple) where there is no real source location.
fn synthesized_typevar(name: &str) -> TspType {
    TspType::Var(DeclaredType {
        declaration: Declaration::Regular(RegularDeclaration {
            category: DeclarationCategory::Typeparam,
            kind: DeclarationKind::Regular,
            name: Some(name.to_owned()),
            node: Node {
                range: zero_range(),
                uri: String::new(),
            },
        }),
        flags: TypeFlags::NONE,
        id: next_id(),
        kind: TypeKind::Typevar,
        type_alias_info: None,
    })
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

/// Convert a local filesystem path to a URI string.
fn path_buf_to_uri(path: &std::path::Path) -> String {
    Url::from_file_path(path)
        .map_or_else(|()| path.to_string_lossy().to_string(), |u| u.to_string())
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
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_types::callable::FuncFlags;
    use pyrefly_types::callable::FuncMetadata;
    use pyrefly_types::callable::Function;
    use pyrefly_types::callable::Param;
    use pyrefly_types::callable::ParamList;
    use pyrefly_types::callable::Required;
    use pyrefly_types::lit_int::LitInt;
    use pyrefly_types::literal::Lit;
    use pyrefly_types::literal::LitStyle;
    use pyrefly_types::module::ModuleType;
    use pyrefly_types::quantified::AnchorIndex;
    use pyrefly_types::quantified::QuantifiedIdentity;
    use pyrefly_types::special_form::SpecialForm;
    use pyrefly_types::type_alias::TypeAliasIndex;
    use pyrefly_types::type_var::PreInferenceVariance;
    use pyrefly_types::type_var::Restriction;
    use pyrefly_types::types::AnyStyle;
    use pyrefly_types::types::NeverStyle;
    use pyrefly_types::types::Type as PyreflyType;
    use pyrefly_types::types::Var;
    use tsp_types::SynthesizedType;
    use tsp_types::SynthesizedTypeMetadata;

    use super::*;

    /// Build a `Quantified` (solver-internal TypeVar placeholder) anchored at
    /// `module` with the given `origin`. Used by the conversion tests.
    fn make_quantified(name: &str, module: &str, origin: QuantifiedOrigin) -> Quantified {
        let identity = QuantifiedIdentity::new(
            ModuleName::from_str(module),
            AnchorIndex::first(TextRange::default()),
            origin,
        );
        Quantified::type_var(
            Name::new(name),
            identity,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

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
            TspType::Class(c) => {
                assert!(c.flags.contains(TypeFlags::INSTANCE));
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("NoneType"));
                assert_eq!(decl.category, DeclarationCategory::Class);
                assert!(
                    decl.node.uri.contains("types.pyi"),
                    "expected types URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class, got {other:?}"),
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
    fn test_convert_type_guard_and_type_is_are_bool_class() {
        // `TypeGuard`/`TypeIs` erase to their runtime type `bool`, emitted as
        // the real `bool` class rather than an off-spec `bool` `BuiltInType`.
        for ty in [
            PyreflyType::TypeGuard(Box::new(PyreflyType::None)),
            PyreflyType::TypeIs(Box::new(PyreflyType::None)),
        ] {
            match convert_type(&ty) {
                TspType::Class(c) => {
                    assert!(c.flags.contains(TypeFlags::INSTANCE));
                    let Declaration::Regular(decl) = c.declaration else {
                        panic!("expected RegularDeclaration");
                    };
                    assert_eq!(decl.name.as_deref(), Some("bool"));
                    assert_eq!(decl.category, DeclarationCategory::Class);
                }
                other => panic!("expected bool Class, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_convert_int_is_int_class() {
        use pyrefly_types::dimension::Int;

        // A `Int` is an integer tensor dimension, emitted as the real `int`
        // class rather than an off-spec `int` `BuiltInType`.
        for ty in [PyreflyType::Int(Int::literal(6))] {
            match convert_type(&ty) {
                TspType::Class(c) => {
                    assert!(c.flags.contains(TypeFlags::INSTANCE));
                    let Declaration::Regular(decl) = c.declaration else {
                        panic!("expected RegularDeclaration");
                    };
                    assert_eq!(decl.name.as_deref(), Some("int"));
                    assert_eq!(decl.category, DeclarationCategory::Class);
                }
                other => panic!("expected int Class, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_convert_sentinel_is_sentinel_literal_class() {
        use pyrefly_python::module::Module;
        use pyrefly_python::nesting_context::NestingContext;
        use ruff_python_ast::Identifier;

        // A sentinel is emitted as a `ClassType` carrying a `SentinelLiteral`
        // sourced from its own definition, not an off-spec `sentinel`
        // `BuiltInType`.
        let module = Module::new(
            ModuleName::from_str("dataclasses"),
            ModulePath::bundled_typeshed(PathBuf::from("dataclasses.pyi")),
            Arc::new(String::new()),
        );
        let sentinel = Sentinel::new(
            Identifier::new(Name::new("_MISSING_TYPE"), TextRange::default()),
            NestingContext::toplevel(),
            module,
        );
        match convert_type(&PyreflyType::Sentinel(sentinel)) {
            TspType::Class(c) => {
                let Some(LiteralValue::Sentinel(lit)) = c.literal_value else {
                    panic!("expected SentinelLiteral, got {:?}", c.literal_value);
                };
                assert_eq!(lit.class_name, "_MISSING_TYPE");
                assert_eq!(lit.module_name, "dataclasses");
                assert!(
                    lit.class_node.uri.contains("dataclasses.pyi"),
                    "expected dataclasses URI, got {}",
                    lit.class_node.uri
                );
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("_MISSING_TYPE"));
            }
            other => panic!("expected sentinel Class, got {other:?}"),
        }
    }

    #[test]
    fn test_unique_ids() {
        let a = convert_type(&PyreflyType::None);
        let b = convert_type(&PyreflyType::Ellipsis);
        let id_a = match &a {
            TspType::Class(c) => c.id,
            _ => panic!("expected Class"),
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
                // First member should be `types.NoneType`.
                match &u.sub_types[0] {
                    TspType::Class(c) => {
                        let Declaration::Regular(decl) = &c.declaration else {
                            panic!("expected RegularDeclaration");
                        };
                        assert_eq!(decl.name.as_deref(), Some("NoneType"));
                    }
                    other => panic!("expected Class for first member, got {other:?}"),
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
        let ty = PyreflyType::type_of(PyreflyType::Any(AnyStyle::Explicit));
        let tsp = convert_type(&ty);
        // Any wrapped in Type() — inner is BuiltIn, not Class, so it passes through unchanged
        match tsp {
            TspType::BuiltInType(b) => assert_eq!(b.name, "any"),
            other => panic!("expected BuiltInType pass-through, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_type_of_typevar_is_instantiable() {
        // `type[T]` where T is a TypeVar must stay instantiable. The inner
        // TypeVar converts to a `TspType::Var`, and the `Type(inner)` arm must
        // propagate the INSTANTIABLE flag onto it rather than dropping it.
        let tv = make_quantified("T", "mod", QuantifiedOrigin::Pep695);
        let ty = PyreflyType::type_of(PyreflyType::Quantified(Box::new(tv)));
        let tsp = convert_type(&ty);
        match tsp {
            TspType::Var(v) => assert!(
                v.flags.contains(TypeFlags::INSTANTIABLE),
                "type[T] should be instantiable, got flags {:?}",
                v.flags
            ),
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_module_without_resolver_has_empty_uri() {
        let ty = PyreflyType::Module(ModuleType::new_as(ModuleName::from_str("pkg")));
        let tsp = convert_type(&ty);
        match tsp {
            TspType::Module(m) => {
                assert_eq!(m.module_name, "pkg");
                assert_eq!(m.uri, "");
            }
            other => panic!("expected ModuleType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_module_with_resolver_sets_uri() {
        let ty = PyreflyType::Module(ModuleType::new_as(ModuleName::from_str("pkg")));
        let module_path_resolver = |module: &ModuleType| {
            if module.to_string() == "pkg" {
                Some(PathBuf::from("/repo/pkg/__init__.pyi"))
            } else {
                None
            }
        };
        let stdlib = TestStdlib::new();
        let tsp = convert_type_with_resolvers(
            &ty,
            None,
            Some(&module_path_resolver),
            None,
            stdlib.classes(),
        );
        match tsp {
            TspType::Module(m) => {
                assert_eq!(m.module_name, "pkg");
                assert!(m.uri.contains("/repo/pkg/__init__.pyi"));
            }
            other => panic!("expected ModuleType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_literal_int_uses_builtins_uri() {
        let ty = Lit::Int(LitInt::new(7)).to_implicit_type();
        let tsp = convert_type(&ty);
        match tsp {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("int"));
                assert!(
                    decl.node.uri.contains("builtins.pyi"),
                    "expected builtins URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class type, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_special_form_emits_typing_class() {
        // SpecialForm must map to a typing.<name> ClassType, not a BuiltIn:
        // the TSP BuiltInType.name is restricted to a fixed sentinel set.
        let ty = PyreflyType::SpecialForm(SpecialForm::Literal);
        match convert_type(&ty) {
            TspType::Class(c) => {
                assert_eq!(c.kind, TypeKind::Class);
                assert!(c.flags.contains(TypeFlags::INSTANTIABLE));
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("Literal"));
                assert_eq!(decl.category, DeclarationCategory::Class);
                assert!(
                    decl.node.uri.contains("typing.pyi"),
                    "expected typing URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_special_form_uses_export_resolver_location() {
        // When an export resolver is available, the typing class declaration
        // should carry the resolved source location instead of a zero range.
        let ty = PyreflyType::SpecialForm(SpecialForm::Final);
        let range = lsp_types::Range {
            start: lsp_types::Position {
                line: 99,
                character: 0,
            },
            end: lsp_types::Position {
                line: 99,
                character: 5,
            },
        };
        let resolver = |module: ModuleName, name: &Name| {
            assert_eq!(module, ModuleName::typing());
            assert_eq!(name.as_str(), "Final");
            Some((
                ModulePath::filesystem(PathBuf::from("/typeshed/typing.pyi")),
                range,
            ))
        };
        match convert_type_with_resolvers(
            &ty,
            None,
            None,
            Some(&resolver),
            TestStdlib::new().classes(),
        ) {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.node.range.start.line, 99);
                assert!(
                    decl.node.uri.contains("typing.pyi"),
                    "expected typing URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_literal_string_emits_typing_class() {
        let ty = PyreflyType::LiteralString(LitStyle::Implicit);
        match convert_type(&ty) {
            TspType::Class(c) => {
                assert!(c.flags.contains(TypeFlags::INSTANCE));
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("LiteralString"));
                assert!(
                    decl.node.uri.contains("typing.pyi"),
                    "expected typing URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_param_spec_value_emits_typing_param_spec_class() {
        let ty = PyreflyType::ParamSpecValue(ParamList::new(vec![]));
        match convert_type(&ty) {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("ParamSpec"));
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_concatenate_emits_typing_class() {
        let ty =
            PyreflyType::Concatenate(Vec::new().into_boxed_slice(), Box::new(PyreflyType::None));
        match convert_type(&ty) {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("Concatenate"));
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_type_alias_ref_resolves_against_own_module_not_typing() {
        let make_ref = || TypeAliasRef {
            name: Name::new_static("MyAlias"),
            args: None,
            module_name: ModuleName::from_str("mymod"),
            module_path: ModulePath::filesystem(PathBuf::from("/repo/mymod.py")),
            index: TypeAliasIndex(0),
        };

        // Without a resolver, the fallback points at the alias's own module
        // file (not `typing.pyi`) — the correct file, just a zero range.
        let ty = PyreflyType::TypeAlias(Box::new(TypeAliasData::Ref(make_ref())));
        match convert_type(&ty) {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("MyAlias"));
                assert!(
                    decl.node.uri.contains("mymod.py"),
                    "expected the alias's own module URI, got {}",
                    decl.node.uri
                );
                assert!(
                    !decl.node.uri.contains("typing"),
                    "must not fall back to typing, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Class, got {other:?}"),
        }

        // With a resolver, the declaration carries the real definition range
        // resolved against the alias's defining module.
        let range = lsp_types::Range {
            start: lsp_types::Position {
                line: 7,
                character: 5,
            },
            end: lsp_types::Position {
                line: 7,
                character: 12,
            },
        };
        let resolver = |module: ModuleName, name: &Name| {
            assert_eq!(module, ModuleName::from_str("mymod"));
            assert_eq!(name.as_str(), "MyAlias");
            Some((
                ModulePath::filesystem(PathBuf::from("/repo/mymod.py")),
                range,
            ))
        };
        let ty = PyreflyType::TypeAlias(Box::new(TypeAliasData::Ref(make_ref())));
        match convert_type_with_resolvers(
            &ty,
            None,
            None,
            Some(&resolver),
            TestStdlib::new().classes(),
        ) {
            TspType::Class(c) => {
                let Declaration::Regular(decl) = c.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.node.range.start.line, 7);
                assert!(decl.node.uri.contains("mymod.py"));
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_quantified_without_resolver_is_synthesized_typevar() {
        // No export resolver → a locationless synthesized TypeVar.
        let ty = PyreflyType::Quantified(Box::new(make_quantified(
            "T",
            "mod",
            QuantifiedOrigin::Pep695,
        )));
        match convert_type(&ty) {
            TspType::Var(v) => {
                assert_eq!(v.kind, TypeKind::Typevar);
                let Declaration::Regular(decl) = v.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("T"));
                assert_eq!(decl.category, DeclarationCategory::Typeparam);
                assert_eq!(decl.node.uri, "");
            }
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_pep695_quantified_with_resolver_uses_source_location() {
        // A PEP 695 type parameter resolves to a Typeparam declaration at the
        // resolved source location.
        let ty = PyreflyType::Quantified(Box::new(make_quantified(
            "T",
            "mymod",
            QuantifiedOrigin::Pep695,
        )));
        let range = lsp_types::Range {
            start: lsp_types::Position {
                line: 3,
                character: 6,
            },
            end: lsp_types::Position {
                line: 3,
                character: 7,
            },
        };
        let resolver = |module: ModuleName, name: &Name| {
            assert_eq!(module, ModuleName::from_str("mymod"));
            assert_eq!(name.as_str(), "T");
            Some((
                ModulePath::filesystem(PathBuf::from("/repo/mymod.py")),
                range,
            ))
        };
        match convert_type_with_resolvers(
            &ty,
            None,
            None,
            Some(&resolver),
            TestStdlib::new().classes(),
        ) {
            TspType::Var(v) => {
                let Declaration::Regular(decl) = v.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.category, DeclarationCategory::Typeparam);
                assert_eq!(decl.node.range.start.line, 3);
                assert!(
                    decl.node.uri.contains("mymod.py"),
                    "expected resolved URI, got {}",
                    decl.node.uri
                );
            }
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_legacy_quantified_with_resolver_is_variable_category() {
        // A legacy `T = TypeVar("T")` resolves to a module-level *variable*.
        let ty = PyreflyType::Quantified(Box::new(make_quantified(
            "T",
            "mymod",
            QuantifiedOrigin::ScopedLegacy,
        )));
        let resolver = |_module: ModuleName, _name: &Name| {
            Some((
                ModulePath::filesystem(PathBuf::from("/repo/mymod.py")),
                lsp_types::Range::default(),
            ))
        };
        match convert_type_with_resolvers(
            &ty,
            None,
            None,
            Some(&resolver),
            TestStdlib::new().classes(),
        ) {
            TspType::Var(v) => {
                let Declaration::Regular(decl) = v.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.category, DeclarationCategory::Variable);
            }
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_args_is_synthesized_typevar() {
        // ParamSpec-related internal placeholders become synthesized TypeVars.
        let ty = PyreflyType::Args(Box::new(make_quantified(
            "P",
            "m",
            QuantifiedOrigin::ScopedLegacy,
        )));
        match convert_type(&ty) {
            TspType::Var(v) => {
                assert_eq!(v.kind, TypeKind::Typevar);
                let Declaration::Regular(decl) = v.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("P"));
                assert_eq!(decl.category, DeclarationCategory::Typeparam);
            }
            other => panic!("expected Var, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_function_populates_specialized_types() {
        // A function's parameter and return types are carried in
        // specialized_types so Pylance can overlay real types.
        let callable = Callable::list(
            ParamList::new(vec![
                Param::Pos(Name::new_static("x"), PyreflyType::None, Required::Required),
                Param::Pos(
                    Name::new_static("y"),
                    PyreflyType::Ellipsis,
                    Required::Required,
                ),
            ]),
            PyreflyType::None,
        );
        let func = Function {
            signature: callable,
            metadata: FuncMetadata {
                kind: FunctionKind::Overload,
                flags: FuncFlags::default(),
            },
        };
        let ty = PyreflyType::Function(Box::new(func));
        match convert_type(&ty) {
            TspType::Function(f) => {
                let specialized = f.specialized_types.expect("expected specialized_types");
                assert_eq!(specialized.parameter_types.len(), 2);
                match &specialized.parameter_types[0] {
                    TspType::Class(c) => {
                        let Declaration::Regular(decl) = &c.declaration else {
                            panic!("expected RegularDeclaration");
                        };
                        assert_eq!(decl.name.as_deref(), Some("NoneType"));
                    }
                    other => panic!("expected Class, got {other:?}"),
                }
                match &specialized.parameter_types[1] {
                    TspType::BuiltInType(b) => assert_eq!(b.name, "ellipsis"),
                    other => panic!("expected BuiltInType, got {other:?}"),
                }
                match specialized.return_type.as_deref() {
                    Some(TspType::Class(c)) => {
                        let Declaration::Regular(decl) = &c.declaration else {
                            panic!("expected RegularDeclaration");
                        };
                        assert_eq!(decl.name.as_deref(), Some("NoneType"));
                    }
                    other => panic!("expected NoneType Class return type, got {other:?}"),
                }
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_callable_populates_specialized_types() {
        // A `typing.Callable`'s parameter and return types are carried in
        // `specialized_types`, mirroring `convert_function`. For a return
        // annotation like `Callable[[int], str]`, the consumer can recover the
        // parameter types instead of rendering them as Unknown/Any.
        let callable = Callable::list(
            ParamList::new(vec![Param::Pos(
                Name::new_static("a"),
                PyreflyType::None,
                Required::Required,
            )]),
            PyreflyType::Ellipsis,
        );
        let ty = PyreflyType::Callable(Box::new(callable));
        match convert_type(&ty) {
            TspType::Function(f) => {
                assert!(f.flags.contains(TypeFlags::CALLABLE));
                assert!(f.return_type.is_some(), "return type should be preserved");
                let specialized = f
                    .specialized_types
                    .expect("Callable parameter types must be carried in specialized_types");
                assert_eq!(specialized.parameter_types.len(), 1);
                match &specialized.parameter_types[0] {
                    TspType::Class(c) => {
                        let Declaration::Regular(decl) = &c.declaration else {
                            panic!("expected RegularDeclaration");
                        };
                        assert_eq!(decl.name.as_deref(), Some("NoneType"));
                    }
                    other => panic!("expected Class, got {other:?}"),
                }
                assert!(specialized.return_type.is_some());
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn test_function_declaration_resolves_special_function_via_export() {
        // `typing.overload` is FunctionKind::Overload (not a Def). The export
        // resolver should give it a real declaration location.
        let callable = Callable::list(ParamList::new(vec![]), PyreflyType::None);
        let func = Function {
            signature: callable,
            metadata: FuncMetadata {
                kind: FunctionKind::Overload,
                flags: FuncFlags::default(),
            },
        };
        let ty = PyreflyType::Function(Box::new(func));
        let range = lsp_types::Range {
            start: lsp_types::Position {
                line: 12,
                character: 4,
            },
            end: lsp_types::Position {
                line: 12,
                character: 12,
            },
        };
        // The `None` return converts through the stdlib `NoneType` class, not
        // the export resolver, so the only lookup here is `typing.overload`;
        // any other lookup is a regression.
        let resolver = |module: ModuleName, name: &Name| {
            if module == ModuleName::typing() && name.as_str() == "overload" {
                return Some((
                    ModulePath::filesystem(PathBuf::from("/typeshed/typing.pyi")),
                    range,
                ));
            }
            panic!("unexpected export lookup for {module}.{name}");
        };
        match convert_type_with_resolvers(
            &ty,
            None,
            None,
            Some(&resolver),
            TestStdlib::new().classes(),
        ) {
            TspType::Function(f) => {
                let Declaration::Regular(decl) = f.declaration else {
                    panic!("expected RegularDeclaration");
                };
                assert_eq!(decl.name.as_deref(), Some("overload"));
                assert_eq!(decl.category, DeclarationCategory::Function);
                assert_eq!(decl.node.range.start.line, 12);
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_var_is_lowercase_unknown() {
        // The protocol-conformant sentinel name is lowercase `unknown`.
        match convert_type(&PyreflyType::Var(Var::ZERO)) {
            TspType::BuiltInType(b) => assert_eq!(b.name, "unknown"),
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_materialization_is_lowercase_unknown() {
        match convert_type(&PyreflyType::Materialization) {
            TspType::BuiltInType(b) => assert_eq!(b.name, "unknown"),
            other => panic!("expected BuiltInType, got {other:?}"),
        }
    }
}
