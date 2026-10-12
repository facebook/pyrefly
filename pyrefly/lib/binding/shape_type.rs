/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::keywords::is_valid_identifier;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::dimension::gradual_size;
use pyrefly_types::meta_shape_dsl::ShapeDslFunction;
use pyrefly_types::meta_shape_dsl::convert_shape_dsl_function;
use pyrefly_types::quantified::AnchorIndex;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedIdentity;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::quantified::QuantifiedOrigin;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::type_level_dsl::ParsedTypeShapeDslFunction;
use pyrefly_types::type_var::PreInferenceVariance;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Type;
use ruff_python_ast::Decorator;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::Identifier;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_map::SmallMap;

use crate::binding::binding::KeyClass;
use crate::binding::binding::LambdaKind;
use crate::binding::binding::ShapedArrayMetadata;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::bindings::LegacyTParamCollector;
use crate::binding::expr::Usage;
use crate::config::error_kind::ErrorKind;
use crate::export::special::SpecialExport;

#[derive(Clone, Debug)]
pub enum TypeParameterBound {
    Ordinary(Expr),
    ShapeFlag {
        domain: Option<Expr>,
        range: TextRange,
    },
    ShapeIndex,
    ShapeNamedInts,
}

impl TypeParameterBound {
    /// Selects value-expression inference while syntax is being bound. The resolved restriction
    /// exposes the same policy later, after the bound has become a semantic `Restriction`.
    pub fn infer_default_as_value(&self) -> bool {
        matches!(self, Self::ShapeFlag { .. } | Self::ShapeIndex)
    }
}

/// Which decorator declared a scope.
///
/// Each kind of shape string resolves only against declarations of its own kind.
/// A jaxtyping shape string is entirely closed -- every bare token is a
/// dimension -- while in a `Shaped` string, names that `@shape_vars` does not
/// declare resolve through ordinary scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDeclarationKind {
    Jaxtyping,
    ShapeVars,
}

impl ShapeDeclarationKind {
    /// The decorator's spelling, for diagnostics.
    pub fn decorator(self) -> &'static str {
        match self {
            Self::Jaxtyping => "@static_jaxtyping",
            Self::ShapeVars => "@shape_vars",
        }
    }

    /// An example declaration, for diagnostics.
    fn example(self) -> &'static str {
        match self {
            Self::Jaxtyping => "@static_jaxtyping(\"batch channels\")",
            Self::ShapeVars => "@shape_vars(\"batch, channels\")",
        }
    }

    /// `@static_jaxtyping("a b")` versus `@shape_vars("a, b")`.
    fn split(self, declaration: &str) -> impl Iterator<Item = &str> {
        declaration
            .split(move |c: char| match self {
                Self::Jaxtyping => c.is_whitespace(),
                Self::ShapeVars => c == ',',
            })
            .map(str::trim)
            .filter(|token| !token.is_empty())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDeclarationOwner {
    Function(Option<Idx<KeyClass>>),
    Class(Idx<KeyClass>),
}

/// The names a definition declares with `@static_jaxtyping` or `@shape_vars`.
///
/// Recording the declaration at binding time is what lets shape strings be
/// resolved by lookup rather than discovery: every name a shape string may
/// mention is known before any annotation is solved.
#[derive(Clone, Debug)]
pub struct ShapeDeclaration {
    kind: ShapeDeclarationKind,
    dims: Box<[Quantified]>,
    /// Range of the name carrying the declaration. Definitions that inherit a
    /// scope use this to distinguish it from their own declaration.
    declared_at: TextRange,
    owner: ShapeDeclarationOwner,
}

impl ShapeDeclaration {
    pub fn kind(&self) -> ShapeDeclarationKind {
        self.kind
    }

    pub fn dims(&self) -> &[Quantified] {
        &self.dims
    }

    fn dim(&self, name: &Name) -> Option<&Quantified> {
        self.dims.iter().find(|dim| dim.name() == name)
    }
}

/// Shape declarations and class boundaries, used to resolve shape strings and
/// record the `Shaped` annotations whose strings were parsed.
#[derive(Clone, Debug, Default)]
pub struct ShapeDeclarations {
    scopes: Vec<(TextRange, Arc<ShapeDeclaration>)>,
    classes: Vec<(TextRange, Idx<KeyClass>)>,
    /// Binding replaces shape strings with parsed syntax, so keep their original
    /// metadata for aliases that must represent the runtime `Annotated` value.
    shaped_annotations: SmallMap<TextRange, Box<str>>,
}

impl ShapeDeclarations {
    pub fn finish(mut self) -> Self {
        if self.scopes.is_empty() {
            self.classes.clear();
        }
        self
    }

    pub fn push_class(&mut self, range: TextRange, class: Idx<KeyClass>) {
        self.classes.push((range, class));
    }

    fn push_scope(&mut self, range: TextRange, scope: ShapeDeclaration) {
        self.scopes.push((range, Arc::new(scope)));
    }

    fn enclosing_class(&self, range: TextRange) -> Option<Idx<KeyClass>> {
        // These ranges cover whole class statements, unlike `Bindings::class_scopes`,
        // because annotations in bases and type parameters are evaluated in the
        // enclosing scope but may refer to the class's own declaration.
        self.classes
            .iter()
            .rev()
            .find(|(class_range, _)| class_range.contains_range(range))
            .map(|(_, class)| *class)
    }

    /// The declarations in scope at `range`, innermost first.
    fn enclosing(&self, range: TextRange) -> impl Iterator<Item = &ShapeDeclaration> {
        let enclosing_class = self.enclosing_class(range);
        self.scopes
            .iter()
            .rev()
            .filter_map(move |(scope_range, scope)| {
                let owner_class = match scope.owner {
                    ShapeDeclarationOwner::Function(owner) => owner,
                    ShapeDeclarationOwner::Class(owner) => Some(owner),
                };
                (scope_range.contains_range(range) && owner_class == enclosing_class)
                    .then_some(scope.as_ref())
            })
    }

    pub fn contains(&self, range: TextRange, kind: ShapeDeclarationKind) -> bool {
        self.enclosing(range).any(|scope| scope.kind == kind)
    }

    pub fn resolve(
        &self,
        range: TextRange,
        kind: ShapeDeclarationKind,
        name: &Name,
    ) -> Option<&Quantified> {
        self.enclosing(range)
            .filter(|scope| scope.kind == kind)
            .find_map(|scope| scope.dim(name))
    }

    /// The declaration carried by the definition whose name is at `range`.
    ///
    /// A dimension inherited from an enclosing definition is bound there, and
    /// binding it again here would produce a second variable of the same name
    /// that no longer substitutes against the first.
    pub fn declared_by(&self, range: TextRange) -> Option<&ShapeDeclaration> {
        self.scopes
            .iter()
            .find(|(_, scope)| scope.declared_at == range)
            .map(|(_, scope)| scope.as_ref())
    }

    pub fn is_shaped_annotation(&self, range: TextRange) -> bool {
        self.shaped_annotations.contains_key(&range)
    }

    pub fn shaped_annotation_metadata(&self, range: TextRange) -> Option<&str> {
        self.shaped_annotations.get(&range).map(Box::as_ref)
    }
}

pub(super) struct ShapeFunctionMetadata {
    pub shape_dsl_def: Option<Arc<ShapeDslFunction>>,
    pub type_shape_dsl_def: Option<Arc<ParsedTypeShapeDslFunction>>,
    pub uses_shape_dsl_ir_name: Option<ShortIdentifier>,
}

impl BindingsBuilder<'_> {
    /// Binds the arguments of the experimental `shape_extensions.MapIntTuples` operation.
    ///
    /// The first argument's lambda parameter denotes a type rather than a runtime value. Record
    /// that shape-specific meaning while import provenance is still available; all other
    /// arguments follow ordinary type-expression binding.
    pub(super) fn bind_map_int_tuples_arguments(
        &mut self,
        slice: &mut Expr,
        mut tparams_builder: Option<&mut LegacyTParamCollector>,
        in_string_literal: bool,
        usage: &mut Usage,
    ) {
        let arguments = match slice {
            Expr::Tuple(tuple) => tuple.elts.as_mut_slice(),
            single => std::slice::from_mut(single),
        };
        for (index, argument) in arguments.iter_mut().enumerate() {
            if index == 0 && argument.is_lambda_expr() {
                self.with_semantic_checker(|semantic, context| {
                    semantic.visit_expr(argument, context)
                });
                let lambda = argument
                    .as_lambda_expr_mut()
                    .expect("is_lambda_expr established that this is a lambda");
                self.bind_lambda(lambda, usage, LambdaKind::TypeLevel);
            } else {
                self.ensure_type_impl(
                    argument,
                    tparams_builder.as_deref_mut(),
                    in_string_literal,
                    true,
                    usage,
                    false,
                );
            }
        }
    }

    /// Binds the metadata of `Shaped[T, "<shape>"]`, whose `Shaped` and `T` are
    /// already bound, when the annotation is inside a `@shape_vars` scope.
    ///
    /// The shape string is parsed and bound in its place, with its bare names
    /// resolving against the `@shape_vars` declarations before ordinary scopes.
    /// Returns `false` outside a `@shape_vars` scope, or after reporting a
    /// malformed annotation; the caller then binds `Shaped` as `Annotated`.
    pub(super) fn bind_shape_string(
        &mut self,
        annotation: TextRange,
        arguments: &mut ExprTuple,
        tparams_builder: Option<&mut LegacyTParamCollector>,
        usage: &mut Usage,
    ) -> bool {
        if !self
            .shape_declarations
            .contains(annotation, ShapeDeclarationKind::ShapeVars)
        {
            return false;
        }
        let shape = match arguments.elts.as_slice() {
            [_, Expr::StringLiteral(shape)] => shape,
            _ => {
                self.error(
                    arguments.range(),
                    ErrorKind::InvalidAnnotation,
                    "`Shaped` takes a type and a shape string, as in `Shaped[ndarray, \"[M, N]\"]`"
                        .to_owned(),
                );
                return false;
            }
        };
        // A shape string made of several parts would put errors inside it at the
        // wrong position, so, as with forward references, only one part is allowed.
        let Some(shape_literal) = shape.as_single_part_string() else {
            self.error(
                shape.range(),
                ErrorKind::InvalidAnnotation,
                "A `Shaped` shape string must be a single string literal, not several concatenated ones"
                    .to_owned(),
            );
            return false;
        };
        let metadata = Box::<str>::from(shape_literal.as_str());
        let Ok(parsed) = Ast::parse_type_literal(shape, self.module_info.contents()) else {
            self.error(
                shape.range(),
                ErrorKind::InvalidAnnotation,
                "Could not parse `Shaped` shape string".to_owned(),
            );
            return false;
        };
        arguments.elts[1] = parsed;
        self.shape_declarations
            .shaped_annotations
            .insert(annotation, metadata);
        let outer = self.shape_string_annotation.replace(annotation);
        self.ensure_type_impl(
            &mut arguments.elts[1],
            tparams_builder,
            true,
            false,
            usage,
            false,
        );
        self.shape_string_annotation = outer;
        true
    }

    /// The `@shape_vars` declaration of `name`, if it is a bare name inside a
    /// `Shaped` shape string.
    pub(super) fn shape_var(&self, name: &Name) -> Option<Quantified> {
        self.shape_declarations
            .resolve(
                self.shape_string_annotation?,
                ShapeDeclarationKind::ShapeVars,
                name,
            )
            .cloned()
    }

    /// Recognizes a shape-extension class through import provenance or in its defining module.
    ///
    /// The local-name case is needed while binding `shape_extensions` itself: its class has no
    /// import provenance, so we additionally require the module-level name to resolve to that
    /// class definition. Consumers use `SpecialExport` provenance, including through aliases and
    /// re-exports, rather than relying on a raw imported name.
    fn is_shape_extensions_class_export_with_provenance(
        &self,
        expr: &Expr,
        special: SpecialExport,
        provenance: Option<SpecialExport>,
    ) -> bool {
        if let Expr::Name(name) = expr
            && SpecialExport::new(&name.id) == Some(special)
            && special.defined_in(self.module_info.name())
        {
            return self.scopes.current_binding_is_module_binding(&name.id)
                && matches!(
                    self.scopes.binding_idx_for_name(&name.id),
                    Some((idx, _)) if self.binding_is_class_def(idx)
                );
        }
        provenance == Some(special)
    }

    pub(super) fn is_map_int_tuples(&self, expr: &Expr) -> bool {
        self.is_map_int_tuples_with_provenance(expr, self.as_special_export(expr))
    }

    pub(super) fn is_map_int_tuples_with_provenance(
        &self,
        expr: &Expr,
        provenance: Option<SpecialExport>,
    ) -> bool {
        self.is_shape_extensions_class_export_with_provenance(
            expr,
            SpecialExport::MapIntTuples,
            provenance,
        )
    }

    /// Bind shape-specific function decorators before ordinary decorator processing consumes them.
    pub(super) fn record_shape_function_metadata(
        &mut self,
        function: &StmtFunctionDef,
        is_top_level: bool,
        enclosing_class: Option<Idx<KeyClass>>,
    ) -> ShapeFunctionMetadata {
        let is_shape_dsl = function.decorator_list.iter().any(|decorator| {
            self.as_special_export(&decorator.expression) == Some(SpecialExport::ShapeDslFunction)
        });
        let is_type_shape_dsl = function.decorator_list.iter().any(|decorator| {
            self.as_special_export(&decorator.expression)
                == Some(SpecialExport::TypeShapeDslFunction)
        });
        if is_shape_dsl && is_type_shape_dsl {
            self.error(
                function.name.range(),
                ErrorKind::InvalidArgument,
                "`@shape_dsl_function` and `@type_shape_dsl_function` cannot be combined"
                    .to_owned(),
            );
        }

        let type_shape_dsl_def = if is_type_shape_dsl && !is_shape_dsl {
            match ParsedTypeShapeDslFunction::try_new(function.clone(), is_top_level) {
                Ok(definition) => Some(Arc::new(definition)),
                Err(error) => {
                    self.error(
                        error.range,
                        ErrorKind::InvalidArgument,
                        format!("@type_shape_dsl_function {}", error.message),
                    );
                    None
                }
            }
        } else {
            None
        };

        let uses_shape_dsl_ir_name = function.decorator_list.iter().find_map(|decorator| {
            let call = decorator.expression.as_call_expr()?;
            if self.as_special_export(&call.func) != Some(SpecialExport::UsesShapeDsl) {
                return None;
            }
            let name = call.arguments.args.first()?.as_name_expr()?;
            Some(ShortIdentifier::expr_name(name))
        });

        let shape_dsl_def = if is_shape_dsl && !is_type_shape_dsl {
            if let Some(vararg) = &function.parameters.vararg {
                self.error(
                    vararg.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: *args parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(kwarg) = &function.parameters.kwarg {
                self.error(
                    kwarg.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: **kwargs parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(keyword_only) = function.parameters.kwonlyargs.first() {
                self.error(
                    keyword_only.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: keyword-only parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }
            if let Some(positional_only) = function.parameters.posonlyargs.first() {
                self.error(
                    positional_only.range(),
                    ErrorKind::InvalidArgument,
                    "@shape_dsl_function: positional-only parameters are not supported in the shape DSL and will be ignored".to_owned(),
                );
            }

            match convert_shape_dsl_function(function) {
                Ok(dsl_function) => {
                    let dsl_function = Arc::new(dsl_function);
                    self.metadata
                        .push_shape_dsl(function.name.id.clone(), Arc::clone(&dsl_function));
                    Some(dsl_function)
                }
                Err(error) => {
                    self.error(
                        error.range,
                        ErrorKind::InvalidArgument,
                        format!("@shape_dsl_function: {}", error.message),
                    );
                    None
                }
            }
        } else {
            None
        };

        self.record_shape_declaration(
            &function.decorator_list,
            &function.name,
            function.range().end(),
            ShapeDeclarationOwner::Function(enclosing_class),
        );

        ShapeFunctionMetadata {
            shape_dsl_def,
            type_shape_dsl_def,
            uses_shape_dsl_ir_name,
        }
    }

    /// Record a shape declaration against the range its names are in scope for,
    /// which runs from the declaring name to `end`.
    ///
    /// Starting at the name leaves the decorators outside, so a declaration
    /// cannot resolve against itself. A class declaration applies to its body
    /// and methods, but scope lookup excludes it from nested classes.
    pub(super) fn record_shape_declaration(
        &mut self,
        decorators: &[Decorator],
        name: &Identifier,
        end: TextSize,
        owner: ShapeDeclarationOwner,
    ) {
        let Some((kind, dims)) = self.extract_shape_declaration(decorators, owner) else {
            return;
        };
        let mut unshadowed = Vec::new();
        for dim in dims {
            // Either kind of enclosing declaration counts: two same-named
            // variables in one scope would print identically.
            if let Some(outer) = self
                .shape_declarations
                .enclosing(name.range())
                .find(|outer| outer.dim(dim.name()).is_some())
            {
                self.error(
                    name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`{}` is declared by `{}` and is already declared by `{}` on an enclosing definition",
                        dim.name(),
                        kind.decorator(),
                        outer.kind.decorator(),
                    ),
                );
            } else {
                unshadowed.push(dim);
            }
        }
        self.shape_declarations.push_scope(
            TextRange::new(name.range().start(), end),
            ShapeDeclaration {
                kind,
                dims: unshadowed.into_boxed_slice(),
                declared_at: name.range(),
                owner,
            },
        );
    }

    /// Extract the names declared by `@static_jaxtyping("batch *rest")` or
    /// `@shape_vars("batch, *rest")`.
    ///
    /// A token either names one dimension or, with a leading `*`, a variadic run
    /// of them. The use-site-only forms -- integer literals, `_`, `...`, broadcast
    /// `#`, and arithmetic -- are rejected here, so a declaration always
    /// introduces exactly one binding whose kind is known from its syntax.
    fn extract_shape_declaration(
        &mut self,
        decorators: &[Decorator],
        owner: ShapeDeclarationOwner,
    ) -> Option<(ShapeDeclarationKind, Box<[Quantified]>)> {
        let mut declaration = None;
        let mut seen: Option<ShapeDeclarationKind> = None;
        for decorator in decorators {
            let Some(call) = decorator.expression.as_call_expr() else {
                if let Some(kind) = self.shape_declaration_kind(&decorator.expression) {
                    if self.reject_repeated_declaration(decorator, seen, kind) {
                        continue;
                    }
                    seen = Some(kind);
                    self.error(
                        decorator.range(),
                        ErrorKind::InvalidArgument,
                        format!(
                            "`{}` requires a declaration string, e.g. `{}`",
                            kind.decorator(),
                            kind.example()
                        ),
                    );
                }
                continue;
            };
            let Some(kind) = self.shape_declaration_kind(&call.func) else {
                continue;
            };
            if self.reject_repeated_declaration(decorator, seen, kind) {
                continue;
            }
            seen = Some(kind);
            declaration = self
                .parse_shape_declaration(call, kind, owner)
                .map(|dims| (kind, dims));
        }
        declaration
    }

    fn shape_declaration_kind(&self, expression: &Expr) -> Option<ShapeDeclarationKind> {
        match self.as_special_export(expression) {
            Some(SpecialExport::StaticJaxtyping) => Some(ShapeDeclarationKind::Jaxtyping),
            Some(SpecialExport::ShapeVars) => Some(ShapeDeclarationKind::ShapeVars),
            _ => None,
        }
    }

    /// One definition declares its dimensions once, with one decorator. Mixing the
    /// two spellings would leave the resolution rules for the scope ambiguous.
    fn reject_repeated_declaration(
        &mut self,
        decorator: &Decorator,
        seen: Option<ShapeDeclarationKind>,
        kind: ShapeDeclarationKind,
    ) -> bool {
        let Some(seen) = seen else {
            return false;
        };
        let message = if seen == kind {
            format!("Duplicate `{}` decorator", kind.decorator())
        } else {
            format!(
                "`{}` and `{}` cannot both declare one definition",
                seen.decorator(),
                kind.decorator()
            )
        };
        self.error(decorator.range(), ErrorKind::InvalidArgument, message);
        true
    }

    /// The dimensions named by a declaration string.
    fn parse_shape_declaration(
        &mut self,
        call: &ExprCall,
        kind: ShapeDeclarationKind,
        owner: ShapeDeclarationOwner,
    ) -> Option<Box<[Quantified]>> {
        // A class's `@shape_vars` dimensions default to gradual ones, so adding the
        // decorator to a published class does not break a downstream `Encoder[T]`.
        // `required=True` lets a library insist on them instead.
        let may_default = matches!(
            (kind, owner),
            (
                ShapeDeclarationKind::ShapeVars,
                ShapeDeclarationOwner::Class(_)
            )
        );
        let mut required = !may_default;
        for keyword in &call.arguments.keywords {
            match (&keyword.arg, &keyword.value) {
                (Some(arg), Expr::BooleanLiteral(value)) if arg == "required" && may_default => {
                    required = value.value;
                }
                (Some(arg), _) if arg == "required" => {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        if may_default {
                            "`required` must be a boolean literal (`True` or `False`)"
                        } else {
                            "`required` is supported only on `@shape_vars` classes"
                        }
                        .to_owned(),
                    );
                    return None;
                }
                _ => {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        format!(
                            "`{}` takes its declaration as a positional string",
                            kind.decorator()
                        ),
                    );
                    return None;
                }
            }
        }
        let [argument] = call.arguments.args.as_ref() else {
            self.error(
                call.range(),
                ErrorKind::InvalidArgument,
                format!(
                    "`{}` takes exactly 1 declaration string, got {}",
                    kind.decorator(),
                    call.arguments.args.len()
                ),
            );
            return None;
        };
        let Expr::StringLiteral(declaration) = argument else {
            self.error(
                argument.range(),
                ErrorKind::InvalidArgument,
                format!(
                    "`{}` requires a string literal declaration",
                    kind.decorator()
                ),
            );
            return None;
        };
        let range = declaration.range();

        // A declaration may introduce any number of variadic shapes: each lowers to
        // its own `IntTuple`-bound quantified, so two of them are independent unless
        // they meet inside one shape string. That case is rejected at the use site.
        let mut dims: Vec<Quantified> = Vec::new();
        for (index, token) in kind.split(declaration.value.to_str()).enumerate() {
            let (quantified_kind, name, restriction, gradual) = match token.strip_prefix('*') {
                Some(name) => (
                    QuantifiedKind::TypeVar,
                    name,
                    Restriction::Bound(Type::IntTuple(Box::new(IntTuple::shapeless()))),
                    Type::IntTuple(Box::new(IntTuple::shapeless())),
                ),
                None => (
                    QuantifiedKind::IntVar,
                    token,
                    Restriction::Unrestricted,
                    gradual_size(),
                ),
            };
            if kind == ShapeDeclarationKind::ShapeVars && name.contains(char::is_whitespace) {
                self.error(
                    range,
                    ErrorKind::InvalidArgument,
                    format!(
                        "`{token}` cannot be declared. Separate the names in a \
                         `@shape_vars` declaration with commas"
                    ),
                );
                continue;
            }
            if name == "_" || !is_valid_identifier(name) {
                self.error(
                    range,
                    ErrorKind::InvalidArgument,
                    format!(
                        "`{token}` cannot be declared. A `{}` declaration holds only \
                         dimension names such as `batch` and variadic shapes such as `*rest`",
                        kind.decorator()
                    ),
                );
                continue;
            }
            let name = Name::new(name);
            if dims.iter().any(|dim| dim.name() == &name) {
                self.error(
                    range,
                    ErrorKind::InvalidArgument,
                    format!("`{name}` is declared more than once"),
                );
                continue;
            }
            let variance = match owner {
                ShapeDeclarationOwner::Function(_) => PreInferenceVariance::Invariant,
                ShapeDeclarationOwner::Class(_) => PreInferenceVariance::Undefined,
            };
            dims.push(Quantified::new(
                QuantifiedIdentity::new(
                    self.module_info.name(),
                    AnchorIndex::new(range, index as u32),
                    QuantifiedOrigin::synthetic(),
                ),
                name,
                quantified_kind,
                (!required).then_some(gradual),
                restriction,
                variance,
            ));
        }

        Some(dims.into_boxed_slice())
    }

    /// Extract `@shaped_array(shape="Shape")` metadata from class decorators.
    pub(super) fn extract_shaped_array_metadata(
        &mut self,
        decorators: &[Decorator],
    ) -> Option<Box<ShapedArrayMetadata>> {
        let mut metadata = None;
        let mut seen_shaped_array = false;
        for decorator in decorators {
            let Some(call) = decorator.expression.as_call_expr() else {
                if self.as_special_export(&decorator.expression) == Some(SpecialExport::ShapedArray)
                {
                    if seen_shaped_array {
                        self.error(
                            decorator.range(),
                            ErrorKind::InvalidArgument,
                            "Duplicate `@shaped_array` decorator".to_owned(),
                        );
                        continue;
                    }
                    seen_shaped_array = true;
                    self.error(
                        decorator.range(),
                        ErrorKind::InvalidArgument,
                        "`@shaped_array` requires a `shape` keyword argument".to_owned(),
                    );
                }
                continue;
            };
            if self.as_special_export(&call.func) != Some(SpecialExport::ShapedArray) {
                continue;
            }
            if seen_shaped_array {
                self.error(
                    decorator.range(),
                    ErrorKind::InvalidArgument,
                    "Duplicate `@shaped_array` decorator".to_owned(),
                );
                continue;
            }
            seen_shaped_array = true;

            let mut invalid = false;
            if let Some(arg) = call.arguments.args.first() {
                self.error(
                    arg.range(),
                    ErrorKind::InvalidArgument,
                    "`@shaped_array` expects `shape` as a keyword argument".to_owned(),
                );
                invalid = true;
            }

            let mut shape_keyword = None;
            let mut builtin_indexing = true;
            for keyword in &call.arguments.keywords {
                let Some(arg) = &keyword.arg else {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        "Unpacking is not supported in `@shaped_array`".to_owned(),
                    );
                    invalid = true;
                    continue;
                };
                if arg.as_str() == "shape" {
                    if shape_keyword.is_none() {
                        shape_keyword = Some(keyword);
                    }
                } else if arg.as_str() == "builtin_indexing" {
                    let Expr::BooleanLiteral(value) = &keyword.value else {
                        self.error(
                            keyword.value.range(),
                            ErrorKind::InvalidArgument,
                            "`@shaped_array` `builtin_indexing` argument must be a boolean literal"
                                .to_owned(),
                        );
                        invalid = true;
                        continue;
                    };
                    builtin_indexing = value.value;
                } else {
                    self.error(
                        keyword.range(),
                        ErrorKind::InvalidArgument,
                        format!(
                            "Unexpected keyword argument `{}` for `@shaped_array`; expected `shape` or `builtin_indexing`",
                            arg.id
                        ),
                    );
                    invalid = true;
                }
            }

            let Some(shape_keyword) = shape_keyword else {
                if !invalid {
                    self.error(
                        call.range(),
                        ErrorKind::InvalidArgument,
                        "`@shaped_array` requires a `shape` keyword argument".to_owned(),
                    );
                }
                continue;
            };
            let Expr::StringLiteral(shape) = &shape_keyword.value else {
                self.error(
                    shape_keyword.value.range(),
                    ErrorKind::InvalidArgument,
                    "`@shaped_array` `shape` argument must be a string literal".to_owned(),
                );
                continue;
            };
            if !invalid {
                metadata = Some(Box::new(ShapedArrayMetadata {
                    shape_name: Name::new(shape.value.to_str()),
                    range: shape_keyword.value.range(),
                    builtin_indexing,
                }));
            }
        }
        metadata
    }

    /// Extract `capture_init` names from `@uses_shape_dsl` on a class's `forward` method.
    pub(super) fn extract_capture_init(&mut self, body: &[Stmt]) -> Option<Vec<Name>> {
        let forward = body
            .iter()
            .filter_map(|stmt| stmt.as_function_def_stmt())
            .find(|function| function.name.as_str() == "forward")?;

        forward.decorator_list.iter().find_map(|decorator| {
            let call = decorator.expression.as_call_expr()?;
            if self.as_special_export(&call.func) != Some(SpecialExport::UsesShapeDsl) {
                return None;
            }
            let capture_init = call.arguments.keywords.iter().find(|keyword| {
                keyword
                    .arg
                    .as_ref()
                    .is_some_and(|arg| arg.as_str() == "capture_init")
            })?;
            let list = capture_init.value.as_list_expr()?;
            Some(
                list.elts
                    .iter()
                    .filter_map(|element| {
                        if let Some(string) = element.as_string_literal_expr() {
                            Some(Name::new(string.value.to_str()))
                        } else {
                            self.error(
                                element.range(),
                                ErrorKind::InvalidArgument,
                                "`capture_init` entries must be string literals".to_owned(),
                            );
                            None
                        }
                    })
                    .collect(),
            )
        })
    }

    /// Record binding dependencies for a type parameter bound, including the
    /// shape-specific handling required by `shape_extensions.Flag` and
    /// `shape_extensions.Index`.
    pub(super) fn record_type_parameter_bound(
        &mut self,
        bound_expr: &mut Expr,
        usage: &mut Usage,
    ) -> TypeParameterBound {
        if let Expr::Subscript(subscript) = bound_expr
            && self.shape_extension_type_parameter_bound_marker(&subscript.value)
                == Some(SpecialExport::Flag)
        {
            self.ensure_expr(&mut subscript.value, usage);
            self.ensure_type_with_usage(&mut subscript.slice, None, usage);
            return TypeParameterBound::ShapeFlag {
                domain: Some((*subscript.slice).clone()),
                range: subscript.range,
            };
        }

        match self.shape_extension_type_parameter_bound_marker(bound_expr) {
            Some(SpecialExport::Flag) => {
                let range = bound_expr.range();
                self.ensure_expr(bound_expr, usage);
                TypeParameterBound::ShapeFlag {
                    domain: None,
                    range,
                }
            }
            Some(SpecialExport::Index) => {
                self.ensure_expr(bound_expr, usage);
                TypeParameterBound::ShapeIndex
            }
            Some(SpecialExport::NamedInts) => {
                self.ensure_expr(bound_expr, usage);
                TypeParameterBound::ShapeNamedInts
            }
            _ => {
                self.ensure_type_with_usage(bound_expr, None, usage);
                TypeParameterBound::Ordinary(bound_expr.clone())
            }
        }
    }

    fn shape_extension_type_parameter_bound_marker(&self, expr: &Expr) -> Option<SpecialExport> {
        let provenance = self.as_special_export(expr);
        [
            SpecialExport::Flag,
            SpecialExport::Index,
            SpecialExport::NamedInts,
        ]
        .into_iter()
        .find(|special| {
            self.is_shape_extensions_class_export_with_provenance(expr, *special, provenance)
        })
    }
}
