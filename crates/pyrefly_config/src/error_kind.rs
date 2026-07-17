/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::LazyLock;

use clap::ValueEnum;
use convert_case::Case;
use convert_case::Casing;
use dupe::Dupe;
use enum_iterator::Sequence;
use parse_display::Display;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use yansi::Paint;
use yansi::Painted;

// IMPORTANT: these cases should be listed in order of severity
#[derive(
    Debug,
    Clone,
    Dupe,
    Copy,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Hash,
    Deserialize,
    Serialize,
    ValueEnum
)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ignore,
    Info,
    Warn,
    Error,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            // INFO and WARN are padded out to five characters to visually align with ERROR in messages
            Severity::Info => " INFO",
            Severity::Warn => " WARN",
            Severity::Error => "ERROR",
            Severity::Ignore => "",
        }
    }

    pub fn painted(self) -> Painted<&'static str> {
        (match self {
            Severity::Info => Paint::green,
            Severity::Warn => Paint::yellow,
            Severity::Error => Paint::red,
            Severity::Ignore => Paint::conceal,
        })(self.label())
    }

    pub fn is_enabled(self) -> bool {
        self != Severity::Ignore
    }
}

/// ErrorKind categorizes an error by the part of the spec the error is related to.
/// They are used in suppressions to identify which error should be suppressed.
//
// Keep ErrorKind sorted lexicographically.
// There are broad categories of error kinds, based on the word used in the name.
// "Bad": Specific, straightforward type errors. Could be a disagreement with a source
//    of truth, e.g. a function definition is how we determine a call has errors.
// "Missing": Same as "Bad" but we know specifically that something is missing.
// "Invalid": Something is being used incorrectly, such as a typing construct or language feature.
// These categories are flexible; use them for guidance when naming new ErrorKinds, but
// go with what feels right.
#[derive(Debug, Copy, Dupe, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
#[derive(Display, Sequence, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorKind {
    /// Attempting to call a method marked with `@abstractmethod`.
    AbstractMethodCall,
    /// Raised when an assert_type() call fails.
    AssertType,
    /// Attempting to call a function with the wrong number of arguments.
    BadArgumentCount,
    /// Attempting to call a function with an argument that does not match the parameter's type.
    BadArgumentType,
    /// Assigning a value of the wrong type to a variable.
    BadAssignment,
    /// A class definition has some typing-related error.
    /// e.g. multiple fields with the same name.
    /// Errors related specifically to inheritance should use InvalidInheritance.
    BadClassDefinition,
    /// Attempting to use a type that cannot be used as a contextmanager in a `with` statement.
    BadContextManager,
    /// An entry in user-defined `__all__` does not exist in the module.
    BadDunderAll,
    /// A function definition has some typing-related error.
    /// e.g. putting a non-default argument after a default argument.
    BadFunctionDefinition,
    /// Attempting to access a container with an incorrect index.
    /// This only occurs when Pyrefly can statically verify that the index is incorrect.
    BadIndex,
    /// Can't instantiate an abstract class or protocol
    BadInstantiation,
    /// Attempting to call a function with an incorrect keyword argument.
    /// e.g. f(x=1, x=2), or perhaps f(y=1) (where `f` has no parameter `y`).
    BadKeywordArgument,
    /// An error caused by a bad match statement.
    /// e.g. Writing a Foo(x, y, z) pattern when Foo only matches on (x, y).
    BadMatch,
    /// A subclass field or method incorrectly overrides a field/method of a parent class.
    BadOverride,
    /// A subclass field overrides a mutable attribute of a parent class with an incompatible type.
    /// Mutable (read-write) attributes require invariant types, unlike read-only attributes or
    /// methods which allow covariant overrides.
    /// This is a sub-kind of [BadOverride]: suppressing `bad-override` also suppresses this error.
    BadOverrideMutableAttribute,
    /// A subclass method incorrectly changes the name of a positional parameter while overriding
    /// a method of a parent class.
    /// This is a sub-kind of [BadOverride]: suppressing `bad-override` also suppresses this error.
    BadOverrideParamName,
    /// DEPRECATED: use [BadOverrideParamName] (`bad-override-param-name`) instead.
    /// Kept so that existing `# pyrefly: ignore[bad-param-name-override]` comments and
    /// config entries continue to work. This variant is never emitted by the type checker.
    BadParamNameOverride,
    /// Invalid exception or cause in `raise` statement.
    BadRaise,
    /// Attempting to return a value that does not match the function's return type.
    /// Can also arise when returning values from generators.
    BadReturn,
    /// A `functools.singledispatch` implementation is registered with a dispatch type that is
    /// not a subtype of the fallback function's first parameter, so it can never be dispatched to.
    BadSingledispatchRegister,
    /// Attempting to specialize a generic class with incorrect type arguments.
    /// e.g. `type[int, str]` is an error because `type` accepts only 1 type arg.
    BadSpecialization,
    /// A TypedDict definition has some typing-related error.
    /// e.g. using invalid keywords in the base class list.
    BadTypedDict,
    /// An error related to TypedDict keys.
    /// e.g. attempting to access a TypedDict with a key that does not exist.
    BadTypedDictKey,
    /// An error caused by unpacking.
    /// e.g. attempting to unpack an iterable into the wrong number of variables.
    BadUnpacking,
    /// A Polars DataFrame column literal has an element that does not fit the column's first-element dtype.
    ColumnTypeMismatch,
    /// A symbol has no type coverage. Emitted only by `pyrefly coverage check`.
    CoverageMissing,
    /// A symbol has partial type coverage. Emitted only by `pyrefly coverage check`.
    CoveragePartial,
    /// Calling a function marked with `@deprecated`
    Deprecated,
    /// Division, floor division, or modulo by a literal zero value.
    DivisionByZero,
    /// Explicit usage of `typing.Any` in an annotation.
    ExplicitAny,
    /// Raised when a class implicitly becomes abstract by defining abstract members without
    /// inheriting from `abc.ABC` or using `abc.ABCMeta`.
    ImplicitAbstractClass,
    /// Umbrella error kind for cases where Pyrefly infers an implicit `Any`.
    /// Most concrete sites emit one of the more specific sub-kinds below;
    /// `implicit-any` itself is reserved for the umbrella suppression/config
    /// code (suppressing `implicit-any` suppresses every sub-kind).
    ImplicitAny,
    /// An implicit `Any` introduced when a class attribute without an explicit
    /// annotation is defined by assignment to `self.x = None` or `self.x = ()`.
    /// This is a sub-kind of [ImplicitAny]: suppressing `implicit-any` also suppresses this error.
    ImplicitAnyAttribute,
    /// An implicit `Any` introduced when an empty container (`[]`, `{}`) cannot
    /// be inferred from context and is pinned to a container of `Any`.
    /// This is a sub-kind of [ImplicitAny]: suppressing `implicit-any` also suppresses this error.
    ImplicitAnyEmptyContainer,
    /// An implicit `Any` introduced when a lambda parameter or return type cannot
    /// be inferred from context.
    /// This is a sub-kind of [ImplicitAny]: suppressing `implicit-any` also suppresses this error.
    ImplicitAnyLambda,
    /// An implicit `Any` introduced because a function parameter has no
    /// annotation. The `self` and `cls` parameters of methods are excluded.
    /// This is a sub-kind of [ImplicitAny]: suppressing `implicit-any` also suppresses this error.
    ImplicitAnyParameter,
    /// An implicit `Any` introduced when a generic class, type alias, or
    /// special form (e.g., `tuple`, `Callable`, `type`) is used without
    /// explicit type arguments. Pyrefly defaults the missing type parameters
    /// to `Any`.
    /// This is a sub-kind of [ImplicitAny]: suppressing `implicit-any` also suppresses this error.
    ImplicitAnyTypeArgument,
    /// Usage of a module that was not actually imported, but does exist.
    ImplicitImport,
    /// An attribute was implicitly defined by assignment to `self` in a method that we
    /// do not recognize as always executing (we recognize constructors and some test setup
    /// methods).
    ImplicitlyDefinedAttribute,
    /// Equality or inequality comparison between incompatible types.
    IncompatibleComparison,
    /// Overload residual branch pruning left no valid branch for a solved type variable.
    IncompatibleOverloadResidual,
    /// An inconsistency between inherited fields or methods from multiple base classes.
    InconsistentInheritance,
    /// An inconsistency between the signature of a function overload and the implementation.
    InconsistentOverload,
    /// An inconsistency between a function parameter's type in an overload signature and its
    /// default value in the implementation.
    InconsistentOverloadDefault,
    /// Internal Pyrefly error.
    InternalError,
    /// Attempting to write an annotation that is invalid for some reason.
    InvalidAnnotation,
    /// Passing an argument that is invalid for reasons besides type.
    InvalidArgument,
    /// A method-only decorator was applied to a top-level function.
    /// e.g. using `@final` or `@override` on a top-level function.
    /// Defaults to `warn` because such usage is harmless at runtime and is
    /// sometimes intentional. Decorator misuse that violates a typing spec
    /// (e.g. `@dataclass` on a `Protocol`, `@disjoint_base` on a function)
    /// is reported under `BadClassDefinition` or `BadFunctionDefinition`
    /// instead, both of which default to `error`.
    InvalidDecorator,
    /// An error caused by incorrect inheritance in a class or type definition.
    /// e.g. a metaclass that is not a subclass of `type`.
    InvalidInheritance,
    /// Attempting to use a value that is not a valid kind of Literal.
    InvalidLiteral,
    /// An error caused by incorrect usage of the @overload decorator.
    /// e.g. not defining multiple variants for an overloaded function.
    InvalidOverload,
    /// An error related to ParamSpec definition or usage.
    InvalidParamSpec,
    /// An error caused by an invalid match pattern.
    InvalidPattern,
    /// A use of `typing.Self` in a context where Pyrefly does not recognize it as
    /// mapping to a valid class type.
    InvalidSelfType,
    /// An error caused by incorrect usage or definition of a Sentinel.
    InvalidSentinel,
    /// Attempting to call `super()` in a way that is not allowed.
    /// e.g. calling `super(Y, x)` on an object `x` that does not match the class `Y`.
    InvalidSuperCall,
    /// Incorrect Python syntax, construct is not allowed in this position.
    /// In many cases a parse error will also be reported.
    InvalidSyntax,
    /// An error related to type alias usage or definition.
    InvalidTypeAlias,
    /// An error caused by incorrect usage or definition of a TypeVar.
    InvalidTypeVar,
    /// An error caused by incorrect usage or definition of a TypeVarTuple.
    InvalidTypeVarTuple,
    /// An error caused by a type variable being used in a position incompatible with its declared variance,
    InvalidVariance,
    /// Attempting to use `yield` in a way that is not allowed.
    /// e.g. `yield from` with something that's not an iterable.
    InvalidYield,
    /// A file-level `# pyrefly: ignore-errors` (or `ignore-errors[code]`) directive
    /// appears after the first line of code, where it is silently inert. File-level
    /// suppressions are only honored in the preamble, at the top of the file.
    MisplacedIgnore,
    /// An error caused by calling a function without all the required arguments.
    /// Should be used when we can name the specific arguments that are missing.
    MissingArgument,
    /// Attempting to access an attribute that does not exist.
    MissingAttribute,
    /// Failed to import a module.
    MissingImport,
    /// Accessing an attribute that does not exist on a module.
    MissingModuleAttribute,
    /// A method overrides a parent class method but does not have the `@override` decorator.
    MissingOverrideDecorator,
    /// The source code for an imported package is missing.
    MissingSource,
    /// We are using bundled stubs for a package but the source code is missing.
    MissingSourceForStubs,
    /// The first string argument to a functional type definition does not match the bound name.
    NameMismatch,
    /// The attribute exists but does not support this access pattern.
    NoAccess,
    /// Umbrella error kind for cases where `Any` is returned from a function with a concrete return type.
    NoAnyReturn,
    /// An explicit `Any` returned from a function with a concrete return type.
    /// This is a sub-kind of [NoAnyReturn]: suppressing `no-any-return` also suppresses this error.
    NoAnyReturnExplicit,
    /// An implicit `Any` returned from a function with a concrete return type.
    /// This is a sub-kind of [NoAnyReturn]: suppressing `no-any-return` also suppresses this error.
    NoAnyReturnImplicit,
    /// Attempting to call an overloaded function, but none of the signatures match.
    NoMatchingOverload,
    /// The SCC fixpoint iteration did not converge within the maximum number of
    /// iterations. The inferred type may be incorrect; adding annotations can help.
    NonConvergentRecursion,
    /// Matching on an enum without covering all possible cases.
    NonExhaustiveMatch,
    /// Attempting to use something that isn't a type where a type is expected.
    /// This is a very general error and should be used sparingly.
    NotAType,
    /// An error raised when async is not used when it should be.
    NotAsync,
    /// Attempting to call a value that is not a callable.
    NotCallable,
    /// Attempting to use a non-iterable value as an iterable.
    NotIterable,
    /// Accessing a `NotRequired` TypedDict key without first proving it exists.
    NotRequiredKeyAccess,
    /// Unpacking an open TypedDict that may contain a bad key via inheritance.
    OpenUnpacking,
    /// An error related to parsing or syntax.
    ParseError,
    /// A potential conflict between an explicit keyword argument and a NotRequired
    /// TypedDict field. The field may be absent at runtime, so the conflict is not
    /// guaranteed. This is a separate error code from BadKeywordArgument to allow
    /// users to opt-in to this stricter check.
    PotentialBadKeywordArgument,
    /// A protocol attribute was first defined inside a method instead of the class body.
    ProtocolImplicitlyDefinedAttribute,
    /// Calling `.cuda()` on a `torch.Tensor` hard-codes the target device.
    /// Use `.to(device)` instead for device-agnostic code.
    PytorchEfficiencyLintCudaCall,
    /// Calling `.item()` on a `torch.Tensor` forces GPU→CPU synchronization,
    /// blocking the training loop until all pending GPU operations complete.
    PytorchEfficiencyLintItemCall,
    /// Passing a `torch.Tensor` to `print()` triggers `__repr__`, which forces
    /// GPU→CPU synchronization.
    PytorchEfficiencyLintPrintTensor,
    /// Calling `.to(device)` on a tensor returned by a factory function like
    /// `torch.zeros()` that already accepts a `device=` parameter. Passing
    /// `device=` directly avoids allocating the tensor on CPU first.
    PytorchEfficiencyLintRedundantToCall,
    /// The attribute exists but cannot be modified.
    ReadOnly,
    /// Attempting to annotate or redefine a name with a type that conflicts with an existing annotation in scope.
    Redefinition,
    /// Warning when casting a value to a type it is already compatible with.
    RedundantCast,
    /// Attempting to use value that is equivalent to True or always False in boolean context.
    RedundantCondition,
    /// Raised by a call to reveal_type().
    RevealType,
    /// Passing a string to something that expects an iterable of strings.
    StringAsIterable,
    /// DEPRECATED: use [ImplicitAnyAttribute] (`implicit-any-attribute`) instead.
    /// Kept so that existing `# pyrefly: ignore[unannotated-attribute]` comments
    /// and config entries continue to work. This variant is never emitted by
    /// the type checker.
    UnannotatedAttribute,
    /// DEPRECATED: use [ImplicitAnyParameter] (`implicit-any-parameter`) instead.
    /// Kept so that existing `# pyrefly: ignore[unannotated-parameter]` comments
    /// and config entries continue to work. This variant is never emitted by
    /// the type checker.
    UnannotatedParameter,
    /// A protocol member is assigned a value in the class body without an explicit type annotation.
    UnannotatedProtocolMember,
    /// A function is missing a return type annotation.
    UnannotatedReturn,
    /// Attempting to use a name that may be unbound or uninitialized
    UnboundName,
    /// An error caused by a keyword argument used in the wrong place.
    UnexpectedKeyword,
    /// An error caused by passing a positional argument for a keyword-only parameter.
    UnexpectedPositionalArgument,
    /// Attempting to use a type checker directive without importing it from `typing`.
    UnimportedDirective,
    /// An unannotated attribute assigned a value with unknown type.
    UnknownAttributeType,
    /// Accessing a DataFrame column that does not exist in the inferred schema.
    UnknownColumn,
    /// Attempting to use a name that is not defined.
    UnknownName,
    /// A variable assigned a value with unknown type without an explicit annotation.
    UnknownVariableType,
    /// Identity comparison (`is` or `is not`) between types that are provably disjoint
    /// or between literals whose comparison result is statically known.
    UnnecessaryComparison,
    /// Warning when calling a builtin type constructor (str, int, float, bool, bytes) on a value that is already of that type.
    UnnecessaryTypeConversion,
    /// A return or yield that can never be reached.
    /// This occurs when a return/yield follows a statement that always exits,
    /// such as return, raise, break, or continue.
    Unreachable,
    /// A match case whose pattern can never match the subject type.
    UnreachableMatchCase,
    /// `__all__` is defined but cannot be statically analyzed.
    UnresolvableDunderAll,
    /// Protocols decorated with `@runtime_checkable` can be used in `isinstance` checks
    /// The runtime only checks that an attribute with that name is present, so the
    /// type checker must warn if the types are not compatible.
    UnsafeOverlap,
    /// Attempting to use a feature that is not yet supported.
    Unsupported,
    /// Attempting to `del` something that cannot be deleted
    UnsupportedDelete,
    /// Attempting to apply an operation to arguments that do not support it.
    UnsupportedOperation,
    /// A class decorator whose own type is `Any`, obscuring the decorated class type.
    UntypedClassDecorator,
    /// A function decorator whose own type is `Any`, obscuring the decorated function type.
    UntypedFunctionDecorator,
    /// Import is missing an expected stubs package
    UntypedImport,
    /// Result of async function call is never used or awaited
    UnusedCoroutine,
    /// A suppression comment is unused (no error to suppress, or specific codes are unused)
    UnusedIgnore,
    /// A `# type: ignore` comment is unused (no error to suppress on that line)
    UnusedTypeIgnore,
    /// `@overload` bodies are never executed, so executable body logic is usually dead code.
    UselessOverloadBody,
    /// The inferred variance of a type variable does not match its declared variance.
    /// For example, a type variable used only in covariant positions in a protocol should be declared covariant.
    VarianceMismatch,
}

impl std::str::FromStr for ErrorKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ERROR_KIND_CACHE.get(s).copied().ok_or(())
    }
}

/// Computing the error kinds is disturbingly expensive, so cache the results.
/// Also means we can grab error code names without allocation, which is nice.
static ERROR_KIND_CACHE: LazyLock<SmallMap<String, ErrorKind>> = LazyLock::new(ErrorKind::cache);

static PYTORCH_EFFICIENCY_LINTS: LazyLock<Vec<ErrorKind>> = LazyLock::new(|| {
    enum_iterator::all::<ErrorKind>()
        .filter(|k| k.to_name().starts_with("pytorch-efficiency-lint-"))
        .collect()
});

impl ErrorKind {
    fn cache() -> SmallMap<String, ErrorKind> {
        let mut map = SmallMap::new();

        for kind in enum_iterator::all::<ErrorKind>() {
            let key = kind.to_string().to_case(Case::Kebab);
            map.insert(key, kind);
        }

        map
    }

    /// All error kinds with the `pytorch-efficiency-lint-` prefix.
    pub fn pytorch_efficiency_lints() -> &'static [ErrorKind] {
        &PYTORCH_EFFICIENCY_LINTS
    }

    pub fn to_name(self) -> &'static str {
        ERROR_KIND_CACHE
            .get_index(self as usize)
            .unwrap()
            .0
            .as_str()
    }

    /// Returns the parent error kind, if this is a sub-kind of another error.
    /// Suppressing the parent kind also suppresses this kind.
    pub fn parent_kind(self) -> Option<ErrorKind> {
        match self {
            ErrorKind::BadOverrideMutableAttribute | ErrorKind::BadOverrideParamName => {
                Some(ErrorKind::BadOverride)
            }
            ErrorKind::ImplicitAnyAttribute
            | ErrorKind::ImplicitAnyEmptyContainer
            | ErrorKind::ImplicitAnyLambda
            | ErrorKind::ImplicitAnyParameter
            | ErrorKind::ImplicitAnyTypeArgument => Some(ErrorKind::ImplicitAny),
            ErrorKind::NoAnyReturnExplicit | ErrorKind::NoAnyReturnImplicit => {
                Some(ErrorKind::NoAnyReturn)
            }
            _ => None,
        }
    }

    /// Returns the deprecated alias for this error kind, if any.
    /// The deprecated name is still accepted in suppressions and config.
    pub fn deprecated_alias(self) -> Option<ErrorKind> {
        match self {
            ErrorKind::BadOverrideParamName => Some(ErrorKind::BadParamNameOverride),
            ErrorKind::ImplicitAnyAttribute => Some(ErrorKind::UnannotatedAttribute),
            ErrorKind::ImplicitAnyParameter => Some(ErrorKind::UnannotatedParameter),
            _ => None,
        }
    }

    /// Returns all names that should match when checking suppressions.
    /// Includes this kind's name, any parent kind's name, and any deprecated alias.
    pub fn suppression_names(self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.to_name())
            .chain(self.parent_kind().map(|p| p.to_name()))
            .chain(self.deprecated_alias().map(|d| d.to_name()))
    }

    pub fn default_severity(self) -> Severity {
        // IMPORTANT: When updating these, also update error-kinds.mdx in the docs
        match self {
            ErrorKind::CoverageMissing => Severity::Warn,
            ErrorKind::CoveragePartial => Severity::Warn,
            ErrorKind::Deprecated => Severity::Warn,
            ErrorKind::DivisionByZero => Severity::Warn,
            ErrorKind::ExplicitAny => Severity::Ignore,
            ErrorKind::ImplicitAbstractClass => Severity::Ignore,
            ErrorKind::ImplicitAny => Severity::Ignore,
            ErrorKind::ImplicitAnyAttribute => Severity::Ignore,
            ErrorKind::ImplicitAnyEmptyContainer => Severity::Ignore,
            ErrorKind::ImplicitAnyParameter => Severity::Ignore,
            ErrorKind::ImplicitAnyTypeArgument => Severity::Ignore,
            ErrorKind::ImplicitImport => Severity::Warn,
            ErrorKind::ImplicitlyDefinedAttribute => Severity::Ignore,
            ErrorKind::IncompatibleComparison => Severity::Ignore,
            ErrorKind::InvalidDecorator => Severity::Warn,
            ErrorKind::MisplacedIgnore => Severity::Warn,
            ErrorKind::MissingOverrideDecorator => Severity::Ignore,
            ErrorKind::MissingSource => Severity::Ignore,
            ErrorKind::NameMismatch => Severity::Warn,
            ErrorKind::NoAnyReturn => Severity::Ignore,
            ErrorKind::NoAnyReturnExplicit => Severity::Ignore,
            ErrorKind::NoAnyReturnImplicit => Severity::Ignore,
            ErrorKind::NonExhaustiveMatch => Severity::Warn,
            ErrorKind::NonConvergentRecursion => Severity::Warn,
            ErrorKind::NotRequiredKeyAccess => Severity::Ignore,
            ErrorKind::OpenUnpacking => Severity::Ignore,
            ErrorKind::PytorchEfficiencyLintCudaCall => Severity::Ignore,
            ErrorKind::PytorchEfficiencyLintItemCall => Severity::Ignore,
            ErrorKind::PytorchEfficiencyLintPrintTensor => Severity::Ignore,
            ErrorKind::PytorchEfficiencyLintRedundantToCall => Severity::Ignore,
            ErrorKind::RedundantCast => Severity::Warn,
            ErrorKind::RedundantCondition => Severity::Warn,
            ErrorKind::RevealType => Severity::Info,
            ErrorKind::StringAsIterable => Severity::Ignore,
            ErrorKind::UnannotatedAttribute => Severity::Ignore,
            ErrorKind::UnannotatedParameter => Severity::Ignore,
            ErrorKind::UnannotatedReturn => Severity::Ignore,
            ErrorKind::ImplicitAnyLambda => Severity::Ignore,
            ErrorKind::UnknownAttributeType => Severity::Ignore,
            ErrorKind::UnknownVariableType => Severity::Ignore,
            ErrorKind::UnnecessaryComparison => Severity::Warn,
            ErrorKind::UnnecessaryTypeConversion => Severity::Warn,
            ErrorKind::Unreachable => Severity::Warn,
            ErrorKind::UnreachableMatchCase => Severity::Warn,
            ErrorKind::UnresolvableDunderAll => Severity::Warn,
            ErrorKind::UntypedClassDecorator => Severity::Ignore,
            ErrorKind::UntypedFunctionDecorator => Severity::Ignore,
            ErrorKind::UntypedImport => Severity::Warn,
            ErrorKind::UnusedIgnore => Severity::Ignore,
            ErrorKind::UnusedTypeIgnore => Severity::Ignore,
            ErrorKind::VarianceMismatch => Severity::Warn,
            // Overload bodies are runtime-dead, so this should warn rather than fail CI by default.
            ErrorKind::UselessOverloadBody => Severity::Warn,
            _ => Severity::Error,
        }
    }

    /// Returns true if this error kind is a type checker directive rather than
    /// a real diagnostic. Directives bypass suppression, baseline exclusion,
    /// and min-severity filtering, but can still be disabled via explicit
    /// per-kind severity overrides (e.g. `--ignore reveal-type`).
    pub fn is_directive(self) -> bool {
        matches!(self, ErrorKind::RevealType)
    }

    /// A soft error is a warning that should not influence overload selection
    /// or other type-inference decisions. The type check itself passed, but the
    /// code pattern is suspicious.
    pub fn is_soft(self) -> bool {
        matches!(self, ErrorKind::StringAsIterable)
    }

    /// Coverage kinds are emitted only by `pyrefly coverage check`.
    pub fn is_coverage(self) -> bool {
        matches!(
            self,
            ErrorKind::CoverageMissing | ErrorKind::CoveragePartial
        )
    }

    /// Returns the public documentation URL for this error kind.
    /// Example: https://pyrefly.org/en/docs/error-kinds/#bad-context-manager
    pub fn docs_url(self) -> String {
        format!(
            "https://pyrefly.org/en/docs/error-kinds/#{}",
            self.to_name()
        )
    }
}

#[cfg(test)]
mod tests {
    use enum_iterator::all;
    use pulldown_cmark::Event;
    use pulldown_cmark::HeadingLevel;
    use pulldown_cmark::Parser;
    use pulldown_cmark::Tag;
    use pulldown_cmark::TagEnd;

    use super::*;

    fn severity_str(s: Severity) -> &'static str {
        match s {
            Severity::Ignore => "ignore",
            Severity::Info => "info",
            Severity::Warn => "warn",
            Severity::Error => "error",
        }
    }

    #[test]
    fn test_error_kind_name() {
        assert_eq!(ErrorKind::Unsupported.to_name(), "unsupported");
        assert_eq!(ErrorKind::ParseError.to_name(), "parse-error");
    }

    #[test]
    fn test_unknown_column_kind_exists() {
        assert_eq!(ErrorKind::UnknownColumn.to_name(), "unknown-column");
        assert_eq!(
            "unknown-column".parse::<ErrorKind>(),
            Ok(ErrorKind::UnknownColumn)
        );
    }

    #[test]
    fn test_column_type_mismatch_kind_exists() {
        assert_eq!(
            ErrorKind::ColumnTypeMismatch.to_name(),
            "column-type-mismatch"
        );
        assert_eq!(
            "column-type-mismatch".parse::<ErrorKind>(),
            Ok(ErrorKind::ColumnTypeMismatch)
        );
        assert_eq!(
            ErrorKind::ColumnTypeMismatch.default_severity(),
            Severity::Error
        );
    }

    #[test]
    fn test_unknown_column_default_severity() {
        assert_eq!(ErrorKind::UnknownColumn.default_severity(), Severity::Error);
    }

    #[test]
    fn test_doc_headers() {
        // Verifies that the secondary headers in error-kinds.mdx contain the same variants as the ErrorKind enum and are sorted lexicographically.

        // Coverage kinds are only emitted by `pyrefly coverage check`, non-configurable, and
        // therefore intentionally undocumented.
        let mut all_error_kinds = all::<ErrorKind>().filter(|k| !k.is_coverage());

        let doc_path = std::env::var("ERROR_KINDS_DOC_PATH").expect(
            "ERROR_KINDS_DOC_PATH env var not set: cargo or buck should set this automatically",
        );
        let doc_contents = std::fs::read_to_string(&doc_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));
        let mut start = false;
        let mut in_header = false;
        let mut last_error_kind = None;
        for event in Parser::new(&doc_contents) {
            match event {
                Event::End(TagEnd::Heading(HeadingLevel::H1)) => {
                    // Don't start checking for error kinds until we get past the document title
                    start = true;
                }
                Event::Start(Tag::Heading {
                    level: HeadingLevel::H2,
                    ..
                }) => {
                    in_header = true;
                }
                Event::End(TagEnd::Heading(HeadingLevel::H2)) => {
                    in_header = false;
                }
                Event::Text(doc_error_kind) if start && in_header => {
                    let expected_error_kind = all_error_kinds
                        .next()
                        .unwrap_or_else(|| {
                            panic!("{doc_path} contains unexpected error kind: {doc_error_kind}")
                        })
                        .to_name();
                    if *expected_error_kind != *doc_error_kind {
                        panic!(
                            "Found inconsistency while iterating through ErrorKind enum and documentation at {doc_path}. The next enum variant is: {expected_error_kind}. The next doc header is: {doc_error_kind}"
                        );
                    }
                    if last_error_kind
                        .is_some_and(|last_error_kind| expected_error_kind < last_error_kind)
                    {
                        panic!(
                            "ErrorKind variant is out of lexicographical order: {expected_error_kind}"
                        );
                    }
                    last_error_kind = Some(expected_error_kind);
                }
                _ => {}
            }
        }
        if let Some(leftover_error_kind) = all_error_kinds.next() {
            panic!(
                "Documentation at {doc_path} is missing error kind: {}",
                leftover_error_kind.to_name()
            );
        }
    }

    #[test]
    fn test_doc_severities() {
        let doc_path = std::env::var("ERROR_KINDS_DOC_PATH").expect(
            "ERROR_KINDS_DOC_PATH env var not set: cargo or buck should set this automatically",
        );
        let doc_contents = std::fs::read_to_string(&doc_path)
            .unwrap_or_else(|e| panic!("Failed to read {doc_path}: {e}"));
        for kind in all::<ErrorKind>().filter(|k| !k.is_coverage()) {
            let header = format!("## {}", kind.to_name());
            let section_start = doc_contents.find(&header).expect(
                "could not validate documented severities due to missing error kind header",
            );
            let rest = &doc_contents[section_start + header.len()..];
            let section_end = rest.find("\n## ").unwrap_or(rest.len());
            let section = &rest[..section_end];
            let expected_severity = severity_str(kind.default_severity());
            if kind.default_severity() != Severity::Error {
                let expected_prefix = format!("\n\nDefault severity: `{expected_severity}`\n");
                if !section.starts_with(&expected_prefix) {
                    panic!(
                        "Error kind `{}` must have `Default severity: `{expected_severity}`` as the first line after the ## header.",
                        kind.to_name(),
                    );
                }
            } else if section.contains("Default severity:") {
                panic!(
                    "Error kind `{}` has default severity `error` (the default) and should not have a `Default severity:` line.",
                    kind.to_name(),
                );
            }
        }
    }
}
