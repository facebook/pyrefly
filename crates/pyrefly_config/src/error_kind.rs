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
    Serialize
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
    /// Attempting to annotate a name with incompatible annotations.
    /// e.g. when a name is annotated in multiple branches of an if statement
    AnnotationMismatch,
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
    /// A subclass method incorrectly changes the name of a positional parameter while overriding
    /// a method of a parent class.
    BadParamNameOverride,
    /// Invalid exception or cause in `raise` statement.
    BadRaise,
    /// Attempting to return a value that does not match the function's return type.
    /// Can also arise when returning values from generators.
    BadReturn,
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
    /// Calling a function marked with `@deprecated`
    Deprecated,
    /// Raised when a class implicitly becomes abstract by defining abstract members without
    /// inheriting from `abc.ABC` or using `abc.ABCMeta`.
    ImplicitAbstractClass,
    /// This error is raised when Pyrefly infers an implicit `Any`
    ImplicitAny,
    /// Usage of a module that was not actually imported, but does exist.
    ImplicitImport,
    /// An attribute was implicitly defined by assignment to `self` in a method that we
    /// do not recognize as always executing (we recognize constructors and some test setup
    /// methods).
    ImplicitlyDefinedAttribute,
    /// An inconsistency between inherited fields or methods from multiple base classes.
    InconsistentInheritance,
    /// An inconsistency between the signature of a function overload and the implementation.
    InconsistentOverload,
    /// Internal Pyrefly error.
    InternalError,
    /// Attempting to write an annotation that is invalid for some reason.
    InvalidAnnotation,
    /// Passing an argument that is invalid for reasons besides type.
    InvalidArgument,
    /// Error caused by incorrect usage of a decorator.
    /// e.g. using @final on a top-level function
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
    /// The attribute exists but does not support this access pattern.
    NoAccess,
    /// Attempting to call an overloaded function, but none of the signatures match.
    NoMatchingOverload,
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
    /// A protocol attribute was first defined inside a method instead of the class body.
    ProtocolImplicitlyDefinedAttribute,
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
    /// An attribute is missing a type annotation and is initialized with the `None` literal.
    UnannotatedAttribute,
    /// A function parameter is missing a type annotation.
    UnannotatedParameter,
    /// A function is missing a return type annotation.
    UnannotatedReturn,
    /// Attempting to use a name that may be unbound or uninitialized
    UnboundName,
    /// An error caused by a keyword argument used in the wrong place.
    UnexpectedKeyword,
    /// An error caused by passing a positional argument for a keyword-only parameter.
    UnexpectedPositionalArgument,
    /// Attempting to use a name that is not defined.
    UnknownName,
    /// Identity comparison (`is` or `is not`) between types that are provably disjoint
    /// or between literals whose comparison result is statically known.
    UnnecessaryComparison,
    /// A return or yield that can never be reached.
    /// This occurs when a return/yield follows a statement that always exits,
    /// such as return, raise, break, or continue.
    Unreachable,
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
    /// Import is missing an expected stubs package
    UntypedImport,
    /// Result of async function call is never used or awaited
    UnusedCoroutine,
    /// A suppression comment is unused (no error to suppress, or specific codes are unused)
    UnusedIgnore,
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

impl ErrorKind {
    fn cache() -> SmallMap<String, ErrorKind> {
        let mut map = SmallMap::new();

        for kind in enum_iterator::all::<ErrorKind>() {
            let key = kind.to_string().to_case(Case::Kebab);
            map.insert(key, kind);
        }

        map
    }

    pub fn to_name(self) -> &'static str {
        ERROR_KIND_CACHE
            .get_index(self as usize)
            .unwrap()
            .0
            .as_str()
    }

    pub fn default_severity(self) -> Severity {
        // IMPORTANT: When updating these, also update error-kinds.mdx in the docs
        match self {
            ErrorKind::RevealType => Severity::Info,
            ErrorKind::Deprecated => Severity::Warn,
            ErrorKind::RedundantCast => Severity::Warn,
            ErrorKind::UnnecessaryComparison => Severity::Warn,
            // TODO: up severity to Warn when https://github.com/facebook/pyrefly/issues/1950 is fixed
            ErrorKind::UntypedImport => Severity::Ignore,
            ErrorKind::NotRequiredKeyAccess => Severity::Ignore,
            ErrorKind::ImplicitlyDefinedAttribute => Severity::Ignore,
            ErrorKind::ImplicitAbstractClass => Severity::Ignore,
            ErrorKind::ImplicitAny => Severity::Ignore,
            ErrorKind::UnannotatedParameter => Severity::Ignore,
            ErrorKind::UnannotatedReturn => Severity::Ignore,
            ErrorKind::UnannotatedAttribute => Severity::Ignore,
            ErrorKind::MissingSource => Severity::Ignore,
            ErrorKind::MissingOverrideDecorator => Severity::Ignore,
            ErrorKind::OpenUnpacking => Severity::Ignore,
            ErrorKind::NonExhaustiveMatch => Severity::Warn,
            ErrorKind::UnusedIgnore => Severity::Ignore,
            _ => Severity::Error,
        }
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
    #[test]
    fn test_error_kind_name() {
        assert_eq!(ErrorKind::Unsupported.to_name(), "unsupported");
        assert_eq!(ErrorKind::ParseError.to_name(), "parse-error");
    }

    #[test]
    fn test_doc() {
        // Verifies that the secondary headers in error-kinds.mdx contain the same variants as the ErrorKind enum and are sorted lexicographically.
        let mut all_error_kinds = all::<ErrorKind>();
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
}
