/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::iter;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_types::callable::Deprecation;
use pyrefly_types::typed_dict::ExtraItems;
use pyrefly_util::display::commas_iter;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;
use vec1::vec1;

use crate::alt::class::class_field::ClassField;
use crate::alt::types::pydantic::PydanticModelKind;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorInfo;
use crate::types::class::Class;
use crate::types::class::ClassType;
use crate::types::display::ClassDisplayContext;
use crate::types::keywords::DataclassKeywords;
use crate::types::keywords::DataclassTransformMetadata;
use crate::types::stdlib::Stdlib;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct ClassMetadata {
    metaclass: Metaclass,
    keywords: Keywords,
    typed_dict_metadata: Option<TypedDictMetadata>,
    named_tuple_metadata: Option<NamedTupleMetadata>,
    enum_metadata: Option<EnumMetadata>,
    protocol_metadata: Option<ProtocolMetadata>,
    dataclass_metadata: Option<DataclassMetadata>,
    extends_abc: bool,
    bases: Vec<Class>,
    has_generic_base_class: bool,
    has_base_any: bool,
    is_new_type: bool,
    is_final: bool,
    deprecation: Option<Deprecation>,
    is_disjoint_base: bool,
    total_ordering_metadata: Option<TotalOrderingMetadata>,
    /// If this class is decorated with `typing.dataclass_transform(...)`, the keyword arguments
    /// that were passed to the `dataclass_transform` call.
    dataclass_transform_metadata: Option<DataclassTransformMetadata>,
    pydantic_model_kind: Option<PydanticModelKind>,
    django_model_metadata: Option<DjangoModelMetadata>,
}

impl VisitMut<Type> for ClassMetadata {
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {
        // TODO: This is definitely wrong. We have types in lots of these places.
        // Doesn't seem to have gone wrong yet, but it will.
    }
}

impl Display for ClassMetadata {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "ClassMetadata(metaclass={})", self.metaclass)
    }
}

impl ClassMetadata {
    pub fn new(
        bases: Vec<Class>,
        metaclass: Metaclass,
        keywords: Vec<(Name, Type)>,
        typed_dict_metadata: Option<TypedDictMetadata>,
        named_tuple_metadata: Option<NamedTupleMetadata>,
        enum_metadata: Option<EnumMetadata>,
        protocol_metadata: Option<ProtocolMetadata>,
        dataclass_metadata: Option<DataclassMetadata>,
        extends_abc: bool,
        has_generic_base_class: bool,
        has_base_any: bool,
        is_new_type: bool,
        is_final: bool,
        deprecation: Option<Deprecation>,
        is_disjoint_base: bool,
        total_ordering_metadata: Option<TotalOrderingMetadata>,
        dataclass_transform_metadata: Option<DataclassTransformMetadata>,
        pydantic_model_kind: Option<PydanticModelKind>,
        django_model_metadata: Option<DjangoModelMetadata>,
    ) -> ClassMetadata {
        ClassMetadata {
            metaclass,
            keywords: Keywords(keywords),
            typed_dict_metadata,
            named_tuple_metadata,
            enum_metadata,
            protocol_metadata,
            dataclass_metadata,
            extends_abc,
            bases,
            has_generic_base_class,
            has_base_any,
            is_new_type,
            is_final,
            deprecation,
            is_disjoint_base,
            total_ordering_metadata,
            dataclass_transform_metadata,
            pydantic_model_kind,
            django_model_metadata,
        }
    }

    pub fn recursive() -> Self {
        ClassMetadata {
            metaclass: Metaclass::default(),
            keywords: Keywords::default(),
            typed_dict_metadata: None,
            named_tuple_metadata: None,
            enum_metadata: None,
            protocol_metadata: None,
            dataclass_metadata: None,
            extends_abc: false,
            bases: Vec::new(),
            has_generic_base_class: false,
            has_base_any: false,
            is_new_type: false,
            is_final: false,
            deprecation: None,
            is_disjoint_base: false,
            total_ordering_metadata: None,
            dataclass_transform_metadata: None,
            pydantic_model_kind: None,
            django_model_metadata: None,
        }
    }

    /// The class's custom (non-`type`) metaclass, if it has one.
    pub fn custom_metaclass(&self) -> Option<&ClassType> {
        self.metaclass.get()
    }

    pub fn custom_metaclass_raw(&self) -> &Metaclass {
        &self.metaclass
    }

    /// The class's metaclass.
    pub fn metaclass<'a>(&'a self, stdlib: &'a Stdlib) -> &'a ClassType {
        self.custom_metaclass()
            .unwrap_or_else(|| stdlib.builtins_type())
    }

    #[allow(dead_code)] // This is used in tests now, and will be needed later in production.
    pub fn keywords(&self) -> &[(Name, Type)] {
        &self.keywords.0
    }

    pub fn is_typed_dict(&self) -> bool {
        self.typed_dict_metadata.is_some()
    }

    pub fn is_pydantic_base_model(&self) -> bool {
        self.pydantic_model_kind.is_some()
    }

    pub fn is_django_model(&self) -> bool {
        self.django_model_metadata.is_some()
    }

    pub fn pydantic_model_kind(&self) -> Option<PydanticModelKind> {
        self.pydantic_model_kind.clone()
    }

    pub fn is_final(&self) -> bool {
        self.is_final
    }

    pub fn extends_abc(&self) -> bool {
        self.extends_abc
    }

    pub fn is_explicitly_abstract(&self) -> bool {
        for base in self.base_class_objects() {
            if base.has_toplevel_qname("abc", "ABC") {
                return true;
            }
        }
        // Only check the metaclass if it's directly specified on this class
        if let Metaclass::Direct(metaclass) = self.custom_metaclass_raw()
            && metaclass
                .class_object()
                .has_toplevel_qname("abc", "ABCMeta")
        {
            return true;
        }
        false
    }

    pub fn deprecation(&self) -> Option<&Deprecation> {
        self.deprecation.as_ref()
    }

    pub fn is_disjoint_base(&self) -> bool {
        self.is_disjoint_base
    }

    pub fn has_generic_base_class(&self) -> bool {
        self.has_generic_base_class
    }

    pub fn has_base_any(&self) -> bool {
        self.has_base_any
    }

    pub fn typed_dict_metadata(&self) -> Option<&TypedDictMetadata> {
        self.typed_dict_metadata.as_ref()
    }

    pub fn named_tuple_metadata(&self) -> Option<&NamedTupleMetadata> {
        self.named_tuple_metadata.as_ref()
    }

    pub fn enum_metadata(&self) -> Option<&EnumMetadata> {
        self.enum_metadata.as_ref()
    }

    pub fn base_class_objects(&self) -> &[Class] {
        &self.bases
    }

    pub fn is_protocol(&self) -> bool {
        self.protocol_metadata.is_some()
    }

    pub fn is_runtime_checkable_protocol(&self) -> bool {
        self.protocol_metadata
            .as_ref()
            .is_some_and(|p| p.is_runtime_checkable)
    }

    pub fn is_new_type(&self) -> bool {
        self.is_new_type
    }

    pub fn is_enum(&self) -> bool {
        self.enum_metadata.is_some()
    }

    pub fn is_total_ordering(&self) -> bool {
        self.total_ordering_metadata.is_some()
    }

    pub fn total_ordering_metadata(&self) -> Option<&TotalOrderingMetadata> {
        self.total_ordering_metadata.as_ref()
    }

    pub fn protocol_metadata(&self) -> Option<&ProtocolMetadata> {
        self.protocol_metadata.as_ref()
    }

    pub fn dataclass_metadata(&self) -> Option<&DataclassMetadata> {
        self.dataclass_metadata.as_ref()
    }

    pub fn dataclass_transform_metadata(&self) -> Option<&DataclassTransformMetadata> {
        self.dataclass_transform_metadata.as_ref()
    }

    pub fn django_model_metadata(&self) -> Option<&DjangoModelMetadata> {
        self.django_model_metadata.as_ref()
    }
}

/// A field that we synthesize and add to a class. Note that if a non-synthesized field already
/// exists on the class, it will take precedence over the synthesized field in attribute lookup.
/// If you want to modify the type of a non-synthesized field, see
/// AnswersSolver::get_special_class_field_type() in class_field.rs.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct ClassSynthesizedField {
    pub inner: Arc<ClassField>,
}

impl VisitMut<Type> for ClassSynthesizedField {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        let mut v = (*self.inner).clone();
        v.recurse_mut(f);
        self.inner = Arc::new(v);
    }
}

impl Display for ClassSynthesizedField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl ClassSynthesizedField {
    pub fn new(ty: Type) -> Self {
        Self {
            inner: Arc::new(ClassField::new_synthesized(ty)),
        }
    }
}

/// A class's synthesized fields, such as a dataclass's `__init__` method.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Default, VisitMut)]
pub struct ClassSynthesizedFields(SmallMap<Name, ClassSynthesizedField>);

impl ClassSynthesizedFields {
    pub fn new(fields: SmallMap<Name, ClassSynthesizedField>) -> Self {
        Self(fields)
    }

    pub fn get(&self, name: &Name) -> Option<&ClassSynthesizedField> {
        self.0.get(name)
    }

    /// Combines two sets of synthesized fields, with the second set
    /// overwriting any fields in the first set that have the same name.
    pub fn combine(mut self, other: Self) -> Self {
        self.0.reserve(other.0.len());
        for (name, field) in other.0.into_iter_hashed() {
            self.0.insert_hashed(name, field);
        }
        self
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = (&Name, &ClassSynthesizedField)> {
        self.0.iter()
    }
}

impl Display for ClassSynthesizedFields {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "ClassSynthesizedFields {{ {} }}",
            commas_iter(|| self
                .0
                .iter()
                .map(|(name, field)| format!("{name} => {field}")))
        )
    }
}

/// A struct representing a class's metaclass. A value of `None` indicates
/// no explicit metaclass, in which case the default metaclass is `type`.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Default)]
pub enum Metaclass {
    Direct(ClassType),
    Inherited(ClassType),
    #[default]
    None,
}

impl Display for Metaclass {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match &self {
            Self::Direct(metaclass) => write!(f, "{metaclass}"),
            Self::Inherited(metaclass) => write!(f, "inherited({metaclass})"),
            Self::None => write!(f, "type"),
        }
    }
}

impl Metaclass {
    /// Convenience function to get the metaclass as a ClassType, regardless of its origin
    pub fn get(&self) -> Option<&ClassType> {
        match self {
            Self::Direct(metaclass) => Some(metaclass),
            Self::Inherited(metaclass) => Some(metaclass),
            Self::None => None,
        }
    }
}

/// A struct representing the keywords in a class header, e.g. for
/// `Class A(foo=True): ...` we will have `"foo": Literal[True]`.
///
/// The `metaclass` keyword is not included, since we store the metaclass
/// separately as part of `ClassMetadata`.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq, Default)]
struct Keywords(Vec<(Name, Type)>);

impl Display for Keywords {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            commas_iter(|| self.0.iter().map(|(n, ty)| format!("{n}: {ty}")))
        )
    }
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct TypedDictMetadata {
    /// Field name to the value of the `total` keyword in the defining class.
    pub fields: SmallMap<Name, bool>,
    pub extra_items: ExtraItems,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct EnumMetadata {
    pub cls: ClassType,
    /// Whether this enum inherits from enum.Flag.
    pub is_flag: bool,
    /// Is there any `_value_` field present.
    pub has_value: bool,
    /// Whether this is a special Django enum.
    pub is_django: bool,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct NamedTupleMetadata {
    pub elements: SmallSet<Name>,
}

/// Defaults for `init_by_name` and `init_by_default`, per-field flags that control the name of
/// a field's corresponding `__init__` parameter. See DataclassFieldKeywords for more information.
#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct InitDefaults {
    pub init_by_name: bool,
    pub init_by_alias: bool,
}

impl Default for InitDefaults {
    fn default() -> Self {
        Self {
            init_by_name: false,
            init_by_alias: true,
        }
    }
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct DataclassMetadata {
    /// The dataclass fields, e.g., `{'x'}` for `@dataclass class C: x: int`.
    pub fields: SmallSet<Name>,
    pub kws: DataclassKeywords,
    pub field_specifiers: Vec<CalleeKind>,
    pub alias_keyword: Name,
    pub init_defaults: InitDefaults,
    /// Whether a default can be passed positionally to field specifier calls
    pub default_can_be_positional: bool,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct DjangoModelMetadata {
    /// The name of the field that has primary_key=True, if any.
    /// If None, the model uses the default auto-generated `id` field.
    pub custom_primary_key_field: Option<Name>,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct ProtocolMetadata {
    /// All members of the protocol, excluding ones defined on `object` and not overridden in a subclass.
    pub members: SmallSet<Name>,
    /// Whether this protocol is decorated with @runtime_checkable
    pub is_runtime_checkable: bool,
}

#[derive(Clone, Debug, TypeEq, PartialEq, Eq)]
pub struct TotalOrderingMetadata {
    /// Location of the decorator for `@total_ordering`.
    pub location: TextRange,
}

/// A struct representing a class's ancestors, in method resolution order (MRO)
/// and after dropping cycles and nonlinearizable inheritance.
///
/// Each ancestor is represented as a pair of a class and the type arguments
/// for that class, relative to the body of the current class, so for example
/// in
/// ```python
/// class A[T]: pass
/// class B[S](A[list[S]]): pass
/// class C(B[int]): pass
/// ```
/// we would get `[B[int], A[list[int]]]`.
///
/// If a class is present in multiple places of the inheritance tree (and is
/// linearizable using C3 linearization), it is possible it appears with
/// different type arguments. The type arguments computed here will always be
/// those coming from the instance that was selected during linearization.
#[derive(Clone, Debug, VisitMut, TypeEq, PartialEq, Eq)]
pub enum ClassMro {
    Resolved(Vec<ClassType>),
    Cyclic,
}

impl Display for ClassMro {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ClassMro::Resolved(xs) => {
                write!(f, "[{}]", commas_iter(|| xs.iter()))
            }
            ClassMro::Cyclic => write!(f, "Cyclic"),
        }
    }
}

impl ClassMro {
    /// Compute all ancestors the method resolution order (MRO).
    ///
    /// Each ancestor is paired with `targs: TArgs` representing type
    /// arguments in terms of the current class's type parameters. The `self`
    /// class is not included and should be considered implicitly at the front
    /// of the MRO.
    ///
    /// Python uses the C3 linearization algorithm to compute MRO. You can read
    /// about the algorithm and a worked-through example here:
    /// https://en.wikipedia.org/wiki/C3_linearization
    ///
    /// TODO: We currently omit some classes that are in the runtime MRO:
    /// `Generic`, `Protocol`, and `object`.
    pub fn new(
        cls: &Class,
        bases_with_mro: Vec<(&ClassType, Arc<ClassMro>)>,
        errors: &ErrorCollector,
    ) -> Self {
        match Linearization::new(cls, bases_with_mro, errors) {
            Linearization::Cyclic => Self::Cyclic,
            Linearization::Resolved(ancestor_chains) => {
                let ancestors = Linearization::merge(cls, ancestor_chains, errors);
                Self::Resolved(ancestors)
            }
        }
    }

    /// The MRO doesn't track `object` directly for efficiency, since it always comes last, and
    /// some use cases (for example checking if the type is an enum) do not care about `object`.
    pub fn ancestors_no_object(&self) -> &[ClassType] {
        match self {
            ClassMro::Resolved(ancestors) => ancestors,
            ClassMro::Cyclic => &[],
        }
    }

    pub fn ancestors<'a>(&'a self, stdlib: &'a Stdlib) -> impl Iterator<Item = &'a ClassType> {
        self.ancestors_no_object()
            .iter()
            .chain(iter::once(stdlib.object()))
    }

    pub fn recursive() -> Self {
        Self::Cyclic
    }
}

/// Represents one linearized "chain" of ancestors in the C3 linearization algorithm, which involves
/// building a series of linearized chains and then merging them. Each chain is one of:
/// - The MRO for a base class of the current class, or
/// - The list of direct base classes, in the order defined
///
/// All chains are represented in reverse order because that allows the merge step to be pop()-based,
/// and we use Vec1 so that we can automatically drop a chain once it's empty as the merge progresses.
struct AncestorChain(Vec1<ClassType>);

impl AncestorChain {
    fn from_base_and_ancestors(base: ClassType, base_ancestors: Vec<ClassType>) -> Self {
        AncestorChain(Vec1::from_vec_push(base_ancestors, base))
    }
}

enum Linearization {
    Resolved(Vec<AncestorChain>),
    Cyclic,
}

impl Linearization {
    pub fn empty() -> Self {
        Linearization::Resolved(vec![])
    }

    /// Implements the linearize stage of the C3 linearization algorithm for method resolution order (MRO).
    ///
    /// This is the `L(...)` function in https://web.archive.org/web/20241001181736/https://en.wikipedia.org/wiki/C3_linearization
    ///
    /// We detect cycles here, and return `Cyclic` if one is detected.
    ///
    /// The output, when successful, is a series of AncestorChains:
    /// - One for each base class's MRO, in the order that base classes are defined
    /// - One consisting of the base classes themselves in the order defined.
    fn new(
        cls: &Class,
        bases_with_mro: Vec<(&ClassType, Arc<ClassMro>)>,
        errors: &ErrorCollector,
    ) -> Linearization {
        let bases = match Vec1::try_from_vec(
            bases_with_mro
                .iter()
                .rev()
                .map(|(base, _)| (*base).clone())
                .collect(),
        ) {
            Ok(bases) => bases,
            Err(_) => return Linearization::empty(),
        };
        let mut ancestor_chains = Vec::new();
        for (base, mro) in bases_with_mro {
            match &*mro {
                ClassMro::Resolved(ancestors) => {
                    let ancestors_through_base = ancestors
                        .iter()
                        .map(|ancestor| ancestor.substitute_with(&base.substitution()))
                        .rev()
                        .collect::<Vec<_>>();
                    ancestor_chains.push(AncestorChain::from_base_and_ancestors(
                        base.clone(),
                        ancestors_through_base,
                    ));
                }
                // None and Cyclic both indicate a cycle, the distinction just
                // depends on how exactly the recursion in resolving keys plays out.
                ClassMro::Cyclic => {
                    let base = base.class_object();
                    let ctx = ClassDisplayContext::new(&[cls, base]);
                    errors.add(
                        cls.range(),
                        ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                        vec1![format!(
                            "Class `{}` inheriting from `{}` creates a cycle",
                            ctx.display(cls),
                            ctx.display(base),
                        )],
                    );
                    // Signal that we detected a cycle
                    return Linearization::Cyclic;
                }
            }
        }
        ancestor_chains.push(AncestorChain(bases));
        Linearization::Resolved(ancestor_chains)
    }

    /// Implements the `merge` step of the C3 linearization algorithm for method resolution order (MRO).
    ///
    /// We detect linearization failures here; if one occurs we abort with the merge results thus far.
    fn merge(
        cls: &Class,
        mut ancestor_chains: Vec<AncestorChain>,
        errors: &ErrorCollector,
    ) -> Vec<ClassType> {
        // Merge the base class ancestors into a single Vec, in MRO order.
        //
        // The merge rule says we take the first available "head" of a chain (which are represented
        // as reversed vecs) that is not in the "tail" of any chain, then strip it from all chains.
        let mut ancestors = Vec::new();
        while !ancestor_chains.is_empty() {
            // Identify a candidate for the next MRO entry: it must be the next ancestor in some chain,
            // and not be in the tail of any chain.
            let mut selected = None;
            for candidate_chain in ancestor_chains.iter() {
                let candidate = candidate_chain.0.last();
                let mut rejected = false;
                for ancestor_chain in ancestor_chains.iter() {
                    if ancestor_chain
                        .0
                        .iter()
                        .rev()
                        .skip(1)
                        .any(|class| class.qname() == candidate.qname())
                    {
                        rejected = true;
                        break;
                    }
                }
                if !rejected {
                    selected = Some(candidate.clone());
                    break;
                }
            }
            if let Some(selected) = selected {
                // Strip the selected class from all chains. Any empty chain is removed.
                let mut chains_to_remove = Vec::new();
                for (idx, ancestors) in ancestor_chains.iter_mut().enumerate() {
                    if ancestors.0.last().class_object().qname() == selected.class_object().qname()
                    {
                        match ancestors.0.pop() {
                            Ok(_) => {}
                            Err(_) => chains_to_remove.push(idx),
                        }
                    }
                }
                for (offset, idx) in chains_to_remove.into_iter().enumerate() {
                    ancestor_chains.remove(idx - offset);
                }
                // Push the selected class onto the result
                ancestors.push(selected);
            } else {
                // The ancestors are not linearizable at this point. Record an error and stop with
                // what we have so far.
                // (The while loop invariant ensures that ancestor_chains is non-empty, so unwrap is safe.)
                let first_candidate = &ancestor_chains
                    .first()
                    .unwrap()
                    .0
                    .last()
                    .class_object()
                    .dupe();
                let ctx = ClassDisplayContext::new(&[cls, first_candidate]);
                errors.add(
                    cls.range(),
                    ErrorInfo::Kind(ErrorKind::InvalidInheritance),
                    vec1![format!(
                        "Class `{}` has a nonlinearizable inheritance chain detected at `{}`",
                        ctx.display(cls),
                        ctx.display(first_candidate),
                    )],
                );

                ancestor_chains = Vec::new()
            }
        }
        ancestors
    }
}
