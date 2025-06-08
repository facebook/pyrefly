use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::class::class_field::ClassField;
use crate::types::class::Class;
use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Variance;
use crate::types::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, TypeEq)]
pub struct VarianceMap(pub SmallMap<String, Variance>);

impl VisitMut<Type> for VarianceMap {
    fn visit_mut(&mut self, _visitor: &mut dyn FnMut(&mut Type)) {
        // No-op: VarianceMap does not contain any Type
    }

    fn recurse_mut(&mut self, _visitor: &mut dyn FnMut(&mut Type)) {
        // No-op: VarianceMap does not contain any Type
    }
}

impl Display for VarianceMap {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{{")?;
        for (key, value) in self.0.iter() {
            write!(f, "{}: {}, ", key, value)?;
        }
        write!(f, "}}")
    }
}

pub fn pre_to_post_variance(pre_variance: PreInferenceVariance) -> Variance {
    match pre_variance {
        PreInferenceVariance::PCovariant => Variance::Covariant,
        PreInferenceVariance::PContravariant => Variance::Contravariant,
        PreInferenceVariance::PInvariant => Variance::Invariant,
        // TODO: we should infer variance for this case
        PreInferenceVariance::PUndefined => Variance::Invariant,
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn variance_map(
        &self,
        class: &Class,
        _base_types: Vec<Type>,
        _fields: Vec<Arc<ClassField>>,
    ) -> Arc<VarianceMap> {
        // TODO: use base_types and fields to obtain a populated variance map
        let mut variances = SmallMap::new();

        for tparam in class.tparams().iter() {
            variances.insert(
                tparam.name().as_str().to_owned(),
                pre_to_post_variance(tparam.variance),
            );
        }
        Arc::new(VarianceMap(variances))
    }
}
