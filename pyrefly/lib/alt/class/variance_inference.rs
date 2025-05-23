use crate::types::type_var::PreInferenceVariance;
use crate::types::type_var::Variance;

pub fn pre_to_post_variance(pre_variance: PreInferenceVariance) -> Variance {
    match pre_variance {
        PreInferenceVariance::PCovariant => Variance::Covariant,
        PreInferenceVariance::PContravariant => Variance::Contravariant,
        PreInferenceVariance::PInvariant => Variance::Invariant,
        // TODO: we should infer variance for this case
        PreInferenceVariance::PUndefined => Variance::Invariant,
    }
}
