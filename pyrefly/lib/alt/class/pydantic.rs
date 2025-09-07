/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::module_name::ModuleName;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::pydantic::PydanticModelKind::RootModel;
use crate::types::class::Class;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_pydantic_root_model_type_via_mro(
        &self,
        class: &Class,
        metadata: &ClassMetadata,
    ) -> Option<Type> {
        if !matches!(metadata.pydantic_model_kind(), Some(RootModel)) {
            return None;
        }

        let mro = self.get_mro_for_class(class);
        for base_type in mro.ancestors_no_object() {
            if base_type.has_qname(ModuleName::pydantic_root_model().as_str(), "RootModel") {
                let targs = base_type.targs().as_slice();
                let root_type = targs.last().cloned();
                if root_type.is_some() {
                    return root_type;
                }
            }
        }

        None
    }
}
