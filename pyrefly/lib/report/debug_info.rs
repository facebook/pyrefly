/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::type_name_of_val;

use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;

use crate::alt::answers::AnswerEntry;
use crate::alt::answers::AnswerTable;
use crate::alt::answers::Answers;
use crate::alt::traits::SolveRecursive;
use crate::binding::bindings::BindingEntry;
use crate::binding::bindings::BindingTable;
use crate::binding::bindings::Bindings;
use crate::binding::table::TableKeyed;
use crate::config::error::ErrorConfigs;
use crate::error::collector::ErrorCollector;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::table_for_each;
use crate::util::display::DisplayWithCtx;
use crate::util::prelude::SliceExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    modules: SmallMap<ModuleName, Module>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Module {
    bindings: Vec<Binding>,
    errors: Vec<Error>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Binding {
    kind: String,
    key: String,
    location: String,
    binding: String,
    result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Error {
    location: String,
    message: String,
}

impl DebugInfo {
    pub fn new(
        modules: &[(&ModuleInfo, &ErrorCollector, &Bindings, &Answers)],
        error_configs: &ErrorConfigs,
    ) -> Self {
        fn f<K: SolveRecursive>(
            t: &AnswerEntry<K>,
            module_info: &ModuleInfo,
            bindings: &Bindings,
            res: &mut Vec<Binding>,
        ) where
            BindingTable: TableKeyed<K, Value = BindingEntry<K>>,
            AnswerTable: TableKeyed<K, Value = AnswerEntry<K>>,
        {
            for (idx, val) in t.iter() {
                let key = bindings.idx_to_key(idx);
                res.push(Binding {
                    kind: type_name_of_val(key).rsplit_once(':').unwrap().1.to_owned(),
                    key: module_info.display(key).to_string(),
                    location: module_info.source_range(key.range()).to_string(),
                    binding: bindings.get(idx).display_with(bindings).to_string(),
                    result: match val.get() {
                        None => "None".to_owned(),
                        Some(v) => v.to_string(),
                    },
                })
            }
        }

        Self {
            modules: modules
                .iter()
                .map(|(module_info, errors, bindings, answers)| {
                    let mut res = Vec::new();
                    let error_config = error_configs.get(module_info.path());
                    table_for_each!(answers.table(), |t| f(t, module_info, bindings, &mut res));
                    let errors = errors.collect(error_config).shown.map(|e| Error {
                        location: e.source_range().to_string(),
                        message: e.msg().to_owned(),
                    });
                    (
                        module_info.name(),
                        Module {
                            bindings: res,
                            errors,
                        },
                    )
                })
                .collect(),
        }
    }
}
