/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::any::type_name_of_val;
use std::sync::Arc;

use pyrefly_util::arc_id::ArcId;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::prelude::SliceExt;
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
use crate::config::config::ConfigFile;
use crate::error::collector::ErrorCollector;
use crate::module::module_info::ModuleInfo;
use crate::module::module_name::ModuleName;
use crate::state::handle::Handle;
use crate::state::load::Load;
use crate::state::state::Transaction;
use crate::table_for_each;

pub fn debug_info(transaction: &Transaction, handles: &[Handle], is_javascript: bool) -> String {
    fn f(
        transaction: &Transaction,
        handles: &[Handle],
    ) -> Option<Vec<(ArcId<ConfigFile>, Arc<Load>, Bindings, Arc<Answers>)>> {
        handles
            .iter()
            .map(|x| {
                Some((
                    transaction.get_config(x)?,
                    transaction.get_load(x)?,
                    transaction.get_bindings(x)?,
                    transaction.get_answers(x)?,
                ))
            })
            .collect()
    }

    let owned = f(transaction, handles).expect("Everything to be computed for debug info");
    let debug_info =
        DebugInfo::new(&owned.map(|x| (&*x.0, &x.1.module_info, &x.1.errors, &x.2, &*x.3)));
    let mut output = serde_json::to_string(&debug_info).unwrap();
    if is_javascript {
        output = format!("var data = {output}");
    }
    output
}

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
        modules: &[(
            &ConfigFile,
            &ModuleInfo,
            &ErrorCollector,
            &Bindings,
            &Answers,
        )],
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
                .map(|(config, module_info, errors, bindings, answers)| {
                    let mut res = Vec::new();
                    let error_config = config.get_error_config(module_info.path().as_path());
                    table_for_each!(answers.table(), |t| f(t, module_info, bindings, &mut res));
                    let errors = errors.collect(&error_config).shown.map(|e| Error {
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
