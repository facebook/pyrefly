/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use serde::Serialize;
use starlark_map::small_map::SmallMap;

use crate::state::handle::Handle;
use crate::state::state::Transaction;

#[derive(Serialize)]
struct Output {
    modules: Vec<ModuleOutput>,
}

#[derive(Serialize)]
struct ModuleOutput {
    name: ModuleName,
    path: ModulePath,
    types: SmallMap<String, String>,
    definitions: SmallMap<String, (String, String)>,
}

fn trace_module(transaction: &Transaction, handle: &Handle) -> Option<ModuleOutput> {
    let info = transaction.get_module_info(handle)?;
    let ast = transaction.get_ast(handle)?;

    let mut types = SmallMap::new();
    let mut definitions = SmallMap::new();
    ast.visit(&mut |x| {
        let loc = match x {
            Expr::Name(x) => x.range,
            Expr::Attribute(x) => x.attr.range,
            _ => return,
        };
        if let Some(ty) = transaction.get_type_at(handle, loc.start()) {
            types.insert(info.display_range(x.range()).to_string(), ty.to_string());
        }
        // TODO: Support multiple definitions
        if let Some(def) = transaction
            .goto_definition(handle, loc.start())
            .into_iter()
            .next()
        {
            definitions.insert(
                info.display_range(x.range()).to_string(),
                (
                    def.module_info.path().to_string(),
                    def.module_info.display_range(def.range).to_string(),
                ),
            );
        }
    });
    Some(ModuleOutput {
        name: handle.module(),
        path: handle.path().dupe(),
        types,
        definitions,
    })
}

pub fn trace(transaction: &Transaction) -> String {
    let mut modules = Vec::new();
    for h in transaction.handles() {
        if let Some(module) = trace_module(transaction, &h) {
            modules.push(module);
        }
    }
    let output = Output { modules };
    serde_json::to_string_pretty(&output).unwrap()
}
