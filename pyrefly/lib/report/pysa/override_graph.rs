/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use pyrefly_build::handle::Handle;
use pyrefly_types::class::Class;
use pyrefly_util::thread_pool::ThreadPool;
use rayon::prelude::*;
use ruff_python_ast::name::Name;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Binding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::KeyDecoratedFunction;
use crate::graph::index::Idx;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::class::get_class_field_declaration;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::get_all_functions;
use crate::report::pysa::function::should_export_function;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleIds;
use crate::state::state::Transaction;

/// A map from a (base) method to methods that directly override it
#[derive(Debug)]
pub(crate) struct OverrideGraph {
    edges: HashMap<FunctionRef, HashSet<FunctionRef>>,
}

pub struct ModuleReversedOverrideGraph(HashMap<FunctionRef, FunctionRef>);

pub struct WholeProgramReversedOverrideGraph(dashmap::ReadOnlyView<FunctionRef, FunctionRef>);

impl OverrideGraph {
    pub fn new() -> Self {
        OverrideGraph {
            edges: HashMap::new(),
        }
    }

    fn add_edge(&mut self, base_method: FunctionRef, overriding_method: FunctionRef) {
        self.edges
            .entry(base_method)
            .or_default()
            .insert(overriding_method);
    }

    pub fn from_reversed(reversed_override_graph: &WholeProgramReversedOverrideGraph) -> Self {
        let mut graph = OverrideGraph::new();
        for (overriding_method, base_method) in reversed_override_graph.0.iter() {
            graph.add_edge(base_method.clone(), overriding_method.clone());
        }
        graph
    }
}

impl WholeProgramReversedOverrideGraph {
    #[cfg(test)]
    pub fn new() -> WholeProgramReversedOverrideGraph {
        WholeProgramReversedOverrideGraph(dashmap::DashMap::new().into_read_only())
    }

    pub fn get<'a>(&'a self, method: &FunctionRef) -> Option<&'a FunctionRef> {
        self.0.get(method)
    }
}

fn get_last_definition(
    key_decorated_function: Idx<KeyDecoratedFunction>,
    context: &ModuleContext,
) -> DecoratedFunction {
    // Follow the successor chain to find the last function
    let mut last_decorated_function = key_decorated_function;
    loop {
        let successor = context.bindings.get(last_decorated_function).successor;
        if let Some(successor) = successor {
            last_decorated_function = successor;
        } else {
            break;
        }
    }
    DecoratedFunction::from_bindings_answers(
        last_decorated_function,
        &context.bindings,
        &context.answers,
    )
}

fn get_super_class_member(
    class: &Class,
    field: &Name,
    context: &ModuleContext,
) -> Option<FunctionRef> {
    let super_class_member = context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
            solver.get_super_class_member(class, None, field)
        })
        .flatten()?;
    get_class_field_declaration(&super_class_member.defining_class, field, context)
        .and_then(|binding_class_field| {
            if let ClassFieldDefinition::MethodLike { definition, .. } =
                binding_class_field.definition
            {
                let binding = context.bindings.get(definition);
                if let Binding::Function(key_decorated_function, _pred, _class_meta) = binding {
                    Some(*key_decorated_function)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .map(|key_decorated_function| {
            let last_function = get_last_definition(key_decorated_function, context);
            let class =
                ClassRef::from_class(&super_class_member.defining_class, context.module_ids);
            FunctionRef {
                module_id: class.module_id,
                module_name: class.module_name,
                function_id: FunctionId::Function {
                    location: PysaLocation::new(
                        context.module_info.display_range(last_function.id_range()),
                    ),
                },
                function_name: field.clone(),
            }
        })
}

pub fn create_reversed_override_graph_for_module(
    context: &ModuleContext,
) -> ModuleReversedOverrideGraph {
    let mut graph = ModuleReversedOverrideGraph(HashMap::new());
    for function in get_all_functions(&context.bindings, &context.answers) {
        if !should_export_function(&function, context) {
            continue;
        }
        let name = function.metadata().kind.as_func_id().func;
        let overridden_base_method = function
            .defining_cls()
            .and_then(|class| get_super_class_member(class, &name, context));
        match overridden_base_method {
            Some(overridden_base_method) => {
                let current_function = FunctionRef::from_decorated_function(&function, context);
                assert!(
                    graph
                        .0
                        .insert(current_function, overridden_base_method)
                        .is_none(),
                    "Found function definitions with the same location"
                );
            }
            _ => (),
        }
    }

    graph
}

pub fn build_reversed_override_graph(
    handles: &Vec<Handle>,
    transaction: &Transaction,
    module_ids: &ModuleIds,
) -> WholeProgramReversedOverrideGraph {
    let reversed_override_graph = dashmap::DashMap::new();

    ThreadPool::new().install(|| {
        handles.par_iter().for_each(|handle| {
            let context = ModuleContext::create(handle, transaction, module_ids).unwrap();
            for (key, value) in create_reversed_override_graph_for_module(&context).0 {
                reversed_override_graph.insert(key, value);
            }
        });
    });

    WholeProgramReversedOverrideGraph(reversed_override_graph.into_read_only())
}
