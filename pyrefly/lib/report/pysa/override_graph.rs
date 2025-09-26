/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use pyrefly_types::class::Class;
use ruff_python_ast::name::Name;

use crate::alt::types::decorated_function::DecoratedFunction;
use crate::binding::binding::Binding;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::KeyDecoratedFunction;
use crate::graph::index::Idx;
use crate::report::pysa::ClassRef;
use crate::report::pysa::DefinitionRef;
use crate::report::pysa::FunctionId;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::ModuleReversedOverrideGraph;
use crate::report::pysa::PysaLocation;
use crate::report::pysa::WholeProgramReversedOverrideGraph;
use crate::report::pysa::get_all_functions;
use crate::report::pysa::get_class_field_declaration;
use crate::report::pysa::should_export_function;

/// A map from a (base) method to methods that directly override it
#[derive(Debug)]
pub(crate) struct OverrideGraph {
    edges: HashMap<DefinitionRef, HashSet<DefinitionRef>>,
}

impl OverrideGraph {
    pub fn new() -> Self {
        OverrideGraph {
            edges: HashMap::new(),
        }
    }

    fn add_edge(&mut self, base_method: DefinitionRef, overriding_method: DefinitionRef) {
        self.edges
            .entry(base_method)
            .or_default()
            .insert(overriding_method);
    }

    pub fn from_reversed(reversed_override_graph: &WholeProgramReversedOverrideGraph) -> Self {
        let mut graph = OverrideGraph::new();
        for entry in reversed_override_graph.0.iter() {
            graph.add_edge(entry.value().clone(), entry.key().clone());
        }
        graph
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
) -> Option<DefinitionRef> {
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
            DefinitionRef {
                module_id: class.module_id,
                module_name: class.module_name,
                function_id: FunctionId::Function {
                    location: PysaLocation(
                        context.module_info.display_range(last_function.id_range()),
                    ),
                },
                identifier: field.to_string(),
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
                let current_function = DefinitionRef::from_decorated_function(&function, context);
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
