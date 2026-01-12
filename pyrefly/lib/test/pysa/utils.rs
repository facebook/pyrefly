/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::num::NonZeroU32;

use pyrefly_build::handle::Handle;
use pyrefly_types::class::Class;
use pyrefly_util::lined_buffer::DisplayPos;
use pyrefly_util::lined_buffer::DisplayRange;
use pyrefly_util::lined_buffer::LineNumber;
use ruff_python_ast::name::Name;

use crate::binding::binding::KeyClass;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::ClassRef;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionNode;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::get_all_functions;
use crate::report::pysa::global_variable::GlobalVariableRef;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleKey;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::test::util::TestEnv;

pub fn create_state(module_name: &str, python_code: &str) -> State {
    let mut test_env = TestEnv::new();
    test_env.add(module_name, python_code);
    let (state, _) = test_env
        .with_default_require_level(Require::Everything)
        .to_state();
    state
}

pub fn get_handle_for_module_name(module_name: &str, transaction: &Transaction) -> Handle {
    // This is slow, but we don't care in tests.
    transaction
        .handles()
        .into_iter()
        .find(|handle| handle.module().as_str() == module_name)
        .expect("valid module name")
}

pub fn get_class(module_name: &str, class_name: &str, context: &ModuleContext) -> Class {
    let handle = get_handle_for_module_name(module_name, context.transaction);

    // This is slow, but we don't care in tests.
    let bindings = context.transaction.get_bindings(&handle).unwrap();
    let answers = context.transaction.get_answers(&handle).unwrap();
    bindings
        .keys::<KeyClass>()
        .map(|idx| answers.get_idx(idx).unwrap().0.clone().unwrap())
        .find(|class| class.name() == class_name)
        .expect("valid class name")
}

pub fn get_class_ref(module_name: &str, class_name: &str, context: &ModuleContext) -> ClassRef {
    let class = get_class(module_name, class_name, context);
    let module_id = context
        .module_ids
        .get(ModuleKey::from_module(class.module()))
        .expect("indexed module");

    ClassRef {
        class_id: ClassId::from_class(&class),
        module_id,
        class,
    }
}

pub fn get_function_ref(
    module_name: &str,
    function_name: &str,
    context: &ModuleContext,
) -> FunctionRef {
    let handle = get_handle_for_module_name(module_name, context.transaction);
    let context = ModuleContext::create(handle, context.transaction, context.module_ids).unwrap();

    // This is slow, but we don't care in tests.
    get_all_functions(&context)
        .filter(|function| function.should_export(&context))
        .find(|function| function.name().as_str() == function_name)
        .expect("valid function name")
        .as_function_ref(&context)
}

pub fn get_method_ref(
    module_name: &str,
    class_name: &str,
    function_name: &str,
    context: &ModuleContext,
) -> FunctionRef {
    let handle = get_handle_for_module_name(module_name, context.transaction);
    let context = ModuleContext::create(handle, context.transaction, context.module_ids).unwrap();

    // This is slow, but we don't care in tests.
    get_all_functions(&context)
        .filter(|function| function.should_export(&context))
        .find(|function| match function {
            FunctionNode::DecoratedFunction(decorated_function) => {
                function.name().as_str() == function_name
                    && decorated_function
                        .defining_cls()
                        .is_some_and(|class| class.name().as_str() == class_name)
            }
            FunctionNode::ClassField { class, name, .. } => {
                class.name().as_str() == class_name && name.as_str() == function_name
            }
        })
        .expect("valid method name")
        .as_function_ref(&context)
}

pub fn get_global_ref(
    module_name: &str,
    global_name: &str,
    context: &ModuleContext,
) -> GlobalVariableRef {
    let handle = get_handle_for_module_name(module_name, context.transaction);
    let context = ModuleContext::create(handle, context.transaction, context.module_ids).unwrap();
    GlobalVariableRef {
        module_id: context.module_id,
        module_name: context.handle.module(),
        name: Name::new(global_name),
    }
}

pub fn create_location(
    start_line: u32,
    start_column: u32,
    end_line: u32,
    end_column: u32,
) -> PysaLocation {
    PysaLocation::new(DisplayRange {
        start: DisplayPos::Source {
            line: LineNumber::new(start_line).expect("line must be greater than 0"),
            column: NonZeroU32::new(start_column).expect("Column must be greater than 0"),
        },
        end: DisplayPos::Source {
            line: LineNumber::new(end_line).expect("line must be greater than 0"),
            column: NonZeroU32::new(end_column).expect("Column must be greater than 0"),
        },
    })
}
