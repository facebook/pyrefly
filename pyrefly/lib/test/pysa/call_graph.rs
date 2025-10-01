/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use crate::report::pysa::call_graph::AttributeAccessCallees;
use crate::report::pysa::call_graph::CallCallees;
use crate::report::pysa::call_graph::CallGraphs;
use crate::report::pysa::call_graph::CallTarget;
use crate::report::pysa::call_graph::ExpressionCallees;
use crate::report::pysa::call_graph::IdentifierCallees;
use crate::report::pysa::call_graph::ImplicitReceiver;
use crate::report::pysa::call_graph::build_call_graphs_for_module;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::collect_function_base_definitions;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::override_graph::build_reversed_override_graph;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

// Omit fields from `FunctionRef` so that we can easily write the expected results
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct DefinitionRefForTest {
    module_name: String,
    identifier: String,
}

fn split_module_name_and_identifier(string: &str) -> (String, String) {
    let parts: Vec<&str> = string.split('.').collect();
    if let Some((last, rest)) = parts.split_last() {
        (rest.join("."), (*last).to_owned())
    } else {
        panic!("Invalid string: {}", string);
    }
}

impl DefinitionRefForTest {
    fn from_definition_ref(definition_ref: &FunctionRef) -> Self {
        Self {
            module_name: definition_ref.module_name.to_string(),
            identifier: definition_ref.function_name.to_string(),
        }
    }

    fn from_string(string: &str) -> Self {
        let (module_name, identifier) = split_module_name_and_identifier(string);
        Self {
            module_name,
            identifier,
        }
    }
}

impl std::fmt::Display for DefinitionRefForTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.module_name, self.identifier)
    }
}

impl<Target> CallTarget<Target> {
    fn with_receiver_class(mut self, receiver_class: String, context: &ModuleContext) -> Self {
        self.receiver_class = Some({
            let (module_name, class_name) = split_module_name_and_identifier(&receiver_class);
            get_class_ref(&module_name, &class_name, context)
        });
        self
    }
}

fn create_call_target(target: &str) -> CallTarget<DefinitionRefForTest> {
    CallTarget {
        target: DefinitionRefForTest::from_string(target),
        implicit_receiver: ImplicitReceiver::False,
        receiver_class: None,
    }
}

fn call_graph_for_test_from_actual(
    call_graph: &CallGraphs<FunctionRef, PysaLocation>,
) -> CallGraphs<DefinitionRefForTest, String> {
    let call_graph_for_test = call_graph
        .0
        .iter()
        .map(|(caller, callees)| {
            let caller = DefinitionRefForTest::from_definition_ref(caller);
            let callees_for_test = callees
                .iter()
                .map(|(location, expression_callees)| {
                    (
                        location.as_key(),
                        expression_callees.map_target(DefinitionRefForTest::from_definition_ref),
                    )
                })
                .collect::<HashMap<_, _>>();
            (caller, callees_for_test)
        })
        .collect::<HashMap<_, _>>();
    CallGraphs::from_map(call_graph_for_test)
}

fn call_graph_for_test_from_expected(
    call_graph: Vec<(
        String,
        Vec<(String, ExpressionCallees<DefinitionRefForTest>)>,
    )>,
) -> CallGraphs<DefinitionRefForTest, String> {
    let call_graph_for_test = call_graph
        .into_iter()
        .map(|(caller, callees)| {
            let callees_for_test = callees
                .into_iter()
                .map(|(location, expression_callees_for_test)| {
                    ((*location).to_owned(), expression_callees_for_test)
                })
                .collect::<HashMap<_, _>>();
            (DefinitionRefForTest::from_string(&caller), callees_for_test)
        })
        .collect::<HashMap<_, _>>();
    CallGraphs::from_map(call_graph_for_test)
}

fn test_building_call_graph_for_module(
    test_module_name: &str,
    code: &str,
    create_expected: &dyn Fn(
        &ModuleContext,
    ) -> Vec<(
        String,
        Vec<(String, ExpressionCallees<DefinitionRefForTest>)>,
    )>,
) {
    let state = create_state(test_module_name, code);
    let transaction = state.transaction();

    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let reversed_override_graph =
        build_reversed_override_graph(&handles, &transaction, &module_ids);
    let function_base_definitions = collect_function_base_definitions(
        &handles,
        &transaction,
        &module_ids,
        &reversed_override_graph,
    );

    let test_module_handle = get_handle_for_module_name(test_module_name, &transaction);
    let context = ModuleContext::create(&test_module_handle, &transaction, &module_ids).unwrap();
    let actual_call_graph = call_graph_for_test_from_actual(&build_call_graphs_for_module(
        &context,
        &function_base_definitions,
    ));
    let expected_call_graph = call_graph_for_test_from_expected(create_expected(&context));
    assert_eq!(
        expected_call_graph, actual_call_graph,
        "Expected: {:#?}. Actual: {:#?}",
        expected_call_graph, actual_call_graph,
    );
}

fn call_callees_from_expected(
    expected: Vec<CallTarget<DefinitionRefForTest>>,
) -> ExpressionCallees<DefinitionRefForTest> {
    ExpressionCallees::Call(CallCallees {
        call_targets: expected.to_vec(),
    })
}

fn attribute_access_callees_from_expected(
    expected: Vec<CallTarget<DefinitionRefForTest>>,
) -> ExpressionCallees<DefinitionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        callable_targets: expected.to_vec(),
    })
}

fn identifier_callees_from_expected(
    expected: Vec<CallTarget<DefinitionRefForTest>>,
) -> ExpressionCallees<DefinitionRefForTest> {
    ExpressionCallees::Identifier(IdentifierCallees {
        callable_targets: expected.to_vec(),
    })
}

static TEST_MODULE_NAME: &str = "test";
static TEST_DEFINITION_NAME: &str = "test.foo";

#[macro_export]
macro_rules! call_graph_testcase {
    ($name:ident, $module_name:ident, $code:literal, $create_expected:expr,) => {
        #[test]
        fn $name() {
            $crate::test::pysa::call_graph::test_building_call_graph_for_module(
                $module_name,
                $code,
                $create_expected,
            );
        }
    };
}

call_graph_testcase! {
    test_simple_function_call,
    TEST_MODULE_NAME,
    r#"
def foo():
  bar()
def bar():
  pass
"#,
   &|_context: &ModuleContext| { vec![(
        TEST_DEFINITION_NAME.to_owned(),
        vec![
            ("3:3-3:6".to_owned(), identifier_callees_from_expected(vec![create_call_target("test.bar")])),
            ("3:3-3:8".to_owned(), call_callees_from_expected(vec![create_call_target("test.bar")])),
        ],
    )] },
}

call_graph_testcase! {
    test_method_call_on_class,
    TEST_MODULE_NAME,
    r#"
class C:
  def m(self):
    pass
def foo(c: C):
  c.m()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.m").with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:3-6:6".to_owned(), attribute_access_callees_from_expected(call_target.clone())),
                ("6:3-6:8".to_owned(), call_callees_from_expected(call_target.clone())),
            ],
        )]
    },
}

call_graph_testcase! {
    test_conditional_function_assignment,
    TEST_MODULE_NAME,
    r#"
def baz() -> int: ...
def bar() -> bool: ...
def foo(b: bool):
  if b:
    f = bar
  else:
    f = baz
  f()
"#,
    &|_context: &ModuleContext| {
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:9-6:12".to_owned(), identifier_callees_from_expected(vec![create_call_target("test.bar")])),
                ("8:9-8:12".to_owned(), identifier_callees_from_expected(vec![create_call_target("test.baz")])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_conditional_function_with_none,
    TEST_MODULE_NAME,
    r#"
def bar() -> bool: ...
def foo(b: bool):
  if b:
    f = bar
  else:
    f = False
  f()
"#,
    &|_context: &ModuleContext| {
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![("5:9-5:12".to_owned(), identifier_callees_from_expected(vec![create_call_target("test.bar")]))],
        )]
    },
}

call_graph_testcase! {
    test_method_call_on_optional_class,
    TEST_MODULE_NAME,
    r#"
from typing import Optional
class C:
  def m():
    ...
def foo(c: Optional[C]):
  c.m()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.m").with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                (
                    "7:3-7:8".to_owned(),
                    call_callees_from_expected(call_target.clone())),
                (
                    "7:3-7:6".to_owned(),
                    attribute_access_callees_from_expected(call_target.clone()),
                )
            ],
        )]
    },
}

call_graph_testcase! {
    test_method_call_on_class_with_inheritance,
    TEST_MODULE_NAME,
    r#"
class C:
  def m():
    ...
class D(C):
  def m():
    ...
class E(D):
  def m():
    ...
def foo(c: C):
  c.m()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.m").with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("12:3-12:8".to_owned(), call_callees_from_expected(call_target.clone())),
                ("12:3-12:6".to_owned(), attribute_access_callees_from_expected(call_target.clone()))
            ],
        )]
    },
}

call_graph_testcase! {
    test_class_method,
    TEST_MODULE_NAME,
    r#"
class C:
  @classmethod
  def f(cls) -> int: ...
  def g(self) -> int: ...
def foo(c: C):
  C.f()  # implicit receiver
  c.f()  # implicit receiver
  C.g(c) # no implicit receiver
  c.g()  # implicit receiver
"#,
    &|context: &ModuleContext| {
        let class_method_target = vec![
            create_call_target("test.f").with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        let class_method_target_2 = vec![
            create_call_target("test.f").with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        let method_target = vec![
            create_call_target("test.g").with_implicit_receiver(ImplicitReceiver::False)
        ];
        let method_target_2 = vec![
            create_call_target("test.g").with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("7:3-7:8".to_owned(), call_callees_from_expected(class_method_target.clone())),
                ("7:3-7:6".to_owned(), attribute_access_callees_from_expected(class_method_target.clone())),
                ("8:3-8:8".to_owned(), call_callees_from_expected(class_method_target_2.clone())),
                ("8:3-8:6".to_owned(), attribute_access_callees_from_expected(class_method_target_2.clone())),
                ("9:3-9:9".to_owned(), call_callees_from_expected(method_target.clone())),
                ("9:3-9:6".to_owned(), attribute_access_callees_from_expected(method_target.clone())),
                ("10:3-10:8".to_owned(), call_callees_from_expected(method_target_2.clone())),
                ("10:3-10:6".to_owned(), attribute_access_callees_from_expected(method_target_2.clone())),
            ],
        )]
    },
}
