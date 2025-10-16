/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::HashMap;

use pretty_assertions::assert_eq;
use serde::Serialize;

use crate::report::pysa::call_graph::AttributeAccessCallees;
use crate::report::pysa::call_graph::CallCallees;
use crate::report::pysa::call_graph::CallGraph;
use crate::report::pysa::call_graph::CallGraphs;
use crate::report::pysa::call_graph::CallTarget;
use crate::report::pysa::call_graph::ExpressionCallees;
use crate::report::pysa::call_graph::FunctionTrait;
use crate::report::pysa::call_graph::IdentifierCallees;
use crate::report::pysa::call_graph::ImplicitReceiver;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::export_call_graphs;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::function::collect_function_base_definitions;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::override_graph::OverrideGraph;
use crate::report::pysa::override_graph::build_reversed_override_graph;
use crate::report::pysa::types::ScalarTypeProperties;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

// Omit fields from `FunctionRef` so that we can easily write the expected results
#[derive(Debug, Hash, Eq, PartialEq, Clone, Serialize, PartialOrd, Ord)]
struct FunctionRefForTest {
    module_name: String,
    defining_class: Option<String>,
    identifier: String,
}

impl FunctionTrait for FunctionRefForTest {}

fn split_module_and_identifier(string: &str) -> (String, String) {
    let parts: Vec<&str> = string.split('.').collect();
    if let Some((last, rest)) = parts.split_last() {
        (rest.join("."), (*last).to_owned())
    } else {
        panic!("Invalid string: {}", string);
    }
}

fn split_module_class_and_identifier(string: &str) -> (String, Option<String>, String) {
    let parts: Vec<&str> = string.split('.').collect();
    if let Some((last, rest)) = parts.split_last()
        && let Some((second_to_last, rest_of_rest)) = rest.split_last()
    {
        if rest_of_rest.is_empty() {
            (rest.join("."), None, (*last).to_owned())
        } else {
            (
                rest_of_rest.join("."),
                Some((*second_to_last).to_owned()),
                (*last).to_owned(),
            )
        }
    } else {
        panic!("Invalid string: {}", string)
    }
}

impl FunctionRefForTest {
    fn from_definition_ref(
        function_ref: FunctionRef,
        function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
    ) -> Self {
        Self {
            module_name: function_ref.module_name.to_string(),
            identifier: function_ref.function_name.to_string(),
            defining_class: function_base_definitions
                .get(function_ref.module_id, &function_ref.function_id)
                .and_then(|definition| {
                    definition
                        .defining_class
                        .as_ref()
                        .map(|class| class.class.name().to_string())
                }),
        }
    }

    fn from_string(string: &str) -> Self {
        let (module_name, defining_class, identifier) = split_module_class_and_identifier(string);
        Self {
            module_name,
            identifier,
            defining_class,
        }
    }
}

impl std::fmt::Display for FunctionRefForTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.module_name, self.identifier)
    }
}

impl CallTarget<FunctionRefForTest> {
    fn with_receiver_class_for_test(
        mut self,
        receiver_class: String,
        context: &ModuleContext,
    ) -> Self {
        self.receiver_class = Some({
            let (module_name, class_name) = split_module_and_identifier(&receiver_class);
            get_class_ref(&module_name, &class_name, context)
        });
        self
    }
}

enum TargetType {
    Function,
    Override,
    #[allow(dead_code)]
    Object,
}

fn create_call_target(target: &str, target_type: TargetType) -> CallTarget<FunctionRefForTest> {
    CallTarget {
        target: match target_type {
            TargetType::Function => Target::Function(FunctionRefForTest::from_string(target)),
            TargetType::Override => Target::Override(FunctionRefForTest::from_string(target)),
            TargetType::Object => Target::Object(target.to_owned()),
        },
        implicit_receiver: ImplicitReceiver::False,
        implicit_dunder_call: false,
        is_class_method: false,
        is_static_method: false,
        receiver_class: None,
        return_type: Some(ScalarTypeProperties::none()),
    }
}

fn call_graph_for_test_from_actual(
    call_graphs: CallGraphs<FunctionRef>,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
) -> CallGraphs<FunctionRefForTest> {
    let create_function_ref_for_test = |function_ref| {
        FunctionRefForTest::from_definition_ref(function_ref, function_base_definitions)
    };
    CallGraphs::from_map(
        call_graphs
            .into_iter()
            .map(|(caller, callees)| {
                let caller = create_function_ref_for_test(caller);
                let callees_for_test = CallGraph::from_map(
                    callees
                        .into_iter()
                        .map(|(location, expression_callees)| {
                            (
                                location.clone(),
                                expression_callees.map_function(create_function_ref_for_test),
                            )
                        })
                        .collect::<HashMap<_, _>>(),
                );
                (caller, callees_for_test)
            })
            .collect::<HashMap<_, _>>(),
    )
}

fn call_graph_for_test_from_expected(
    call_graph: Vec<(String, Vec<(String, ExpressionCallees<FunctionRefForTest>)>)>,
) -> CallGraphs<FunctionRefForTest> {
    CallGraphs::from_map(
        call_graph
            .into_iter()
            .map(|(caller, callees)| {
                let callees_for_test = CallGraph::from_map(
                    callees
                        .into_iter()
                        .map(|(location, expression_callees_for_test)| {
                            (
                                PysaLocation::from_key(&location).unwrap(),
                                expression_callees_for_test,
                            )
                        })
                        .collect::<HashMap<_, _>>(),
                );
                (FunctionRefForTest::from_string(&caller), callees_for_test)
            })
            .collect::<HashMap<_, _>>(),
    )
}

fn sort_call_graphs(
    call_graphs: &CallGraphs<FunctionRefForTest>,
) -> BTreeMap<FunctionRefForTest, BTreeMap<String, ExpressionCallees<FunctionRefForTest>>> {
    call_graphs
        .iter()
        .map(|(caller, callees)| {
            let sorted_callees = callees
                .iter()
                .map(|(location, expression_callees)| {
                    (location.as_key(), expression_callees.clone())
                })
                .collect::<BTreeMap<_, _>>();
            (caller.clone(), sorted_callees)
        })
        .collect::<BTreeMap<_, _>>()
}

fn test_building_call_graph_for_module(
    test_module_name: &str,
    code: &str,
    create_expected: &dyn Fn(
        &ModuleContext,
    )
        -> Vec<(String, Vec<(String, ExpressionCallees<FunctionRefForTest>)>)>,
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
    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    let expected_call_graph = call_graph_for_test_from_expected(create_expected(&context));

    let override_graph =
        OverrideGraph::from_reversed(&reversed_override_graph, &function_base_definitions);
    let mut actual_call_graph = call_graph_for_test_from_actual(
        export_call_graphs(&context, &function_base_definitions, &override_graph),
        &function_base_definitions,
    );
    // We don't care about callables that are not specified in the expected call graphs
    actual_call_graph.intersect(&expected_call_graph);

    assert_eq!(
        sort_call_graphs(&expected_call_graph),
        sort_call_graphs(&actual_call_graph)
    );
}

fn call_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
    init_targets: Vec<CallTarget<FunctionRefForTest>>,
    new_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Call(CallCallees {
        call_targets: call_targets.to_vec(),
        init_targets: init_targets.to_vec(),
        new_targets: new_targets.to_vec(),
    })
}

#[allow(dead_code)]
fn attribute_access_callees(
    expected: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        callable_targets: expected.to_vec(),
    })
}

fn identifier_callees(
    expected: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
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
            ("3:3-3:8".to_owned(), call_callees(vec![create_call_target("test.bar", TargetType::Function)], /* init_targets */ vec![], /* new_targets */ vec![])),
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
            create_call_target("test.C.m", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:3-6:8".to_owned(), call_callees(call_target.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
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
                ("6:9-6:12".to_owned(), identifier_callees(vec![create_call_target("test.bar", TargetType::Function).with_return_type(Some(ScalarTypeProperties::bool()))])),
                ("8:9-8:12".to_owned(), identifier_callees(vec![create_call_target("test.baz", TargetType::Function).with_return_type(Some(ScalarTypeProperties::int()))])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_method_call_on_optional_class,
    TEST_MODULE_NAME,
    r#"
from typing import Optional
class C:
  def m(self):
    ...
def foo(c: Optional[C]):
 if c is not None:
    c.m()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.C.m", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("8:5-8:10".to_owned(), call_callees(call_target.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_method_call_on_class_with_inheritance,
    TEST_MODULE_NAME,
    r#"
class C:
  def m(self):
    ...
class D(C):
  def m(self):
    ...
class E(D):
  def m(self):
    ...
def foo(c: C):
  c.m()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.C.m", TargetType::Override).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("12:3-12:8".to_owned(), call_callees(call_target.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
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
            create_call_target("test.C.f", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver).with_receiver_class_for_test("test.C".to_owned(), context).with_is_class_method(true).with_return_type(Some(ScalarTypeProperties::int()))
        ];
        let class_method_target_2 = vec![
            create_call_target("test.C.f", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context).with_is_class_method(true).with_return_type(Some(ScalarTypeProperties::int()))
        ];
        let method_target = vec![
            create_call_target("test.C.g", TargetType::Function).with_implicit_receiver(ImplicitReceiver::False).with_return_type(Some(ScalarTypeProperties::int()))
        ];
        let method_target_2 = vec![
            create_call_target("test.C.g", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context).with_return_type(Some(ScalarTypeProperties::int()))
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("7:3-7:8".to_owned(), call_callees(class_method_target.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
                ("8:3-8:8".to_owned(), call_callees(class_method_target_2.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
                ("9:3-9:9".to_owned(), call_callees(method_target.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
                ("10:3-10:8".to_owned(), call_callees(method_target_2.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_method_call_on_inherited_class_without_override,
    TEST_MODULE_NAME,
    r#"
class C:
  def m(self):
    ...
class D(C):
  pass
class E(D):
  def m(self):
    ...
def foo(d: D):
  d.m()
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.m", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.D".to_owned(), context),
            create_call_target("test.E.m", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.D".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("11:3-11:8".to_owned(), call_callees(call_targets.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_callable_instance_with_dunder_call_method,
    TEST_MODULE_NAME,
    r#"
class C:
  def __call__(self, a: int): ...
def foo(c: C):
   c(1)
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.__call__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_implicit_dunder_call(true).with_receiver_class_for_test("test.C".to_owned(), context)
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("5:4-5:8".to_owned(), call_callees(call_targets.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_callable_instance_with_static_dunder_call,
    TEST_MODULE_NAME,
    r#"
class C:
  @staticmethod
  def __call__(a: int) -> bool: ...
def foo(c: C):
   c(1)
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.__call__", TargetType::Function).with_implicit_dunder_call(true).with_is_static_method(true).with_return_type(Some(ScalarTypeProperties::bool()))
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:4-6:8".to_owned(), call_callees(call_targets.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_explicit_dunder_call_method_invocation,
    TEST_MODULE_NAME,
    r#"
class C:
  def __call__(self, a: int) -> bool: ...
def foo(c: C):
   c.__call__(1)
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.__call__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context).with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("5:4-5:17".to_owned(), call_callees(call_targets.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_function_passed_as_argument,
    TEST_MODULE_NAME,
    r#"
def bar():
  return
def baz(f):
  f()
def foo():
   baz(bar)
"#,
    &|_context: &ModuleContext| {
        let baz = vec![
            create_call_target("test.baz", TargetType::Function),
        ];
        let bar = vec![
            create_call_target("test.bar", TargetType::Function),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("7:4-7:12".to_owned(), call_callees(baz.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
                ("7:8-7:11".to_owned(), identifier_callees(bar.clone()))
            ],
        )]
    },
}

call_graph_testcase! {
    test_callable_protocol_instance,
    TEST_MODULE_NAME,
    r#"
from typing import Protocol
class C(Protocol):
  def __call__(self, a: int) -> bool: ...
def foo(c: C):
   c(1)
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.__call__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_return_type(Some(ScalarTypeProperties::bool())).with_implicit_dunder_call(true).with_receiver_class_for_test("test.C".to_owned(), context),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:4-6:8".to_owned(), call_callees(call_targets.clone(), /* init_targets */ vec![], /* new_targets */ vec![])),
            ],
        )]
    },
}

call_graph_testcase! {
    test_class_constructor,
    TEST_MODULE_NAME,
    r#"
class C:
  def __init__(self, a): ...
def foo():
  C(1)
"#,
    &|context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("test.C.__init__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.C".to_owned(), context),
        ];

        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("5:3-5:7".to_owned(), call_callees(/* call_targets */ vec![], init_targets, new_targets)),
            ],
        )]
    },
}

call_graph_testcase! {
    test_int_class_constructor,
    TEST_MODULE_NAME,
    r#"
def foo(x: str) -> int:
  return int(x)
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let new_targets = vec![
            create_call_target("builtins.int.__new__", TargetType::Function).with_is_static_method(true).with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("3:10-3:16".to_owned(), call_callees(/* call_targets */ vec![], init_targets, new_targets)),
            ],
        )]
    },
}

call_graph_testcase! {
    test_class_constructor_override_new,
    TEST_MODULE_NAME,
    r#"
class C:
  def __new__(cls, a): ...
def foo():
  C(1)
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("test.C.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("5:3-5:7".to_owned(), call_callees(/* call_targets */ vec![], init_targets, new_targets)),
            ],
        )]
    },
}

call_graph_testcase! {
    test_class_constructor_with_inheritance,
    TEST_MODULE_NAME,
    r#"
class A(): ...
class B(A):
  def __init__(self, a): ...
def foo():
  B(1)
"#,
    &|context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("test.B.__init__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver).with_receiver_class_for_test("test.B".to_owned(), context),
        ];

        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:3-6:7".to_owned(), call_callees(/* call_targets */ vec![], init_targets, new_targets)),
            ],
        )]
    },
}

call_graph_testcase! {
    test_class_constructor_override_new_with_inheritance,
    TEST_MODULE_NAME,
    r#"
class A(): ...
class B(A):
  def __new__(cls, a): ...
def foo():
  B(1)
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function).with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("test.B.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            TEST_DEFINITION_NAME.to_owned(),
            vec![
                ("6:3-6:7".to_owned(), call_callees(/* call_targets */ vec![], init_targets, new_targets)),
            ],
        )]
    },
}
