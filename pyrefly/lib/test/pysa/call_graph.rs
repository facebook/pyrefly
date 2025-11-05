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
use crate::report::pysa::call_graph::DefineCallees;
use crate::report::pysa::call_graph::ExpressionCallees;
use crate::report::pysa::call_graph::FunctionTrait;
use crate::report::pysa::call_graph::HigherOrderParameter;
use crate::report::pysa::call_graph::IdentifierCallees;
use crate::report::pysa::call_graph::ImplicitReceiver;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::Unresolved;
use crate::report::pysa::call_graph::UnresolvedReason;
use crate::report::pysa::call_graph::export_call_graphs;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionId;
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
    is_decorated_target: bool,
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
        let (function_id, is_decorated_target) = match function_ref.function_id {
            FunctionId::FunctionDecoratedTarget { location } => {
                (FunctionId::Function { location }, true)
            }
            function_id => (function_id, false),
        };
        Self {
            module_name: function_ref.module_name.to_string(),
            identifier: function_ref.function_name.to_string(),
            defining_class: function_base_definitions
                .get(function_ref.module_id, &function_id)
                .and_then(|definition| {
                    definition
                        .defining_class
                        .as_ref()
                        .map(|class| class.class.name().to_string())
                }),
            is_decorated_target,
        }
    }

    fn from_string(string: &str) -> Self {
        let (string, is_decorated_target) = match string.strip_suffix("@decorated") {
            Some(string) => (string, true),
            None => (string, false),
        };
        let (module_name, defining_class, identifier) = split_module_class_and_identifier(string);
        Self {
            module_name,
            identifier,
            defining_class,
            is_decorated_target,
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
        receiver_class: &str,
        context: &ModuleContext,
    ) -> Self {
        self.receiver_class = Some({
            let (module_name, class_name) = split_module_and_identifier(receiver_class);
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
                                expression_callees.map_function(&create_function_ref_for_test),
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
    call_graph: Vec<(&str, Vec<(&str, ExpressionCallees<FunctionRefForTest>)>)>,
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
                                PysaLocation::from_key(location).unwrap(),
                                expression_callees_for_test,
                            )
                        })
                        .collect::<HashMap<_, _>>(),
                );
                (FunctionRefForTest::from_string(caller), callees_for_test)
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
    ) -> Vec<(
        &'static str,
        Vec<(&'static str, ExpressionCallees<FunctionRefForTest>)>,
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

fn create_higher_order_parameters(
    inputs: Vec<(u32, Vec<CallTarget<FunctionRefForTest>>, Unresolved)>,
) -> HashMap<u32, HigherOrderParameter<FunctionRefForTest>> {
    inputs
        .into_iter()
        .map(|(index, call_targets, unresolved)| {
            (
                index,
                HigherOrderParameter {
                    index,
                    call_targets,
                    unresolved,
                },
            )
        })
        .collect()
}

fn call_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
    init_targets: Vec<CallTarget<FunctionRefForTest>>,
    new_targets: Vec<CallTarget<FunctionRefForTest>>,
    higher_order_parameters: Vec<(u32, Vec<CallTarget<FunctionRefForTest>>, Unresolved)>,
    unresolved: Unresolved,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Call(CallCallees {
        call_targets,
        init_targets,
        new_targets,
        higher_order_parameters: create_higher_order_parameters(higher_order_parameters),
        unresolved,
    })
}

fn define_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Define(DefineCallees {
        define_targets: call_targets,
    })
}

fn regular_call_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Call(CallCallees {
        call_targets,
        init_targets: vec![],
        new_targets: vec![],
        higher_order_parameters: HashMap::new(),
        unresolved: Unresolved::False,
    })
}

fn constructor_call_callees(
    init_targets: Vec<CallTarget<FunctionRefForTest>>,
    new_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Call(CallCallees {
        call_targets: vec![],
        init_targets,
        new_targets,
        higher_order_parameters: HashMap::new(),
        unresolved: Unresolved::False,
    })
}

fn attribute_access_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
    init_targets: Vec<CallTarget<FunctionRefForTest>>,
    new_targets: Vec<CallTarget<FunctionRefForTest>>,
    property_setters: Vec<CallTarget<FunctionRefForTest>>,
    property_getters: Vec<CallTarget<FunctionRefForTest>>,
    higher_order_parameters: Vec<(u32, Vec<CallTarget<FunctionRefForTest>>, Unresolved)>,
    unresolved: Unresolved,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        if_called: CallCallees {
            call_targets,
            init_targets,
            new_targets,
            higher_order_parameters: create_higher_order_parameters(higher_order_parameters),
            unresolved,
        },
        property_setters,
        property_getters,
    })
}

#[allow(dead_code)]
fn identifier_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
    init_targets: Vec<CallTarget<FunctionRefForTest>>,
    new_targets: Vec<CallTarget<FunctionRefForTest>>,
    higher_order_parameters: Vec<(u32, Vec<CallTarget<FunctionRefForTest>>, Unresolved)>,
    unresolved: Unresolved,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Identifier(IdentifierCallees {
        if_called: CallCallees {
            call_targets,
            init_targets,
            new_targets,
            higher_order_parameters: create_higher_order_parameters(higher_order_parameters),
            unresolved,
        },
    })
}

fn regular_identifier_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Identifier(IdentifierCallees {
        if_called: CallCallees {
            call_targets,
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        },
    })
}

static TEST_MODULE_NAME: &str = "test";

#[macro_export]
macro_rules! call_graph_testcase {
    ($name:ident, $module_name:ident, $code:literal, $create_expected:expr) => {
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

call_graph_testcase!(
    test_simple_function_call,
    TEST_MODULE_NAME,
    r#"
def foo():
  bar()
def bar():
  pass
"#,
    &|_context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![(
                "3:3-3:8",
                regular_call_callees(vec![create_call_target("test.bar", TargetType::Function)]),
            )],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.m", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("6:3-6:8", regular_call_callees(call_target))],
        )]
    }
);

call_graph_testcase!(
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
            "test.foo",
            vec![
                (
                    "6:9-6:12",
                    regular_identifier_callees(vec![
                        create_call_target("test.bar", TargetType::Function)
                            .with_return_type(Some(ScalarTypeProperties::bool())),
                    ]),
                ),
                (
                    "8:9-8:12",
                    regular_identifier_callees(vec![
                        create_call_target("test.baz", TargetType::Function)
                            .with_return_type(Some(ScalarTypeProperties::int())),
                    ]),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.m", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("8:5-8:10", regular_call_callees(call_target))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.m", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("12:3-12:8", regular_call_callees(call_target))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.f", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_is_class_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let class_method_target_2 = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_is_class_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let method_target = vec![
            create_call_target("test.C.g", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::False)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let method_target_2 = vec![
            create_call_target("test.C.g", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                ("7:3-7:8", regular_call_callees(class_method_target)),
                ("8:3-8:8", regular_call_callees(class_method_target_2)),
                ("9:3-9:9", regular_call_callees(method_target)),
                ("10:3-10:8", regular_call_callees(method_target_2)),
            ],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.m", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.D", context),
            create_call_target("test.E.m", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.D", context),
        ];
        vec![(
            "test.foo",
            vec![("11:3-11:8", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.__call__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_implicit_dunder_call(true)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("5:4-5:8", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.__call__", TargetType::Function)
                .with_implicit_dunder_call(true)
                .with_is_static_method(true)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        vec![(
            "test.foo",
            vec![("6:4-6:8", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.__call__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        vec![(
            "test.foo",
            vec![("5:4-5:17", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
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
        let baz = vec![create_call_target("test.baz", TargetType::Function)];
        let bar = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.foo",
            vec![
                (
                    "7:4-7:12",
                    call_callees(
                        baz,
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* higher_order_parameters */
                        vec![(0, bar.clone(), Unresolved::False)],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                ("7:8-7:11", regular_identifier_callees(bar)),
            ],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.__call__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::bool()))
                .with_implicit_dunder_call(true)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("6:4-6:8", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.C.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];

        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        vec![(
            "test.foo",
            vec![(
                "5:3-5:7",
                constructor_call_callees(init_targets, new_targets),
            )],
        )]
    }
);

call_graph_testcase!(
    test_int_class_constructor,
    TEST_MODULE_NAME,
    r#"
def foo(x: str) -> int:
  return int(x)
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let new_targets = vec![
            create_call_target("builtins.int.__new__", TargetType::Function)
                .with_is_static_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![(
                "3:10-3:16",
                constructor_call_callees(init_targets, new_targets),
            )],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("test.C.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            "test.foo",
            vec![(
                "5:3-5:7",
                constructor_call_callees(init_targets, new_targets),
            )],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("test.B.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.B", context),
        ];

        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        vec![(
            "test.foo",
            vec![(
                "6:3-6:7",
                constructor_call_callees(init_targets, new_targets),
            )],
        )]
    }
);

call_graph_testcase!(
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
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("test.B.__new__", TargetType::Function).with_is_static_method(true),
        ];
        vec![(
            "test.foo",
            vec![(
                "6:3-6:7",
                constructor_call_callees(init_targets, new_targets),
            )],
        )]
    }
);

call_graph_testcase!(
    test_property_setter_getter,
    TEST_MODULE_NAME,
    r#"
class C:
  @property
  def p(self) -> int: ...
  @p.setter
  def p(self, v: int) -> None: ...
def foo(c: C):
  c.p = c.p
"#,
    &|context: &ModuleContext| {
        let property_setters = vec![
            create_call_target("test.C.p", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        let property_getters = vec![
            create_call_target("test.C.p", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "8:3-8:6",
                    attribute_access_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ property_setters,
                        /* property_getters */ vec![],
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                (
                    "8:9-8:12",
                    attribute_access_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ vec![],
                        /* property_getters */ property_getters,
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_property_setter_with_property_getter_receiver,
    TEST_MODULE_NAME,
    r#"
class C:
  @property
  def p(self) -> "C":
    ...
  @p.setter
  def p(self, new_value: "C") -> None:
    ...
def foo(c: C):
  c.p.p = c
"#,
    &|context: &ModuleContext| {
        let property_setters = vec![
            create_call_target("test.C.p", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![(
                "10:3-10:8",
                attribute_access_callees(
                    /* call_targets */ vec![],
                    /* init_targets */ vec![],
                    /* new_targets */ vec![],
                    /* property_setters */ property_setters,
                    /* property_getters */ vec![],
                    /* higher_order_parameters */ vec![],
                    /* unresolved */ Unresolved::False,
                ),
            )],
        )]
    }
);

call_graph_testcase!(
    test_property_getter_with_union_types,
    TEST_MODULE_NAME,
    r#"
class C:
  @property
  def foo(self) -> int:
    return 0
class D:
  @property
  def foo(self) -> bool:
    return True
class E:
  foo: int = 1
def foo(c_or_d: C | D, c_or_e: C | E):
  x = c_or_d.foo
  y = c_or_e.foo
"#,
    &|_context: &ModuleContext| {
        let property_getters_c_or_d = vec![
            // TODO: Missing return type
            create_call_target("test.C.foo", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
            create_call_target("test.D.foo", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let property_getters_c_or_e = vec![
            create_call_target("test.C.foo", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "13:7-13:17",
                    attribute_access_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ vec![],
                        /* property_getters */ property_getters_c_or_d,
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                (
                    "14:7-14:17",
                    attribute_access_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ vec![],
                        /* property_getters */ property_getters_c_or_e,
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_property_getter_with_typevar,
    TEST_MODULE_NAME,
    r#"
from typing import TypeVar
class C:
    @property
    def foo(self) -> int:
      return 0
class D:
    @property
    def foo(self) -> int:
      return 0
TCOrD = TypeVar("TCOrD", C, D)
def foo(c_or_d: TCOrD):
    x = c_or_d.foo
"#,
    &|_context: &ModuleContext| {
        let property_getters = vec![
            create_call_target("test.C.foo", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![(
                "13:9-13:19",
                attribute_access_callees(
                    /* call_targets */ vec![],
                    /* init_targets */ vec![],
                    /* new_targets */ vec![],
                    /* property_setters */ vec![],
                    /* property_getters */ property_getters,
                    /* higher_order_parameters */ vec![],
                    /* unresolved */ Unresolved::False,
                ),
            )],
        )]
    }
);

call_graph_testcase!(
    test_static_method,
    TEST_MODULE_NAME,
    r#"
class C:
  @staticmethod
  def f(a: int) -> int: ...
def foo():
  C.f(1)
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_is_static_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![("6:3-6:9", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_classmethod_override_in_conditional_block,
    TEST_MODULE_NAME,
    r#"
# Original code: https://fburl.com/code/k6hypgar
class A:
  def foo(cls) -> None:
    raise NotImplementedError
class B(A):
  # The type of B.foo would be different without the if-else here.
  if 1 == 1:
    @classmethod
    def foo(cls) -> None:
      pass
  else:
    @classmethod
    def foo(cls) -> None:
      pass
def bar():
  B.foo()
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.B.foo", TargetType::Function)
                .with_is_class_method(true)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_receiver_class_for_test("test.B", context),
        ];
        vec![(
            "test.bar",
            vec![("17:3-17:10", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_abstract_method_call,
    TEST_MODULE_NAME,
    r#"
from abc import abstractmethod
class A:
  def f(self):
      return self.g()
  @abstractmethod
  def g(self):
      pass
class B(A):
  def g(self):
      pass
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.A.g", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.A", context),
        ];
        vec![(
            "test.A.f",
            vec![("5:14-5:22", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_overriden_repr_call,
    TEST_MODULE_NAME,
    r#"
class C:
  def __repr__(self) -> str: ...
def foo(c: C):
  repr(c)
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.__repr__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.foo",
            vec![("5:3-5:10", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_default_repr_call,
    TEST_MODULE_NAME,
    r#"
class C:
  pass
def foo(c: C):
  repr(c)
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![create_call_target("builtins.repr", TargetType::Function)];
        vec![(
            "test.foo",
            vec![("5:3-5:10", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_default_parameter_call,
    TEST_MODULE_NAME,
    r#"
def bar():
  pass
def foo(x=bar()):
  pass
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.foo",
            vec![("4:11-4:16", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_super_method_call,
    TEST_MODULE_NAME,
    r#"
class C:
  def f(self, x: int) -> int:
    return x
class D(C):
  def f(self, x: int) -> int:
    return x
  def g(self) -> None:
    super().f(1)
"#,
    &|context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.super.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.super", context),
        ];
        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        let call_targets = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::int()))
                .with_receiver_class_for_test("test.C", context),
        ];
        vec![(
            "test.D.g",
            vec![
                (
                    "9:5-9:12",
                    constructor_call_callees(init_targets, new_targets),
                ),
                ("9:5-9:17", regular_call_callees(call_targets)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_explicit_class_method_call_with_instance,
    TEST_MODULE_NAME,
    r#"
class C:
  def f(self, x: int) -> int:
    return x
class D(C):
  def f(self, x: int) -> int:
    return x
def foo(c: C):
  C.f(c, 1)
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![("9:3-9:12", regular_call_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_decorated_target,
    TEST_MODULE_NAME,
    r#"
def bar(f):
  return f
@bar
def foo():
  pass
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.foo@decorated",
            vec![("4:2-4:5", regular_identifier_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_classmethod_overrides,
    TEST_MODULE_NAME,
    r#"
class C:
  @classmethod
  def f(cls, x: int) -> int:
    return x
  @classmethod
  def g(cls):
    pass
class D(C):
  @classmethod
  def f(cls, x: int) -> int:
    return x
def foo(c: C):
  C.f(1)
  D.f(1)
  D.g()
"#,
    &|context: &ModuleContext| {
        let call_targets_c_f = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_receiver_class_for_test("test.C", context)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_return_type(Some(ScalarTypeProperties::int()))
                .with_is_class_method(true),
        ];
        let call_targets_d_f = vec![
            create_call_target("test.D.f", TargetType::Function)
                .with_receiver_class_for_test("test.D", context)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_return_type(Some(ScalarTypeProperties::int()))
                .with_is_class_method(true),
        ];
        let call_targets_d_g = vec![
            create_call_target("test.C.g", TargetType::Function)
                .with_receiver_class_for_test("test.D", context)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_is_class_method(true),
        ];
        vec![(
            "test.foo",
            vec![
                ("14:3-14:9", regular_call_callees(call_targets_c_f)),
                ("15:3-15:9", regular_call_callees(call_targets_d_f)),
                ("16:3-16:8", regular_call_callees(call_targets_d_g)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_higher_order_function_call,
    TEST_MODULE_NAME,
    r#"
def hof(f, arg) -> bool:
  f(arg)
  return True
def bar(x) -> int:
  return 0
def foo():
  hof(bar, 1)
"#,
    &|_context: &ModuleContext| {
        let hof = vec![
            create_call_target("test.hof", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        let bar = vec![
            create_call_target("test.bar", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "8:3-8:14",
                    call_callees(
                        hof,
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* higher_order_parameters */
                        vec![(0, bar.clone(), Unresolved::False)],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                ("8:7-8:10", regular_identifier_callees(bar)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_higher_order_function_with_multiple_callables,
    TEST_MODULE_NAME,
    r#"
def hof(f, g, arg) -> bool:
  f(arg)
  g(arg)
  return True
def foo(x) -> int:
  return 0
def bar(x) -> int:
  return 0
def main():
  hof(foo, bar, 1)
"#,
    &|_context: &ModuleContext| {
        let hof = vec![
            create_call_target("test.hof", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        let foo = vec![
            create_call_target("test.foo", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let bar = vec![
            create_call_target("test.bar", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.main",
            vec![
                (
                    "11:3-11:19",
                    call_callees(
                        hof,
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* higher_order_parameters */
                        vec![
                            (0, foo.clone(), Unresolved::False),
                            (1, bar.clone(), Unresolved::False),
                        ],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                ("11:12-11:15", regular_identifier_callees(bar)),
                ("11:7-11:10", regular_identifier_callees(foo)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_lambda_argument_as_higher_order_parameter,
    TEST_MODULE_NAME,
    r#" 
def foo():
  return map(lambda x: x, [0])
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("builtins.map.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        vec![(
            "test.foo",
            vec![(
                "3:10-3:31",
                call_callees(
                    /* call_targets */ vec![],
                    init_targets,
                    new_targets,
                    /* higher_order_parameters */
                    vec![(
                        0,
                        vec![],
                        Unresolved::True(UnresolvedReason::LambdaArgument),
                    )],
                    /* unresolved */ Unresolved::False,
                ),
            )],
        )]
    }
);

call_graph_testcase!(
    test_property_access_on_property_decorator_alias,
    TEST_MODULE_NAME,
    r#" 
from typing import List, Any
_magic_enum = property
class Enum:
  @_magic_enum
  def value(self) -> List[Any]: ...
class Permission(Enum):
  @property
  def action_name(self) -> bool:
    if len(self.value):
        return True
    return False
"#,
    &|context: &ModuleContext| {
        let len = vec![
            create_call_target("builtins.len", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let enum_value = vec![
            create_call_target("test.Enum.value", TargetType::Function)
                .with_receiver_class_for_test("test.Permission", context)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        vec![(
            "test.Permission.action_name",
            vec![
                (
                    "10:8-10:23",
                    call_callees(
                        len,
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* higher_order_parameters */
                        vec![(0, enum_value.clone(), Unresolved::False)],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                (
                    "10:12-10:22",
                    attribute_access_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ vec![],
                        /* property_getters */ enum_value,
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_builder_pattern_method_chaining,
    TEST_MODULE_NAME,
    r#"
from typing import Optional
class Builder:
    def __init__(self) -> None:
        self._saved: Optional[str] = None
        self._not_saved: Optional[str] = None
    def set_saved(self, saved: str) -> "Builder":
        self._saved = saved
        return self
    def set_not_saved(self, not_saved: str) -> "Builder":
        self._not_saved = not_saved
        return self
def foo():
    builder = Builder()
    builder.set_not_saved("true").set_saved("false")
"#,
    &|context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("test.Builder.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Builder", context),
        ];
        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        let set_not_saved = vec![
            create_call_target("test.Builder.set_not_saved", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Builder", context),
        ];
        let set_saved = vec![
            create_call_target("test.Builder.set_saved", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Builder", context),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "14:15-14:24",
                    constructor_call_callees(init_targets, new_targets),
                ),
                ("15:5-15:34", regular_call_callees(set_not_saved)),
                ("15:5-15:53", regular_call_callees(set_saved)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_decorated_callable_class_infinite_recursion,
    TEST_MODULE_NAME,
    r#"
from typing import Any, Callable
def to_c(callable: Callable[..., Any]) -> C:
  ...
class C:
  @to_c
  def __call__(self) -> "C":
    return self
def foo(c: C) -> None:
  x = c
"#,
    &|_context: &ModuleContext| { vec![("test.foo", vec![],)] }
);

call_graph_testcase!(
    test_protocol_method_calls,
    TEST_MODULE_NAME,
    r#"
from typing import Protocol
class C(Protocol):
  def f(self) -> int: ...
def foo(c: C):
  c.f()
  C.f(c)
"#,
    &|context: &ModuleContext| {
        let c_f = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let c_f_explicit = vec![
            create_call_target("test.C.f", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                ("6:3-6:8", regular_call_callees(c_f)),
                ("7:3-7:9", regular_call_callees(c_f_explicit)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_nested_function_call_in_outer_function,
    TEST_MODULE_NAME,
    r#"
def outer(x: int) -> None:
  def inner(x: int) -> int:
    return 0
  inner(x)
"#,
    &|_context: &ModuleContext| {
        let inner_target = vec![
            create_call_target("test.inner", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.outer",
            vec![
                ("5:3-5:11", regular_call_callees(inner_target.clone())),
                ("3:3-4:13", define_callees(inner_target)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_nested_function_call_in_class_method,
    TEST_MODULE_NAME,
    r#"
class Foo:
  def outer(self, x: int) -> None:
    def inner(x: int) -> int:
      return 0
    inner(x)
"#,
    &|_context: &ModuleContext| {
        let inner_target = vec![
            create_call_target("test.inner", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.Foo.outer",
            vec![
                ("6:5-6:13", regular_call_callees(inner_target.clone())),
                ("4:5-5:15", define_callees(inner_target)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_property_returning_callable,
    TEST_MODULE_NAME,
    r#"
from typing import Callable
class C:
  @property
  def attribute(self) -> Callable[[], int]:
    return lambda: 0
def foo(c: C) -> int:
  return c.attribute()
"#,
    &|_context: &ModuleContext| { vec![("test.foo", vec![])] }
);

call_graph_testcase!(
    test_try_finally_with_return_tracks_both_branches,
    TEST_MODULE_NAME,
    r#"
def foo() -> None:
  pass
def bar() -> None:
  pass
def main(x) -> None:
  try:
    return foo()
  finally:
    bar()
"#,
    &|_context: &ModuleContext| {
        let foo_target = vec![create_call_target("test.foo", TargetType::Function)];
        let bar_target = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.main",
            vec![
                ("10:5-10:10", regular_call_callees(bar_target)),
                ("8:12-8:17", regular_call_callees(foo_target)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_try_finally_with_raise_tracks_finally_branch_only,
    TEST_MODULE_NAME,
    r#"
def foo() -> None:
  pass
def bar() -> None:
  pass
def main(x) -> None:
  try:
    raise Exception()
  finally:
    bar()
"#,
    &|_context: &ModuleContext| {
        let init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        let bar_target = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.main",
            vec![
                ("10:5-10:10", regular_call_callees(bar_target)),
                (
                    "8:11-8:22",
                    constructor_call_callees(init_targets, new_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_list_of_methods_and_functions_called_in_loop,
    TEST_MODULE_NAME,
    r#"
class Foo:
  def bar(self) -> None:
    pass
def baz() -> None:
  pass
def f(foo: Foo):
  for g in [foo.bar, baz]:
    g()
"#,
    &|context: &ModuleContext| {
        // TODO(T105570363): Resolve calls with mixed function and methods
        let foo_bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Foo", context),
        ];
        let baz = vec![create_call_target("test.baz", TargetType::Function)];
        vec![(
            "test.f",
            vec![
                (
                    "8:13-8:20",
                    attribute_access_callees(
                        /* call_targets */ foo_bar.clone(),
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* property_setters */ vec![],
                        /* property_getters */ vec![],
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                ("8:22-8:25", regular_identifier_callees(baz)),
                ("9:5-9:8", regular_call_callees(foo_bar)),
            ],
        )]
    }
);

call_graph_testcase!(
    test_list_of_classes_and_functions_called_in_loop,
    TEST_MODULE_NAME,
    r#"
class Foo:
  def __init__(self) -> None: ...
def bar() -> None:
  pass
def f():
  for g in [Foo, bar]:
    g()
"#,
    &|context: &ModuleContext| {
        // TODO(T105570363): Resolve calls with mixed function and constructors
        let init_targets = vec![
            create_call_target("test.Foo.__init__", TargetType::Function)
                .with_receiver_class_for_test("test.Foo", context)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let bar = vec![create_call_target("test.bar", TargetType::Function)];
        let new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        vec![(
            "test.f",
            vec![
                (
                    "7:13-7:16",
                    identifier_callees(
                        /* call_targets */ vec![],
                        /* init_targets */ init_targets.clone(),
                        /* new_targets */ new_targets.clone(),
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                ("7:18-7:21", regular_identifier_callees(bar)),
                (
                    "8:5-8:8",
                    constructor_call_callees(init_targets, new_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_decorator_with_paramspec_on_function,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
@timer("bar")
def foo(x: int) -> int:
  return x
def caller() -> None:
  foo(1)
"#,
    &|_context: &ModuleContext| {
        let foo = vec![
            create_call_target("test.foo", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("14:3-14:9", regular_call_callees(foo))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_with_paramspec_on_method,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
class Foo:
  @timer("bar")
  def bar(self, x: int) -> int:
    return x
def caller(foo: Foo) -> None:
  foo.bar(1)
"#,
    &|context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Foo", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("15:3-15:13", regular_call_callees(bar))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_with_ellipsis_on_function,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar
_T = TypeVar("_T")
class Timer:
  def __call__(self, func: Callable[..., _T]) -> Callable[..., _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
@timer("bar")
def foo(x: int) -> int:
  return x
def caller() -> None:
  foo(1)
"#,
    &|_context: &ModuleContext| {
        let foo = vec![
            create_call_target("test.foo", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("13:3-13:9", regular_call_callees(foo))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_with_ellipsis_on_method,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar
_T = TypeVar("_T")
class Timer:
  def __call__(self, func: Callable[..., _T]) -> Callable[..., _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
class Foo:
  @timer("bar")
  def bar(self, x: int) -> int:
    return x
def caller(foo: Foo) -> None:
  foo.bar(1)
"#,
    &|context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Foo", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("14:3-14:13", regular_call_callees(bar))],
        )]
    }
);

call_graph_testcase!(
    test_simple_decorator_on_function,
    TEST_MODULE_NAME,
    r#"
from typing import Callable
def timer(name: str) -> Callable: ...
@timer("bar")
def foo(x: int) -> int:
  return x
def caller() -> None:
  foo(1)
"#,
    &|_context: &ModuleContext| {
        let foo = vec![create_call_target("test.foo", TargetType::Function)];
        vec![("test.caller", vec![("8:3-8:9", regular_call_callees(foo))])]
    }
);

call_graph_testcase!(
    test_simple_decorator_on_method,
    TEST_MODULE_NAME,
    r#"
from typing import Callable
def timer(name: str) -> Callable: ...
class Foo:
  @timer("bar")
  def bar(self, x: int) -> int:
    return x
def caller(foo: Foo) -> None:
  foo.bar(1)
"#,
    &|context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Foo", context),
        ];
        vec![("test.caller", vec![("9:3-9:13", regular_call_callees(bar))])]
    }
);

call_graph_testcase!(
    test_decorator_on_classmethod,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
class Foo:
  @classmethod
  @timer("bar")
  def bar(cls, x: int) -> int:
    return x
def caller() -> None:
  Foo.bar(1)
"#,
    &|context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_receiver_class_for_test("test.Foo", context)
                .with_is_class_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("16:3-16:13", regular_call_callees(bar))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_on_staticmethod,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
class Foo:
  @staticmethod
  @timer("bar")
  def bar(x: int) -> int:
    return x
def caller() -> None:
  Foo.bar(1)
"#,
    &|_context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_is_static_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("16:3-16:13", regular_call_callees(bar))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_on_classmethod_calling_from_classmethod,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
class Foo:
  @classmethod
  @timer("bar")
  def bar(cls, x: int) -> int:
    return x
  @classmethod
  def caller(cls) -> None:
    cls.bar(1)
"#,
    &|_context: &ModuleContext| {
        let bar = vec![
            create_call_target("test.Foo.bar", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_is_class_method(true)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.Foo.caller",
            vec![("17:5-17:15", regular_call_callees(bar))],
        )]
    }
);

call_graph_testcase!(
    test_decorator_with_type_error_does_not_affect_call_graph,
    TEST_MODULE_NAME,
    r#"
from typing import Callable, TypeVar, ParamSpec
_T = TypeVar("_T")
_TParams = ParamSpec("_TParams")
class Timer:
  def __call__(self, func: Callable[_TParams, _T]) -> Callable[_TParams, _T]:
    return func
def timer(name: str) -> Timer:
  return Timer()
@timer(1) # Intended type error here.
def foo(x: int) -> int:
  return x
def caller() -> None:
  foo(1)
"#,
    &|_context: &ModuleContext| {
        let foo = vec![
            create_call_target("test.foo", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.caller",
            vec![("14:3-14:9", regular_call_callees(foo))],
        )]
    }
);
