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
use crate::report::pysa::call_graph::ExpressionIdentifier;
use crate::report::pysa::call_graph::FormatStringArtificialCallees;
use crate::report::pysa::call_graph::FunctionTrait;
use crate::report::pysa::call_graph::HigherOrderParameter;
use crate::report::pysa::call_graph::IdentifierCallees;
use crate::report::pysa::call_graph::ImplicitReceiver;
use crate::report::pysa::call_graph::Target;
use crate::report::pysa::call_graph::Unresolved;
use crate::report::pysa::call_graph::UnresolvedReason;
use crate::report::pysa::call_graph::export_call_graphs;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::collect::CollectNoDuplicateKeys;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::FunctionBaseDefinition;
use crate::report::pysa::function::FunctionId;
use crate::report::pysa::function::FunctionRef;
use crate::report::pysa::function::WholeProgramFunctionDefinitions;
use crate::report::pysa::function::collect_function_base_definitions;
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
    is_property_setter: bool,
    is_class_toplevel: Option<ClassId>,
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
        let function_definition =
            function_base_definitions.get(function_ref.module_id, &function_id);
        let defining_class = function_definition.and_then(|definition| {
            definition
                .defining_class
                .as_ref()
                .map(|class| class.class.name().to_string())
        });
        let is_property_setter =
            function_definition.is_some_and(|definition| definition.is_property_setter);
        let is_class_toplevel = match function_id {
            FunctionId::ClassTopLevel { class_id } => Some(class_id),
            _ => None,
        };
        Self {
            module_name: function_ref.module_name.to_string(),
            identifier: function_ref.function_name.to_string(),
            defining_class,
            is_decorated_target,
            is_property_setter,
            is_class_toplevel,
        }
    }

    fn from_string(string: &str) -> Self {
        let (string, is_decorated_target) = match string.strip_suffix("@decorated") {
            Some(string) => (string, true),
            None => (string, false),
        };
        let (string, is_property_setter) = match string.strip_suffix("@setter") {
            Some(string) => (string, true),
            None => (string, false),
        };
        let (module_name, defining_class, identifier) = split_module_class_and_identifier(string);
        Self {
            module_name,
            identifier,
            defining_class,
            is_decorated_target,
            is_property_setter,
            is_class_toplevel: None,
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
    FormatString,
}

fn create_call_target(target: &str, target_type: TargetType) -> CallTarget<FunctionRefForTest> {
    CallTarget {
        target: match target_type {
            TargetType::Function => Target::Function(FunctionRefForTest::from_string(target)),
            TargetType::Override => Target::Override(FunctionRefForTest::from_string(target)),
            TargetType::Object => Target::Object(target.to_owned()),
            TargetType::FormatString => Target::FormatString,
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
    call_graphs: CallGraphs<ExpressionIdentifier, FunctionRef>,
    function_base_definitions: &WholeProgramFunctionDefinitions<FunctionBaseDefinition>,
) -> CallGraphs<String, FunctionRefForTest> {
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
                        .map(|(expression_identifier, expression_callees)| {
                            (
                                expression_identifier.as_key(),
                                expression_callees.map_function(&create_function_ref_for_test),
                            )
                        })
                        .collect_no_duplicate_keys()
                        .unwrap(),
                );
                (caller, callees_for_test)
            })
            .collect_no_duplicate_keys()
            .unwrap(),
    )
}

fn call_graph_for_test_from_expected(
    call_graph: Vec<(&str, Vec<(&str, ExpressionCallees<FunctionRefForTest>)>)>,
) -> CallGraphs<String, FunctionRefForTest> {
    CallGraphs::from_map(
        call_graph
            .into_iter()
            .map(|(caller, callees)| {
                let callees_for_test = CallGraph::from_map(
                    callees
                        .into_iter()
                        .map(|(expression_identifier, expression_callees_for_test)| {
                            (
                                expression_identifier.to_owned(),
                                expression_callees_for_test,
                            )
                        })
                        .collect_no_duplicate_keys()
                        .unwrap(),
                );
                (FunctionRefForTest::from_string(caller), callees_for_test)
            })
            .collect_no_duplicate_keys()
            .unwrap(),
    )
}

fn sort_call_graphs(
    call_graphs: CallGraphs<String, FunctionRefForTest>,
) -> BTreeMap<FunctionRefForTest, BTreeMap<String, ExpressionCallees<FunctionRefForTest>>> {
    call_graphs
        .into_iter()
        .map(|(caller, callees)| {
            let sorted_callees: BTreeMap<String, ExpressionCallees<FunctionRefForTest>> =
                callees.into_iter().collect();
            (caller, sorted_callees)
        })
        .collect()
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
        sort_call_graphs(expected_call_graph),
        sort_call_graphs(actual_call_graph)
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

#[allow(dead_code)]
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

fn property_getter_callees(
    property_getters: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        if_called: CallCallees {
            call_targets: vec![],
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        },
        property_setters: vec![],
        property_getters,
    })
}

fn property_setter_callees(
    property_setters: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        if_called: CallCallees {
            call_targets: vec![],
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        },
        property_setters,
        property_getters: vec![],
    })
}

fn regular_attribute_access_callees(
    call_targets: Vec<CallTarget<FunctionRefForTest>>,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
        if_called: CallCallees {
            call_targets,
            init_targets: vec![],
            new_targets: vec![],
            higher_order_parameters: HashMap::new(),
            unresolved: Unresolved::False,
        },
        property_setters: vec![],
        property_getters: vec![],
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

fn format_string_artificial_callees() -> ExpressionCallees<FunctionRefForTest> {
    let format_string_artificial_targets = vec![{
        let mut target = create_call_target("", TargetType::FormatString)
            .with_implicit_receiver(ImplicitReceiver::False);
        target.return_type = None;
        target
    }];
    ExpressionCallees::FormatStringArtificial(FormatStringArtificialCallees {
        targets: format_string_artificial_targets,
    })
}

fn format_string_stringify_callees(
    module_name: &str,
    class_name: &str,
    method_name: &str,
    target_type: TargetType,
    context: &ModuleContext,
) -> ExpressionCallees<FunctionRefForTest> {
    let format_call_targets = vec![
        create_call_target(
            &format!("{module_name}.{class_name}.{method_name}"),
            target_type,
        )
        .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
        .with_receiver_class_for_test(&format!("{module_name}.{class_name}"), context),
    ];
    regular_call_callees(format_call_targets)
}

fn unresolved_expression_callees(
    reason: UnresolvedReason,
) -> ExpressionCallees<FunctionRefForTest> {
    ExpressionCallees::Call(CallCallees::new_unresolved(reason))
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
                (
                    "9:3-9:6",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnsupportedFunctionTarget,
                    )),
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
            vec![
                (
                    "7:14-7:18|artificial-call|comparison",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnresolvedMagicDunderAttrDueToNoAttribute,
                    )),
                ),
                ("8:5-8:10", regular_call_callees(call_target)),
            ],
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
                ("8:3-8:6", property_setter_callees(property_setters)),
                ("8:9-8:12", property_getter_callees(property_getters)),
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
            vec![("10:3-10:8", property_setter_callees(property_setters))],
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
                    property_getter_callees(/* property_getters */ property_getters_c_or_d),
                ),
                (
                    "14:7-14:17",
                    property_getter_callees(/* property_getters */ property_getters_c_or_e),
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
                property_getter_callees(/* property_getters */ property_getters),
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
            vec![(
                "4:2-4:5|artificial-call|for-decorated-target",
                regular_call_callees(call_targets.clone()),
            )],
        )]
    }
);

call_graph_testcase!(
    test_decorated_target_decorator_factory,
    TEST_MODULE_NAME,
    r#"
class MyClass:
  def __init__(self) -> None:
    return
  def __call__(self, f):
    return f
def bar(x: int) -> MyClass:
  return MyClass()
@bar(1)
def foo():
  pass
"#,
    &|_context: &ModuleContext| {
        let call_targets = vec![create_call_target("test.bar", TargetType::Function)];
        vec![(
            "test.foo@decorated",
            vec![
                ("9:2-9:8", regular_call_callees(call_targets.clone())),
                (
                    "9:2-9:8|artificial-call|for-decorated-target",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnexpectedCalleeExpression,
                    )),
                ),
            ],
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
                        /* higher_order_parameters */ vec![],
                        /* unresolved */ Unresolved::False,
                    ),
                ),
                (
                    "10:12-10:22",
                    property_getter_callees(/* property_getters */ enum_value),
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
    &|_context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![(
                "8:10-8:23",
                ExpressionCallees::Call(CallCallees {
                    call_targets: vec![],
                    init_targets: vec![],
                    new_targets: vec![],
                    higher_order_parameters: HashMap::new(),
                    unresolved: Unresolved::False,
                }),
            )],
        )]
    }
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
                    regular_attribute_access_callees(foo_bar.clone()),
                ),
                ("8:22-8:25", regular_identifier_callees(baz)),
                (
                    "9:5-9:8",
                    call_callees(
                        /* call_targets */ foo_bar,
                        /* init_targets */ vec![],
                        /* new_targets */ vec![],
                        /* higher_order_parameters */ vec![],
                        /* unresolved */
                        Unresolved::True(UnresolvedReason::Mixed),
                    ),
                ),
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
                    ExpressionCallees::Call(CallCallees {
                        call_targets: vec![],
                        init_targets,
                        new_targets,
                        higher_order_parameters: HashMap::new(),
                        unresolved: Unresolved::True(UnresolvedReason::Mixed),
                    }),
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

call_graph_testcase!(
    test_call_via_getattr_return_self_type,
    TEST_MODULE_NAME,
    r#"
from typing import Union
class CallViaGetattr:
  def __getattr__(self, name: str) -> Union[None, CallViaGetattr]:
    return None
def baz(x: CallViaGetattr) -> None:
  y = x.attribute
"#,
    &|_context: &ModuleContext| { vec![("test.baz", vec![])] }
);

call_graph_testcase!(
    test_call_via_getattr_return_callable_class,
    TEST_MODULE_NAME,
    r#"
from typing import Union
class CallableClass:
  def __call__(self) -> None:
    return None
class CallViaGetattr:
  def __getattr__(self, name: str) -> Union[None, CallableClass]:
    return CallableClass()
def baz(x: CallViaGetattr) -> None:
  y = x.attribute
"#,
    &|context: &ModuleContext| {
        // TODO(T244609756): Add the implicit calls to `__getattr__`
        let call_targets = vec![
            create_call_target("test.CallableClass.__call__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_implicit_dunder_call(true)
                .with_receiver_class_for_test("test.CallableClass", context),
        ];
        vec![(
            "test.baz",
            vec![("10:7-10:18", regular_attribute_access_callees(call_targets))],
        )]
    }
);

call_graph_testcase!(
    test_attribute_access_on_typed_class_attributes,
    TEST_MODULE_NAME,
    r#"
class Token:
  token: str = ""
class Token2:
  token2: str = ""
def foo(obj: Token, obj2: Token2):
  x = (obj.token, obj2.token2)
"#,
    &|_context: &ModuleContext| { vec![("test.foo", vec![])] }
);

call_graph_testcase!(
    test_attribute_access_on_union_type,
    TEST_MODULE_NAME,
    r#"
from typing import Union
class A:
  attribute: str = ""
class B:
  attribute: str = ""
class C:
  attribute: str = ""
def foo(obj: Union[A, B, C]):
  x = obj.attribute
"#,
    &|_context: &ModuleContext| { vec![("test.foo", vec![])] }
);

call_graph_testcase!(
    test_chained_attribute_access_on_optional_type,
    TEST_MODULE_NAME,
    r#"
from typing import Optional
class Token:
  token: str = ""
class Request:
  access_token: Optional[Token]
def foo(request: Request):
  if request.access_token:
    x = request.access_token.token
"#,
    &|_context: &ModuleContext| { vec![("test.foo", vec![])] }
);

call_graph_testcase!(
    test_getattr_on_typed_class_attribute,
    TEST_MODULE_NAME,
    r#"
class Token:
  token: str = ""
def foo(obj: Token):
  x = getattr(obj, "token", None)
"#,
    &|_context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![
                (
                    "5:7-5:34",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnexpectedPyreflyTarget,
                    )),
                ),
                (
                    "5:7-5:34|artificial-attribute-access|get-attr-constant-literal",
                    ExpressionCallees::AttributeAccess(AttributeAccessCallees {
                        if_called: CallCallees {
                            call_targets: vec![],
                            init_targets: vec![],
                            new_targets: vec![],
                            higher_order_parameters: HashMap::new(),
                            unresolved: Unresolved::False,
                        },
                        property_setters: vec![],
                        property_getters: vec![],
                    }),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_getattr_on_methods,
    TEST_MODULE_NAME,
    r#"
class Token:
  def token(self) -> None: ...
def foo(obj: Token):
  getattr(obj, "token", None)()
"#,
    &|context: &ModuleContext| {
        let token = vec![
            create_call_target("test.Token.token", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.Token", context),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "5:3-5:30",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnexpectedPyreflyTarget,
                    )),
                ),
                (
                    "5:3-5:30|artificial-attribute-access|get-attr-constant-literal",
                    regular_attribute_access_callees(token),
                ),
                (
                    "5:3-5:32",
                    ExpressionCallees::Call(CallCallees::new_unresolved(
                        UnresolvedReason::UnexpectedCalleeExpression,
                    )),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_explicit_dunder_setattr_call,
    TEST_MODULE_NAME,
    r#"
class Token:
  token: str = ""
def foo(obj: Token, x: str):
  obj.__setattr__("token", x)
"#,
    &|_context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![(
                "5:3-5:30",
                ExpressionCallees::Call(CallCallees::new_unresolved(
                    UnresolvedReason::UnresolvedMagicDunderAttr,
                )),
            )],
        )]
    }
);

call_graph_testcase!(
    test_attribute_assignment_with_custom_setattr,
    TEST_MODULE_NAME,
    r#"
class Test:
  def __setattr__(self, name: str, value):
    return
def foo(obj: Test):
  obj.attribute = "value"
"#,
    &|_context: &ModuleContext| {
        // TODO(T137969662): We should see a call to `Test.__setattr__`
        vec![("test.foo", vec![])]
    }
);

call_graph_testcase!(
    test_comparison_operator,
    TEST_MODULE_NAME,
    r#"
def foo():
  1 > 2
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("builtins.int.__gt__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.int", context)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        vec![(
            "test.foo",
            vec![(
                "3:7-3:8|artificial-call|comparison",
                regular_call_callees(call_targets),
            )],
        )]
    }
);

call_graph_testcase!(
    test_chained_comparison_operators,
    TEST_MODULE_NAME,
    r#"
def foo():
  1 > 2 > 3
"#,
    &|context: &ModuleContext| {
        let call_targets = vec![
            create_call_target("builtins.int.__gt__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.int", context)
                .with_return_type(Some(ScalarTypeProperties::bool())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "3:7-3:8|artificial-call|comparison",
                    regular_call_callees(call_targets.clone()),
                ),
                (
                    "3:11-3:12|artificial-call|comparison",
                    regular_call_callees(call_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_list_comprehension_with_method_call,
    TEST_MODULE_NAME,
    r#"
from typing import List
class C:
  def run(self) -> str:
    return ""
def foo() -> None:
  cs: List[C] = [C()]
  result = [c.run() for c in cs]
"#,
    &|context: &ModuleContext| {
        let c_init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        let c_new_targets = vec![
            create_call_target("builtins.object.__new__", TargetType::Function)
                .with_is_static_method(true),
        ];
        let c_run = vec![
            create_call_target("test.C.run", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context),
        ];
        let iter_targets = vec![{
            create_call_target("builtins.list.__iter__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.list", context)
        }];
        let next_targets = vec![
            create_call_target("typing.Iterator.__next__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.Iterator", context),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "7:18-7:21",
                    constructor_call_callees(c_init_targets, c_new_targets),
                ),
                ("8:13-8:20", regular_call_callees(c_run)),
                (
                    "8:30-8:32|artificial-call|generator-iter",
                    regular_call_callees(iter_targets),
                ),
                (
                    "8:30-8:32|artificial-call|generator-next",
                    regular_call_callees(next_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_nested_list_comprehension_with_multiple_iterators,
    TEST_MODULE_NAME,
    r#"
def foo() -> None:
  container = range(3)
  result = [x + y for x in container for y in container]
"#,
    &|context: &ModuleContext| {
        let range_init_targets = vec![
            create_call_target("builtins.object.__init__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        let range_new_targets = vec![
            create_call_target("builtins.range.__new__", TargetType::Function)
                .with_is_static_method(true)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        let iter_targets = vec![{
            create_call_target("builtins.range.__iter__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.range", context)
        }];
        let next_targets = vec![
            create_call_target("typing.Iterator.__next__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.Iterator", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "3:15-3:23",
                    constructor_call_callees(range_init_targets, range_new_targets),
                ),
                (
                    "4:28-4:37|artificial-call|generator-iter",
                    regular_call_callees(iter_targets.clone()),
                ),
                (
                    "4:28-4:37|artificial-call|generator-next",
                    regular_call_callees(next_targets.clone()),
                ),
                (
                    "4:47-4:56|artificial-call|generator-iter",
                    regular_call_callees(iter_targets),
                ),
                (
                    "4:47-4:56|artificial-call|generator-next",
                    regular_call_callees(next_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_various_comprehensions,
    TEST_MODULE_NAME,
    r#"
import typing
class A:
  def f(self) -> typing.List[int]:
    return [1, 2]
def g() -> A:
  return A()
def id(arg):
  return arg
def foo(l0: typing.AsyncIterator[int], l1: typing.List[int], l2: typing.AsyncIterable[int]):
  x = [x async for x in l0]
  x = [x for x in l1]  # List comprehension
  x = [x async for x in l2]  # List comprehension
  x = [x for x in g().f()]  # Iterator as a compound AST node
  x = {x for x in l1}  # Set comprehension
  x = {x async for x in l2}  # Set comprehension
  x = {x:0 for x in l1}  # Dictionary comprehension
  x = {x:0 async for x in l2}  # Dictionary comprehension
  x = (x for x in l1) # Generator comprehension
  x = (x async for x in l2)  # Generator comprehension
"#,
    &|context: &ModuleContext| {
        let list_iter_targets = vec![{
            create_call_target("builtins.list.__iter__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.list", context)
        }];
        let list_next_targets = vec![
            create_call_target("typing.Iterator.__next__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.Iterator", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let async_iterator_aiter_targets = vec![{
            create_call_target("typing.AsyncIterator.__aiter__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.AsyncIterator", context)
        }];
        let async_iterator_anext_targets = vec![
            create_call_target("typing.AsyncIterator.__anext__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.AsyncIterator", context)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let async_iterable_aiter_targets = vec![{
            create_call_target("typing.AsyncIterable.__aiter__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("typing.AsyncIterable", context)
        }];
        let g_target = vec![create_call_target("test.g", TargetType::Function)];
        let a_f_target = vec![
            create_call_target("test.A.f", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.A", context),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "11:25-11:27|artificial-call|generator-iter",
                    regular_call_callees(async_iterator_aiter_targets.clone()),
                ),
                (
                    "11:25-11:27|artificial-call|generator-next",
                    regular_call_callees(async_iterator_anext_targets.clone()),
                ),
                (
                    "12:19-12:21|artificial-call|generator-iter",
                    regular_call_callees(list_iter_targets.clone()),
                ),
                (
                    "12:19-12:21|artificial-call|generator-next",
                    regular_call_callees(list_next_targets.clone()),
                ),
                (
                    "13:25-13:27|artificial-call|generator-iter",
                    regular_call_callees(async_iterable_aiter_targets.clone()),
                ),
                (
                    "13:25-13:27|artificial-call|generator-next",
                    regular_call_callees(async_iterator_anext_targets.clone()),
                ),
                ("14:19-14:22", regular_call_callees(g_target)),
                ("14:19-14:26", regular_call_callees(a_f_target)),
                (
                    "14:19-14:26|artificial-call|generator-iter",
                    regular_call_callees(list_iter_targets.clone()),
                ),
                (
                    "14:19-14:26|artificial-call|generator-next",
                    regular_call_callees(list_next_targets.clone()),
                ),
                (
                    "15:19-15:21|artificial-call|generator-iter",
                    regular_call_callees(list_iter_targets.clone()),
                ),
                (
                    "15:19-15:21|artificial-call|generator-next",
                    regular_call_callees(list_next_targets.clone()),
                ),
                (
                    "16:25-16:27|artificial-call|generator-iter",
                    regular_call_callees(async_iterable_aiter_targets.clone()),
                ),
                (
                    "16:25-16:27|artificial-call|generator-next",
                    regular_call_callees(async_iterator_anext_targets.clone()),
                ),
                (
                    "17:21-17:23|artificial-call|generator-iter",
                    regular_call_callees(list_iter_targets.clone()),
                ),
                (
                    "17:21-17:23|artificial-call|generator-next",
                    regular_call_callees(list_next_targets.clone()),
                ),
                (
                    "18:27-18:29|artificial-call|generator-iter",
                    regular_call_callees(async_iterable_aiter_targets.clone()),
                ),
                (
                    "18:27-18:29|artificial-call|generator-next",
                    regular_call_callees(async_iterator_anext_targets.clone()),
                ),
                (
                    "19:19-19:21|artificial-call|generator-iter",
                    regular_call_callees(list_iter_targets),
                ),
                (
                    "19:19-19:21|artificial-call|generator-next",
                    regular_call_callees(list_next_targets),
                ),
                (
                    "20:25-20:27|artificial-call|generator-iter",
                    regular_call_callees(async_iterable_aiter_targets),
                ),
                (
                    "20:25-20:27|artificial-call|generator-next",
                    regular_call_callees(async_iterator_anext_targets),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_context_manager,
    TEST_MODULE_NAME,
    r#"
from typing import ContextManager
def to_cm() -> ContextManager[int]: ...
def foo():
  with to_cm() as my_int:
    pass
"#,
    &|context: &ModuleContext| {
        let enter_target = create_call_target(
            "contextlib.AbstractContextManager.__enter__",
            TargetType::Override,
        )
        .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
        .with_receiver_class_for_test("contextlib.AbstractContextManager", context)
        .with_return_type(Some(ScalarTypeProperties::int()));
        vec![(
            "test.foo",
            vec![
                (
                    "5:8-5:15",
                    regular_call_callees(vec![create_call_target(
                        "test.to_cm",
                        TargetType::Function,
                    )]),
                ),
                (
                    "5:8-5:15|artificial-call|with-enter",
                    regular_call_callees(vec![enter_target]),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_async_context_manager,
    TEST_MODULE_NAME,
    r#"
from typing import AsyncContextManager
def to_cm() -> AsyncContextManager[int]: ...
async def foo():
  async with to_cm() as my_int:
    pass
"#,
    &|context: &ModuleContext| {
        let aenter_target = create_call_target(
            "contextlib.AbstractAsyncContextManager.__aenter__",
            TargetType::Override,
        )
        .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
        .with_receiver_class_for_test("contextlib.AbstractAsyncContextManager", context)
        .with_return_type(Some(ScalarTypeProperties::int()));
        vec![(
            "test.foo",
            vec![
                (
                    "5:14-5:21",
                    regular_call_callees(vec![create_call_target(
                        "test.to_cm",
                        TargetType::Function,
                    )]),
                ),
                (
                    "5:14-5:21|artificial-call|with-enter",
                    regular_call_callees(vec![aenter_target]),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_dict_subscript_getitem,
    TEST_MODULE_NAME,
    r#"
class C:
  @classmethod
  def foo(cls):
    pass
d = {
  "a": C,
  "b": C,
}
def calls_d_method(s: str):
  d[s].foo()
"#,
    &|context: &ModuleContext| {
        let call_target = vec![
            create_call_target("test.C.foo", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithClassReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_is_class_method(true),
        ];
        let dict_getitem_target = vec![
            create_call_target("builtins.dict.__getitem__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.dict", context),
        ];
        vec![(
            "test.calls_d_method",
            vec![
                ("11:3-11:13", regular_call_callees(call_target)),
                (
                    "11:3-11:7|artificial-call|subscript-get-item",
                    regular_call_callees(dict_getitem_target),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_dict_subscript_setitem,
    TEST_MODULE_NAME,
    r#"
import typing
def foo() -> typing.Dict[str, int]:
  return {"a": 0}
def bar():
  return 1
def baz():
  return "b"
def fun(d: typing.Dict[str, int], e: typing.Dict[str, typing.Dict[str, int]]):
  foo()["a"] = bar()
  d[baz()] = bar()
  e["a"]["b"] = 0
"#,
    &|context: &ModuleContext| {
        let foo_target = vec![
            create_call_target("test.foo", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        let bar_target = vec![
            create_call_target("test.bar", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::int())),
        ];
        let baz_target = vec![
            create_call_target("test.baz", TargetType::Function)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        let dict_setitem_target = vec![
            create_call_target("builtins.dict.__setitem__", TargetType::Override)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.dict", context),
        ];
        let dict_getitem_target = vec![
            create_call_target("builtins.dict.__getitem__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("builtins.dict", context)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        vec![(
            "test.fun",
            vec![
                ("10:3-10:8", regular_call_callees(foo_target)),
                (
                    "10:3-10:21|artificial-call|subscript-set-item",
                    regular_call_callees(dict_setitem_target.clone()),
                ),
                ("10:16-10:21", regular_call_callees(bar_target.clone())),
                ("11:5-11:10", regular_call_callees(baz_target)),
                (
                    "11:3-11:19|artificial-call|subscript-set-item",
                    regular_call_callees(dict_setitem_target.clone()),
                ),
                ("11:14-11:19", regular_call_callees(bar_target)),
                (
                    "12:3-12:9|artificial-call|subscript-get-item",
                    regular_call_callees(dict_getitem_target),
                ),
                (
                    "12:3-12:18|artificial-call|subscript-set-item",
                    regular_call_callees(dict_setitem_target),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_method_call_in_f_string,
    TEST_MODULE_NAME,
    r#"
class C:
  def m(self) -> str:
    return "world"
def foo(c: C) -> str:
  return f"hello {c.m()}"
"#,
    &|context: &ModuleContext| {
        let method_call_targets = vec![
            create_call_target("test.C.m", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver)
                .with_receiver_class_for_test("test.C", context)
                .with_return_type(Some(ScalarTypeProperties::none())),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "6:10-6:26|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                ("6:19-6:24", regular_call_callees(method_call_targets)),
                (
                    "6:19-6:24|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "str",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_f_string_with_multiple_typed_and_untyped_variables,
    TEST_MODULE_NAME,
    r#"
def foo(a: int, b: float, c: str, d: typing.List[int], e):
  w = [1, 2, 3]
  x = 1
  y = "str"
  z = 2.3
  return f"{a}{b}{c}{d}{w}{x}{y}{z}{e}"
"#,
    &|context: &ModuleContext| {
        let literal_string_target = vec![
            create_call_target("builtins.str.__format__", TargetType::Function)
                .with_implicit_receiver(ImplicitReceiver::TrueWithObjectReceiver),
        ];
        vec![(
            "test.foo",
            vec![
                (
                    "7:10-7:40|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                (
                    "7:13-7:14|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "int",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
                (
                    "7:16-7:17|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "float",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
                (
                    "7:19-7:20|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "str",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
                (
                    "7:22-7:23|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
                (
                    "7:25-7:26|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnknownClassField),
                ),
                (
                    "7:28-7:29|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "int",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
                (
                    "7:31-7:32|artificial-call|format-string-stringify",
                    regular_call_callees(literal_string_target),
                ),
                (
                    "7:34-7:35|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "float",
                        "__format__",
                        TargetType::Function,
                        context,
                    ),
                ),
                (
                    "7:37-7:38|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_f_string_concatenation_and_in_if_condition,
    TEST_MODULE_NAME,
    r#"
def foo(x) -> bool:
  y = f"{x}" f"{x}"
  if foo(f"{x}"):
    return True
  else:
    return True
"#,
    &|_context: &ModuleContext| {
        let foo = create_call_target("test.foo", TargetType::Function)
            .with_return_type(Some(ScalarTypeProperties::bool()));
        vec![(
            "test.foo",
            vec![
                (
                    "3:7-3:20|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                (
                    "3:10-3:11|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
                (
                    "3:17-3:18|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
                ("4:6-4:17", regular_call_callees(vec![foo])),
                (
                    "4:10-4:16|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                (
                    "4:13-4:14|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_f_string_with_object_type,
    TEST_MODULE_NAME,
    r#"
def foo(x: object):
  return f"{x}"
"#,
    &|context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![
                (
                    "3:10-3:16|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                (
                    "3:13-3:14|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "object",
                        "__format__",
                        TargetType::Override,
                        context,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_f_string_with_multiple_references_to_same_variable,
    TEST_MODULE_NAME,
    r#"
def foo(x: object):
  return f"{x}:{x}"
"#,
    &|context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![
                (
                    "3:10-3:20|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                (
                    "3:13-3:14|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "object",
                        "__format__",
                        TargetType::Override,
                        context,
                    ),
                ),
                (
                    "3:17-3:18|artificial-call|format-string-stringify",
                    format_string_stringify_callees(
                        "builtins",
                        "object",
                        "__format__",
                        TargetType::Override,
                        context,
                    ),
                ),
            ],
        )]
    }
);

call_graph_testcase!(
    test_f_string_with_any_type,
    TEST_MODULE_NAME,
    r#"
def foo(x: Any):
  return f"{x}"
"#,
    &|_context: &ModuleContext| {
        vec![(
            "test.foo",
            vec![
                (
                    "3:10-3:16|artificial-call|format-string-artificial",
                    format_string_artificial_callees(),
                ),
                // TODO(T112761296): Probably wrong call resolution. Expect an additional call target.
                (
                    "3:13-3:14|artificial-call|format-string-stringify",
                    unresolved_expression_callees(UnresolvedReason::UnexpectedPyreflyTarget),
                ),
            ],
        )]
    }
);
