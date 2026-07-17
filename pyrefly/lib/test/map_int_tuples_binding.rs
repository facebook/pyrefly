/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_graph::index::Idx;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_types::quantified::QuantifiedOrigin;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Type;
use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
use ruff_text_size::Ranged;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::Bindings;
use crate::state::load::FileContents;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::TestEnv;

const SHAPE_EXTENSIONS: &str = "class MapIntTuples: ...\n";
const CONSUMER: &str = "from mapmod import anchor\n\nvalue = anchor\n";

fn map_module(parameter: &str) -> String {
    format!(
        r#"
from shape_extensions import MapIntTuples

anchor: int = 0
mapped: MapIntTuples[lambda {parameter}: tuple[{parameter}], int]
"#,
    )
}

fn lazy_map_state(parameter: &str) -> (State, Handle, Handle) {
    let mut env = TestEnv::new();
    env.add("shape_extensions", SHAPE_EXTENSIONS);
    env.add("main", CONSUMER);
    env.add("mapmod", &map_module(parameter));
    let sys_info = env.sys_info();
    let handle = |name: &str| {
        Handle::new(
            ModuleName::from_str(name),
            ModulePath::memory(PathBuf::from(format!("{name}.py"))),
            sys_info.dupe(),
        )
    };
    let (main, mapmod) = (handle("main"), handle("mapmod"));
    let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
    let mut transaction = state.new_committable_transaction(Require::Exports, None);
    transaction.as_mut().set_memory(env.get_memory());
    transaction
        .as_mut()
        .run(&[main.dupe()], Require::Exports, None);
    state.commit_transaction(transaction, None);
    (state, main, mapmod)
}

fn mapper_parameter(bindings: &Bindings) -> Idx<Key> {
    bindings
        .keys::<Key>()
        .find(|idx| matches!(bindings.get(*idx), Binding::TypeLevelLambdaParameter(_)))
        .expect("map module should bind an IntTuples mapper parameter")
}

fn mapped_annotation(bindings: &Bindings) -> Idx<Key> {
    let module = bindings.module();
    bindings
        .keys::<Key>()
        .find(|idx| {
            matches!(
                bindings.idx_to_key(*idx),
                Key::Definition(name) if module.code_at(name.range()) == "mapped"
            )
        })
        .expect("map module should bind the mapped annotation")
}

fn solve_mapper_parameter(state: &State, mapmod: &Handle) -> (bool, Type) {
    let transaction = state.transaction();
    let bindings = transaction
        .get_bindings(mapmod)
        .expect("lazily reached module should retain bindings");
    let answers = transaction
        .get_answers(mapmod)
        .expect("lazily reached module should retain answers");
    let parameter = mapper_parameter(&bindings);
    let annotation = mapped_annotation(&bindings);
    let both_unsolved =
        answers.get_idx(parameter).is_none() && answers.get_idx(annotation).is_none();

    let parameter = transaction
        .ad_hoc_solve(mapmod, "map_int_tuples_mapper_parameter", |solver| {
            solver.for_display(solver.get_idx(parameter).ty().clone())
        })
        .expect("map module should support an ad hoc solve");
    (both_unsolved, parameter)
}

fn assert_mapper_parameter(parameter: Type, expected_name: &str) {
    let Type::Type(parameter) = parameter else {
        panic!("mapper parameter should denote a type form, got `{parameter}`");
    };
    let Type::Quantified(parameter) = *parameter else {
        panic!("mapper parameter should denote a quantified, got `{parameter}`");
    };
    assert_eq!(parameter.name().as_str(), expected_name);
    assert_eq!(
        parameter.identity().origin,
        QuantifiedOrigin::MapIntTuplesParameter
    );
    let Restriction::Bound(bound) = parameter.restriction() else {
        panic!("mapper parameter should be bounded by IntTuple");
    };
    assert_eq!(bound, &IntTuple::shapeless().to_shape_arg_type());
}

#[test]
fn map_int_tuples_mapper_parameter_is_order_independent_and_incremental() {
    let (state, main, mapmod) = lazy_map_state("Elem");
    let (both_unsolved, parameter) = solve_mapper_parameter(&state, &mapmod);
    assert!(
        both_unsolved,
        "the parameter and enclosing annotation must be unsolved to test solve order"
    );
    assert_mapper_parameter(parameter, "Elem");

    let mut transaction = state.new_committable_transaction(Require::Everything, None);
    transaction.as_mut().set_memory(vec![(
        PathBuf::from("mapmod.py"),
        Some(Arc::new(FileContents::from_source(map_module("Item")))),
    )]);
    transaction
        .as_mut()
        .run(&[main.dupe(), mapmod.dupe()], Require::Everything, None);
    state.commit_transaction(transaction, None);

    let (_, parameter) = solve_mapper_parameter(&state, &mapmod);
    assert_mapper_parameter(parameter, "Item");
}

#[test]
fn map_int_tuples_mapper_recognition_follows_import_provenance() {
    let mut env = TestEnv::new();
    env.add("shape_extensions", SHAPE_EXTENSIONS);
    env.add(
        "reexport",
        "from shape_extensions import MapIntTuples as ReExported\n",
    );
    env.add("other", "class MapIntTuples: ...\n");
    env.add(
        "main",
        r#"
from shape_extensions import MapIntTuples as Direct
from shape_extensions import *
from reexport import ReExported
from other import MapIntTuples as OtherMap
import shape_extensions as se
import shape_extensions

Alias = MapIntTuples

direct: Direct[lambda DirectParam: DirectParam, int]
starred: MapIntTuples[lambda StarParam: StarParam, int]
module: se.MapIntTuples[lambda ModuleParam: ModuleParam, int]
module_direct: shape_extensions.MapIntTuples[lambda DirectModuleParam: DirectModuleParam, int]
assigned: Alias[lambda AssignedParam: AssignedParam, int]
reexported: ReExported[lambda ReexportedParam: ReexportedParam, int]
other: OtherMap[lambda OtherParam: OtherParam, int]
ordinary: tuple[lambda OrdinaryParam: OrdinaryParam]
second_argument: MapIntTuples[int, lambda SecondParam: SecondParam]

def shadowed() -> None:
    class MapIntTuples: ...
    value: MapIntTuples[lambda ShadowedParam: ShadowedParam, int]
"#,
    );
    let (state, handle) = env.to_state();
    let main = handle("main");
    let transaction = state.transaction();
    let bindings = transaction
        .get_bindings(&main)
        .expect("checked module should retain bindings");
    let module = bindings.module();
    let kind_for = |expected_name: &str| {
        bindings
            .keys::<Key>()
            .find_map(|idx| match (bindings.idx_to_key(idx), bindings.get(idx)) {
                (Key::Definition(name), binding)
                    if module.code_at(name.range()) == expected_name =>
                {
                    Some(matches!(binding, Binding::TypeLevelLambdaParameter(_)))
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("module should bind lambda parameter `{expected_name}`"))
    };

    assert!(kind_for("DirectParam"));
    assert!(kind_for("StarParam"));
    assert!(kind_for("ModuleParam"));
    assert!(kind_for("DirectModuleParam"));
    assert!(kind_for("AssignedParam"));
    assert!(kind_for("ReexportedParam"));
    assert!(!kind_for("OtherParam"));
    assert!(!kind_for("OrdinaryParam"));
    assert!(!kind_for("SecondParam"));
    assert!(!kind_for("ShadowedParam"));
}

#[test]
fn sibling_type_level_lambdas_may_reuse_a_parameter_name() {
    let mut env = TestEnv::new();
    env.add("shape_extensions", SHAPE_EXTENSIONS);
    env.add(
        "main",
        r#"
from shape_extensions import MapIntTuples

first: MapIntTuples[lambda S: tuple[S], int]
second: MapIntTuples[lambda S: list[S], int]
"#,
    );
    let (state, handle) = env.to_state();
    let transaction = state.transaction();
    let bindings = transaction
        .get_bindings(&handle("main"))
        .expect("checked module should retain bindings");
    let parameters = bindings
        .keys::<Key>()
        .filter_map(|idx| match bindings.get(idx) {
            Binding::TypeLevelLambdaParameter(parameter) => Some(parameter.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        parameters
            .iter()
            .all(|parameter| parameter.identifier.id.as_str() == "S")
    );
    let mut ids = parameters
        .iter()
        .map(|parameter| parameter.id.0)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 2);
}
