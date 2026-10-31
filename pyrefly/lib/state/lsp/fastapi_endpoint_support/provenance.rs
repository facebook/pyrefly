/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Conservative, typed receiver classification for endpoint support.

use pyrefly_build::handle::Handle;
use pyrefly_types::class::Class;
use pyrefly_types::types::Type;
use ruff_text_size::TextRange;

use crate::state::state::Transaction;

/// Whether the expression at `receiver` is a canonical TestClient instance or subclass.
/// Missing or ambiguous types are not evidence of provenance.
pub(super) fn is_test_client(
    transaction: &Transaction<'_>,
    handle: &Handle,
    receiver: TextRange,
) -> bool {
    is_instance_of(
        transaction,
        handle,
        receiver,
        &[
            ("starlette.testclient", "TestClient"),
            ("fastapi.testclient", "TestClient"),
        ],
    )
}

/// Whether the expression at `receiver` is a known FastAPI or Starlette instance.
/// This classifies its type, not the identity of a particular application object.
pub(super) fn is_application(
    transaction: &Transaction<'_>,
    handle: &Handle,
    receiver: TextRange,
) -> bool {
    is_instance_of(
        transaction,
        handle,
        receiver,
        &[
            ("fastapi", "FastAPI"),
            ("fastapi.applications", "FastAPI"),
            ("starlette.applications", "Starlette"),
        ],
    )
}

fn is_instance_of(
    transaction: &Transaction<'_>,
    handle: &Handle,
    receiver: TextRange,
    qualified_names: &[(&str, &str)],
) -> bool {
    let Some(Type::ClassType(instance) | Type::SelfType(instance)) =
        transaction.get_type_trace(handle, receiver)
    else {
        return false;
    };
    let matches = |class: &Class| {
        qualified_names
            .iter()
            .any(|(module, name)| class.has_toplevel_qname(module, name))
    };
    let class = instance.class_object();
    if matches(class) {
        return true;
    }
    transaction
        .ad_hoc_solve(handle, "endpoint_receiver_provenance", |solver| {
            let mro = solver.get_mro_for_class(class);
            mro.linearization_complete()
                && mro
                    .ancestors_no_object()
                    .iter()
                    .any(|base| matches(base.class_object()))
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::is_application;
    use super::is_test_client;
    use crate::test::util::TestEnv;

    fn framework_env() -> TestEnv {
        let mut env = TestEnv::new();
        for (module, path, source) in [
            ("starlette", "starlette/__init__.pyi", ""),
            (
                "starlette.applications",
                "starlette/applications.pyi",
                "class Starlette: ...",
            ),
            (
                "starlette.testclient",
                "starlette/testclient.pyi",
                "class TestClient:\n    def __init__(self, app: object) -> None: ...",
            ),
            (
                "fastapi.applications",
                "fastapi/applications.pyi",
                "from starlette.applications import Starlette\nclass FastAPI(Starlette): ...",
            ),
            (
                "fastapi",
                "fastapi/__init__.pyi",
                "from fastapi.applications import FastAPI as FastAPI",
            ),
            (
                "fastapi.testclient",
                "fastapi/testclient.pyi",
                "from starlette.testclient import TestClient as TestClient",
            ),
        ] {
            env.add_with_path(module, path, source);
        }
        env
    }

    fn assert_provenance(mut env: TestEnv, source: &str, cases: &[(&str, bool, bool)]) {
        env.add("main", source);
        let (state, handles) = env.to_state();
        let handle = handles("main");
        let transaction = state.transaction();
        let errors = transaction.get_errors([&handle]).collect_errors().ordinary;
        assert!(errors.is_empty(), "Unexpected type errors: {errors:#?}");
        for &(expression, test_client, application) in cases {
            let start = source.rfind(expression).expect("probe expression exists");
            let range = TextRange::new(
                TextSize::try_from(start).unwrap(),
                TextSize::try_from(start + expression.len()).unwrap(),
            );
            assert!(
                transaction.get_type_trace(&handle, range).is_some(),
                "Missing type trace for {expression}",
            );
            assert_eq!(
                is_test_client(&transaction, &handle, range),
                test_client,
                "TestClient provenance for {expression}",
            );
            assert_eq!(
                is_application(&transaction, &handle, range),
                application,
                "Application provenance for {expression}",
            );
        }
        let missing = TextRange::empty(TextSize::new(0));
        assert!(!is_test_client(&transaction, &handle, missing));
        assert!(!is_application(&transaction, &handle, missing));
    }

    #[test]
    fn typed_aliases_and_reexports() {
        assert_provenance(
            framework_env(),
            r#"
from fastapi import FastAPI as App
from fastapi.testclient import TestClient as Reexported
from starlette.applications import Starlette as StarletteApp
from starlette.testclient import TestClient as CanonicalClient
import starlette.testclient as tc

app = App()
app_alias = app
starlette_app = StarletteApp()
client = CanonicalClient(app)
client_alias = client
reexported = Reexported(app)
qualified = tc.TestClient(app)
app_alias
starlette_app
client_alias
reexported
qualified
"#,
            &[
                ("app_alias", false, true),
                ("starlette_app", false, true),
                ("client_alias", true, false),
                ("reexported", true, false),
                ("qualified", true, false),
            ],
        );
    }

    #[test]
    fn canonical_stub_definitions() {
        let mut env = TestEnv::new();
        env.add_with_path("fastapi", "fastapi/__init__.pyi", "class FastAPI: ...");
        env.add_with_path(
            "fastapi.testclient",
            "fastapi/testclient.pyi",
            "class TestClient:\n    def __init__(self, app: object) -> None: ...",
        );
        assert_provenance(
            env,
            r#"
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
ClientAlias = TestClient
client = ClientAlias(app)
app
client
"#,
            &[("app", false, true), ("client", true, false)],
        );
    }

    #[test]
    fn typed_subclasses_and_self() {
        assert_provenance(
            framework_env(),
            r#"
from fastapi import FastAPI
from starlette.testclient import TestClient
from typing import Self

class Intermediate(TestClient): ...
class CustomClient(Intermediate):
    def identity(self) -> Self:
        return self
class CustomApp(FastAPI): ...

custom_app = CustomApp()
custom_client = CustomClient(custom_app)
custom_app
custom_client
"#,
            &[
                ("self", true, false),
                ("custom_app", false, true),
                ("custom_client", true, false),
            ],
        );
    }

    #[test]
    fn unrelated_classes_and_shadowed_names() {
        let mut env = framework_env();
        env.add_with_path("httpx", "httpx.pyi", "class Client: ...");
        env.add_with_path("requests", "requests.pyi", "class Session: ...");
        env.add_with_path("unrelated", "unrelated.pyi", "class TestClient: ...");
        assert_provenance(
            env,
            r#"
import httpx
import requests
from unrelated import TestClient as OtherTestClient
from starlette.testclient import TestClient as RealTestClient

class TestClient: ...
class FastAPI: ...
class Starlette: ...
local_client = TestClient()
other_client = OtherTestClient()
http_client = httpx.Client()
session = requests.Session()
local_app = FastAPI()
other_app = Starlette()
client_dict: dict[str, int] = {}
local_client
other_client
http_client
session
local_app
other_app
client_dict
RealTestClient

def shadowed(TestClient: dict[str, int]) -> dict[str, int]:
    return TestClient
"#,
            &[
                ("local_client", false, false),
                ("other_client", false, false),
                ("http_client", false, false),
                ("session", false, false),
                ("local_app", false, false),
                ("other_app", false, false),
                ("client_dict", false, false),
                ("RealTestClient", false, false),
                ("TestClient", false, false),
            ],
        );
    }

    #[test]
    fn ambiguous_types_are_not_provenance() {
        assert_provenance(
            framework_env(),
            r#"
from starlette.testclient import TestClient
from typing import Any, TypedDict

class Data(TypedDict):
    path: str

def probes(ambiguous: TestClient | dict[str, int], unknown: Any,
           constructor: type[TestClient], data: Data) -> None:
    ambiguous
    unknown
    constructor
    data
"#,
            &[
                ("ambiguous", false, false),
                ("unknown", false, false),
                ("constructor", false, false),
                ("data", false, false),
            ],
        );
    }

    #[test]
    fn canonical_namespace_requires_toplevel_class() {
        let mut env = TestEnv::new();
        env.add_with_path("starlette", "starlette/__init__.pyi", "");
        env.add_with_path(
            "starlette.testclient",
            "starlette/testclient.pyi",
            "class Namespace:\n    class TestClient: ...",
        );
        assert_provenance(
            env,
            r#"
from starlette.testclient import Namespace
nested_client = Namespace.TestClient()
nested_client
"#,
            &[("nested_client", false, false)],
        );
    }
}
