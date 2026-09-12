/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;

use crate::state::require::Require;
use crate::test::util::TestEnv;

#[test]
fn test_workspace_symbols_underscore_prefixed_methods() {
    let code = r#"
class Example:
    def _private_method(self) -> None:
        pass

    def public_method(self) -> None:
        pass
"#;
    let (state, _) = TestEnv::one("main", code).to_state();
    let transaction = state.transaction();
    for name in ["_private_method", "public_method"] {
        let symbols = transaction.workspace_symbols(name, None).unwrap();
        assert_eq!(symbols.len(), 1, "expected {name} in workspace symbols");
        let symbol = &symbols[0];
        assert_eq!(symbol.name, name);
        assert_eq!(symbol.kind, SymbolKind::METHOD);
        assert_eq!(symbol.container_name.as_deref(), Some("Example"));
        assert_eq!(symbol.location.module.code_at(symbol.location.range), name);
    }
}

// https://github.com/facebook/pyrefly/issues/4688
#[test]
fn test_workspace_symbols_instance_attributes() {
    let code = r#"
class Example:
    def __init__(self) -> None:
        self._private_member = 1
        self.public_member = 2
"#;
    for require in [Require::Everything, Require::Indexing] {
        let (state, _) = TestEnv::one("main", code)
            .with_run_require(require)
            .to_state();
        let transaction = state.transaction();
        for name in ["_private_member", "public_member"] {
            let symbols = transaction.workspace_symbols(name, None).unwrap();
            assert_eq!(symbols.len(), 1, "expected {name} with {require:?}");
            let symbol = &symbols[0];
            assert_eq!(symbol.name, name);
            assert_eq!(symbol.kind, SymbolKind::FIELD);
            assert_eq!(symbol.container_name.as_deref(), Some("Example"));
            assert_eq!(symbol.location.module.code_at(symbol.location.range), name);
            assert!(
                transaction
                    .search_exports_fuzzy(name, None)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

#[test]
fn test_workspace_symbols_instance_attribute_deduplication() {
    let code = r#"
class Example:
    def __init__(self) -> None:
        self._private_member = 1

    def reset(self) -> None:
        self._private_member = 2

class Other:
    def __init__(self) -> None:
        self._private_member = 3
"#;
    let (state, _) = TestEnv::one("main", code).to_state();
    let transaction = state.transaction();
    let symbols = transaction
        .workspace_symbols("_private_member", None)
        .unwrap();
    let mut containers = symbols
        .iter()
        .map(|symbol| {
            assert_eq!(symbol.name, "_private_member");
            assert_eq!(symbol.kind, SymbolKind::FIELD);
            symbol.container_name.as_deref().unwrap()
        })
        .collect::<Vec<_>>();
    containers.sort_unstable();
    assert_eq!(containers, ["Example", "Other"]);
    let symbol = symbols
        .iter()
        .find(|symbol| symbol.container_name.as_deref() == Some("Example"))
        .unwrap();
    assert_eq!(
        symbol.location.range.start().to_usize(),
        code.find("_private_member").unwrap(),
    );
}
