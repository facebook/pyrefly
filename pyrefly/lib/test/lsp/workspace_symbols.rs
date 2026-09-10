/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::SymbolKind;

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
    let (state, _) = TestEnv::one("main", code).to_state();
    let transaction = state.transaction();
    for name in ["_private_member", "public_member"] {
        let symbols = transaction.workspace_symbols(name, None).unwrap();
        assert_eq!(symbols.len(), 1, "expected {name} in workspace symbols");
        let symbol = &symbols[0];
        assert_eq!(symbol.name, name);
        assert_eq!(symbol.kind, SymbolKind::FIELD);
        assert_eq!(symbol.container_name.as_deref(), Some("Example"));
        assert_eq!(symbol.location.module.code_at(symbol.location.range), name);
    }
}
