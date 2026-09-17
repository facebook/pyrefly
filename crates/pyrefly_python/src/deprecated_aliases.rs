/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::sys_info::PythonVersion;

/// Whether `typing.<name>` is a deprecated alias for a builtin or `collections.abc` type
/// on the given Python version.
///
/// These aliases are not marked with `@deprecated` in typeshed, but the `typing`
/// documentation steers users away from them:
/// - The generic aliases (`typing.List`, `typing.Dict`, `typing.Iterable`, ...) are
///   superseded by their builtin / `collections.abc` counterparts (`list`, `dict`,
///   `collections.abc.Iterable`, ...) once PEP 585 made those subscriptable in 3.9.
/// - `typing.Optional` / `typing.Union` are superseded by the `X | None` / `X | Y`
///   syntax from PEP 604 in 3.10.
///
/// We use this to prefer the non-deprecated spelling in auto-imports and quick fixes.
pub fn is_deprecated_stdlib_alias(
    python_version: PythonVersion,
    module_name: &str,
    name: &str,
) -> bool {
    if module_name != "typing" {
        return false;
    }
    if python_version.at_least(3, 10) && matches!(name, "Optional" | "Union") {
        return true;
    }
    python_version.at_least(3, 9)
        && matches!(
            name,
            "AbstractSet"
                | "AsyncContextManager"
                | "AsyncGenerator"
                | "AsyncIterable"
                | "AsyncIterator"
                | "Awaitable"
                | "ByteString"
                | "Callable"
                | "ChainMap"
                | "Collection"
                | "Container"
                | "ContextManager"
                | "Coroutine"
                | "Counter"
                | "DefaultDict"
                | "Deque"
                | "Dict"
                | "FrozenSet"
                | "Generator"
                | "ItemsView"
                | "Iterable"
                | "Iterator"
                | "KeysView"
                | "List"
                | "Mapping"
                | "MappingView"
                | "Match"
                | "MutableMapping"
                | "MutableSequence"
                | "MutableSet"
                | "OrderedDict"
                | "Pattern"
                | "Reversible"
                | "Sequence"
                | "Set"
                | "Tuple"
                | "Type"
                | "ValuesView"
        )
}
