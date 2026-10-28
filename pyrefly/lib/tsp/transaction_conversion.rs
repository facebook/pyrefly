/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Convert analyzed types without depending on the native LSP server.

use dupe::Dupe;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use ruff_python_ast::name::Name;

use crate::lsp::module_helpers::to_real_path;
use crate::state::state::Transaction;
use crate::tsp::type_conversion::StdlibClasses;
use crate::tsp::type_conversion::convert_type_with_resolvers;

/// Convert `ty` using the transaction and import context that produced it.
///
/// The caller must compute the type first so this transaction's stdlib is warm.
/// Reusing the source handle's `SysInfo` keeps cross-module declarations in the
/// same analysis context, even when the transaction holds multiple platforms.
pub fn convert_type_in_transaction(
    transaction: &Transaction,
    source_handle: &Handle,
    ty: &pyrefly_types::types::Type,
) -> tsp_types::Type {
    let resolve_func_range = |func_id: &pyrefly_types::function::FuncDefId| {
        let handle = Handle::new(
            func_id.qname.module_name(),
            func_id.qname.module_path().dupe(),
            source_handle.sys_info().dupe(),
        );
        transaction
            .get_answers(&handle)?
            .bindings()
            .function_def_range(func_id.def_index)
    };
    let resolve_module_path = |module: &pyrefly_types::module::ModuleType| {
        let module_name = ModuleName::from_str(&module.to_string());
        let finding = transaction
            .import_handle(source_handle, module_name, None)
            .finding()?;
        let path = to_real_path(finding.path())?;
        Some(path.canonicalize().unwrap_or(path))
    };
    let resolve_export = |module_name: ModuleName, name: &Name| {
        resolve_export_location(transaction, source_handle, module_name, name)
    };
    let stdlib = transaction.get_stdlib(source_handle);
    convert_type_with_resolvers(
        ty,
        Some(&resolve_func_range),
        Some(&resolve_module_path),
        Some(&resolve_export),
        StdlibClasses {
            none_type: stdlib.none_type(),
            bool_type: stdlib.bool(),
            int_type: stdlib.int(),
        },
    )
}

/// Follow re-exports within the source handle's already-computed import context.
pub(crate) fn resolve_export_location(
    transaction: &Transaction,
    source_handle: &Handle,
    module_name: ModuleName,
    name: &Name,
) -> Option<(ModulePath, lsp_types::Range)> {
    let target_handle = transaction
        .import_handle(source_handle, module_name, None)
        .finding()?;
    let (module, range) = transaction.lookup_export_location(&target_handle, name)?;
    Some((module.path().dupe(), module.to_lsp_range(range)))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use lsp_types::Uri;
    use pyrefly_python::sys_info::PythonPlatform;
    use pyrefly_python::sys_info::PythonVersion;
    use pyrefly_python::sys_info::SysInfo;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use tempfile::TempDir;
    use tsp_types::TypeKind;

    use super::*;
    use crate::module::finder::DirEntryCache;
    use crate::module::finder::FindImportOptions;
    use crate::module::finder::find_import;
    use crate::state::require::Require;
    use crate::state::state::State;
    use crate::test::util::TestEnv;

    #[test]
    fn converted_reexports_keep_declarations_and_source_sysinfo() {
        let mut env = TestEnv::new();
        let directory = TempDir::new().unwrap();
        let library =
            "class Widget:\n    pass\n\ndef identity(value: Widget) -> Widget:\n    return value\n";
        let source = "from facade import Widget, identity\nitem = Widget()\nchosen = identity\n";
        env.add_with_path(
            "library",
            directory.path().join("library.py").to_str().unwrap(),
            library,
        );
        env.add_with_path(
            "facade",
            directory.path().join("facade.py").to_str().unwrap(),
            "from library import Widget, identity\n",
        );
        env.add_with_path(
            "main",
            directory.path().join("main.py").to_str().unwrap(),
            source,
        );
        let config = env.config();
        let handle = |module: &str, sys_info: &SysInfo| {
            let name = ModuleName::from_str(module);
            let path = find_import(&config, name, FindImportOptions::new(&DirEntryCache::new()))
                .finding()
                .unwrap();
            Handle::new(name, path, sys_info.dupe())
        };
        let platforms = [
            SysInfo::new(PythonVersion::default(), PythonPlatform::windows()),
            SysInfo::new(PythonVersion::default(), PythonPlatform::mac()),
        ];
        let handles = platforms.map(|sys_info| handle("main", &sys_info));
        let state = State::new(env.config_finder(), TEST_THREAD_COUNT);
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.set_memory(env.get_memory());
        transaction.run(&handles, Require::Everything, None);
        assert!(
            transaction
                .get_errors(handles.iter())
                .collect_display_errors()
                .is_empty()
        );
        for source_handle in &handles {
            let library_handle = handle("library", source_handle.sys_info());
            let expected_uri = Uri::from_file_path(&to_real_path(library_handle.path()).unwrap())
                .unwrap()
                .to_string();
            for (variable, name, kind, line, character) in [
                ("item", "Widget", TypeKind::Class, 0, 6),
                ("chosen", "identity", TypeKind::Function, 3, 4),
            ] {
                let offset = TextSize::new(source.find(variable).unwrap() as u32);
                let range = TextRange::at(offset, TextSize::new(variable.len() as u32));
                let ty = transaction
                    .get_computed_type_at_range(source_handle, range)
                    .unwrap();
                let converted = convert_type_in_transaction(&transaction, source_handle, &ty);
                let wire = serde_json::to_value(converted).unwrap();
                assert_eq!(wire["kind"], kind as u8);
                assert_eq!(wire["declaration"]["kind"], 0);
                assert_eq!(wire["declaration"]["name"], name);
                assert_eq!(wire["declaration"]["node"]["uri"], expected_uri);
                assert_eq!(
                    wire["declaration"]["node"]["range"],
                    serde_json::json!({
                        "start": {"line": line, "character": character},
                        "end": {"line": line, "character": character + name.len()}
                    })
                );
            }
        }
    }
}
