/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]

mod basic;
mod call_hierarchy;
mod completion;
mod configuration;
mod definition;
mod diagnostic;
mod did_change;
mod document_symbols;
mod file_watcher;
mod hover;
mod implementation;
mod inlay_hint;
mod io;
mod notebook_code_action;
mod notebook_completion;
mod notebook_definition;
mod notebook_hover;
mod notebook_inlay_hint;
mod notebook_references;
mod notebook_signature_help;
mod notebook_sync;
mod notebook_tokens;
mod object_model;
mod provide_type;
mod references;
mod rename;
mod type_definition;
mod unsaved_file;
mod util;
mod will_rename_files;
mod workspace_symbol;
