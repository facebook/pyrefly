/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]

mod basic;
mod completion;
mod configuration;
mod definition;
mod diagnostic;
mod did_change;
mod file_watcher;
mod hover;
mod inlay_hint;
mod io;
mod notebook_hover;
mod notebook_sync;
mod object_model;
mod provide_type;
mod references;
mod rename;
mod type_definition;
mod util;
mod will_rename_files;
mod workspace_symbol;
