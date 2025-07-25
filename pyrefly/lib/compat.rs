/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// These are things that got moved. We introduce compat wrappers here to make the
// move as small as possible, then aim to slowly unwind them over time.

pub(crate) use pyrefly_config as config;
pub(crate) use pyrefly_types as types;
pub(crate) use pyrefly_types::equality;
pub(crate) type ModuleInfo = pyrefly_python::module::Module;
