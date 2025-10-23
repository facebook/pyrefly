/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use dupe::Dupe;

/// The maximum number of times we'll follow imports when indexing
const INDEXING_DEPTH: usize = 5;

/// How much information do we require about a module?
#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Require {
    /// We require nothing about the module.
    /// It's only purpose is to provide information about dependencies, namely Exports.
    Exports,
    /// We want to know what errors this module produces.
    Errors,
    /// We want to retain enough information about a file (e.g. references),
    /// so that IDE features that require an index can work. The `usize` is the remaining indexing
    /// depth - i.e., how many more times we'll follow imports.
    Indexing(usize),
    /// We want to retain all information about this module in memory,
    /// including the AST and bindings/answers.
    Everything,
}

impl Require {
    pub fn compute_errors(self) -> bool {
        self >= Require::Errors
    }

    pub fn keep_index(self) -> bool {
        self >= Require::Indexing(0)
    }

    pub fn keep_answers_trace(self) -> bool {
        self >= Require::Everything
    }

    pub fn keep_ast(self) -> bool {
        self >= Require::Everything
    }

    pub fn keep_bindings(self) -> bool {
        self >= Require::Everything
    }

    pub fn keep_answers(self) -> bool {
        self >= Require::Everything
    }

    pub fn indexing() -> Self {
        Self::Indexing(INDEXING_DEPTH)
    }
}
