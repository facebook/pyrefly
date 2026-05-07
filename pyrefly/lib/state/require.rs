/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use dupe::Dupe;

#[derive(Debug, Clone, Dupe, Copy)]
pub struct RequireLevels {
    pub specified: Require,
    pub default: Require,
}

const EXPORTS: u8 = 0;
const ERRORS: u8 = 1;
const INDEXING: u8 = 2;
const EVERYTHING: u8 = 3;

/// How much information do we require about a module?
#[derive(Debug, Clone, Dupe, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Require {
    /// We require nothing about the module.
    /// It's only purpose is to provide information about dependencies, namely Exports.
    Exports = EXPORTS as isize,
    /// We want to know what errors this module produces.
    Errors = ERRORS as isize,
    /// We want to retain enough information about a file (e.g. references),
    /// so that IDE features that require an index can work.
    Indexing = INDEXING as isize,
    /// We want to retain all information about this module in memory,
    /// including the AST and bindings/answers.
    Everything = EVERYTHING as isize,
}

impl Require {
    /// Encode as a u8 for atomic storage.
    fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a u8 back to a Require. Panics on invalid values.
    fn from_u8(v: u8) -> Self {
        match v {
            EXPORTS => Require::Exports,
            ERRORS => Require::Errors,
            INDEXING => Require::Indexing,
            EVERYTHING => Require::Everything,
            _ => panic!("Invalid Require encoding: {v}"),
        }
    }

    pub fn compute_errors(self) -> bool {
        self >= Require::Errors
    }

    pub fn keep_index(self) -> bool {
        self >= Require::Indexing
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
}

/// Atomic version of `Require` for lock-free access across threads.
/// Uses `AtomicU8` internally with the same encoding as `Require::to_u8`.
#[derive(Debug)]
pub struct AtomicRequire(AtomicU8);

impl AtomicRequire {
    pub fn new(require: Require) -> Self {
        Self(AtomicU8::new(require.to_u8()))
    }

    pub fn load(&self) -> Require {
        Require::from_u8(self.0.load(Ordering::Acquire))
    }

    /// Try to increase the require level. Returns true if the level was
    /// actually increased (the new level is strictly greater than the old).
    pub fn increase(&self, require: Require) -> bool {
        let new_val = require.to_u8();
        let mut old_val = self.0.load(Ordering::Acquire);
        loop {
            if new_val <= old_val {
                return false;
            }
            match self.0.compare_exchange_weak(
                old_val,
                new_val,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(x) => old_val = x,
            }
        }
    }
}
