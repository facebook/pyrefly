/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Metadata collected during the binding phase that can be queried
//! without going through the solve/calculation code paths.
//!
//! This module stores per-class metadata (starting with field information)
//! in a `Vec` indexed by `ClassDefIndex`, enabling efficient lookups.

use pyrefly_types::class::ClassDefIndex;
use pyrefly_types::class::ClassFields;

/// Metadata for a single class definition, populated during binding.
#[derive(Debug, Clone, Default)]
pub struct ClassMetadata {
    /// The fields are all the names declared on the class that we were able to detect
    /// from an AST traversal, which includes:
    /// - any name defined in the class body (e.g. by assignment or a def statement)
    /// - attributes annotated in the class body (but not necessarily defined)
    /// - anything assigned to something we think is a `self` or `cls` argument
    ///
    /// The last case may include names that are actually declared in a parent class,
    /// because at binding time we cannot know that so we have to treat assignment
    /// as potentially defining a field that would not otherwise exist.
    pub fields: ClassFields,
}

/// Metadata collected during the binding phase for all classes in a module.
///
/// Stored in an `Arc` so it can be shared between `Bindings`/`Answers` and
/// `Solutions` without copying. During access, callers hold a `Guard` or
/// borrow through the `Arc` rather than cloning it, avoiding contended
/// atomic reference count operations.
#[derive(Debug, Clone)]
pub struct BindingsMetadata {
    classes: Vec<ClassMetadata>,
}

impl BindingsMetadata {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
        }
    }

    /// Allocate a new `ClassDefIndex` by pushing a default `ClassMetadata`.
    pub fn push_class(&mut self) -> ClassDefIndex {
        let idx = ClassDefIndex(self.classes.len() as u32);
        self.classes.push(ClassMetadata::default());
        idx
    }

    pub fn get_class(&self, idx: ClassDefIndex) -> &ClassMetadata {
        &self.classes[idx.0 as usize]
    }

    /// Bounds-checked version for cross-module lookups where the
    /// `ClassDefIndex` may be stale after an incremental rebuild.
    pub fn get_class_checked(&self, idx: ClassDefIndex) -> Option<&ClassMetadata> {
        self.classes.get(idx.0 as usize)
    }

    pub fn get_class_mut(&mut self, idx: ClassDefIndex) -> &mut ClassMetadata {
        &mut self.classes[idx.0 as usize]
    }
}
