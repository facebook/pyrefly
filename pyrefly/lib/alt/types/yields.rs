/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::VisitMut;
use pyrefly_types::heap::TypeHeap;

use crate::types::types::Type;

#[derive(Clone, Debug, VisitMut, TypeEq, Eq, PartialEq)]
pub struct YieldResult {
    pub yield_ty: Type,
    pub send_ty: Type,
}

#[derive(Clone, Debug, VisitMut, TypeEq, Eq, PartialEq)]
pub struct YieldFromResult {
    pub yield_ty: Type,
    pub send_ty: Type,
    pub return_ty: Type,
}

impl YieldResult {
    pub fn recursive(heap: &TypeHeap) -> Self {
        YieldResult {
            yield_ty: heap.mk_any_implicit(),
            send_ty: heap.mk_any_implicit(),
        }
    }

    pub fn any_error(heap: &TypeHeap) -> Self {
        YieldResult {
            yield_ty: heap.mk_any_error(),
            send_ty: heap.mk_any_error(),
        }
    }
}

impl YieldFromResult {
    pub fn recursive(heap: &TypeHeap) -> Self {
        YieldFromResult {
            yield_ty: heap.mk_any_implicit(),
            send_ty: heap.mk_any_implicit(),
            return_ty: heap.mk_any_implicit(),
        }
    }

    pub fn any_error(heap: &TypeHeap) -> Self {
        YieldFromResult {
            yield_ty: heap.mk_any_error(),
            send_ty: heap.mk_any_error(),
            return_ty: heap.mk_any_error(),
        }
    }

    pub fn from_iterable(heap: &TypeHeap, yield_ty: Type) -> Self {
        YieldFromResult {
            yield_ty,
            send_ty: heap.mk_any_implicit(),
            return_ty: heap.mk_any_implicit(),
        }
    }

    pub fn from_generator(generator: (Type, Type, Type)) -> Self {
        let (yield_ty, send_ty, return_ty) = generator;
        YieldFromResult {
            yield_ty,
            send_ty,
            return_ty,
        }
    }
}

impl Display for YieldResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "yield: {}, send: {}", self.yield_ty, self.send_ty)
    }
}

impl Display for YieldFromResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "yield: {}, send: {}, return: {}",
            self.yield_ty, self.send_ty, self.return_ty
        )
    }
}
