/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_types::tuple::Tuple;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;

/// Tracks the current position in the type argument list while matching
/// arguments to type parameters in `create_targs`.
pub struct TArgsCursor {
    targs: Vec<Type>,
    idx: usize,
    /// Set when a TypeVarTuple was filled with an unbounded tuple unpack,
    /// leaving an infinite supply for subsequent TypeVar parameters.
    unbounded_supply: bool,
}

impl TArgsCursor {
    pub fn new(targs: Vec<Type>) -> Self {
        Self {
            targs: Self::expand_unpacked_targs(targs),
            idx: 0,
            unbounded_supply: false,
        }
    }

    pub fn peek(&self) -> Option<&Type> {
        self.targs.get(self.idx)
    }

    pub fn nargs(&self) -> usize {
        self.targs.len()
    }

    /// How many args are left that must be consumed?
    pub fn nargs_unconsumed(&self, stop: usize) -> usize {
        if self.idx + 1 == stop
            && matches!(
                self.targs[self.idx],
                Type::Unpack(box Type::Tuple(Tuple::Unbounded(_)))
            )
        {
            // All we have left is an unbounded tuple that supplies 0 or more args.
            0
        } else {
            stop.saturating_sub(self.idx)
        }
    }

    pub fn consume_for_typevartuple_arg(&mut self, param_idx: usize, tparams: &TParams) -> &[Type] {
        let args_to_consume = self.num_typevartuple_args_to_consume(param_idx, tparams);
        if args_to_consume == 0
            && matches!(
                &self.targs[self.idx],
                Type::Unpack(box Type::Tuple(Tuple::Unbounded(_)))
            )
        {
            // The current arg is an unbounded tuple that is used to fill the param but not consumed.
            self.unbounded_supply = true;
            &self.targs[self.idx..self.idx + 1]
        } else {
            let old_idx = self.idx;
            self.idx += args_to_consume;
            &self.targs[old_idx..self.idx]
        }
    }

    pub fn consume_for_paramspec_value(&mut self) -> &[Type] {
        self.idx = self.targs.len();
        &self.targs
    }

    pub fn consume_for_paramspec_arg(&mut self) -> &Type {
        let arg = &self.targs[self.idx];
        self.idx += 1;
        arg
    }

    pub fn consume_for_typevar_arg(&mut self) -> &Type {
        let arg = &self.targs[self.idx];
        match arg {
            Type::Unpack(box Type::Tuple(Tuple::Unbounded(box elt))) if self.unbounded_supply => {
                elt
            }
            _ => {
                self.idx += 1;
                arg
            }
        }
    }

    /// Expand unpacked tuple arguments so they can fill multiple type parameters.
    fn expand_unpacked_targs(targs: Vec<Type>) -> Vec<Type> {
        let mut expanded = Vec::with_capacity(targs.len());
        for arg in targs {
            match arg {
                Type::Unpack(box Type::Tuple(Tuple::Concrete(elts))) => {
                    expanded.extend(elts);
                }
                Type::Unpack(box Type::Tuple(Tuple::Unpacked(box (prefix, middle, suffix)))) => {
                    expanded.extend(prefix);
                    expanded.push(Type::Unpack(Box::new(middle)));
                    expanded.extend(suffix);
                }
                arg => expanded.push(arg),
            }
        }
        expanded
    }

    fn peek_next_paramspec_param(start_idx: usize, tparams: &TParams) -> Option<usize> {
        for (i, param) in tparams.iter().enumerate().skip(start_idx) {
            if param.is_param_spec() {
                return Some(i);
            }
        }
        None
    }

    fn peek_next_paramspec_arg(&self) -> Option<usize> {
        for (i, arg) in self.targs.iter().enumerate().skip(self.idx) {
            if arg.is_kind_param_spec() {
                return Some(i);
            }
        }
        None
    }

    fn num_typevartuple_args_to_consume(&self, param_idx: usize, tparams: &TParams) -> usize {
        // We know that ParamSpec params must be matched by ParamSpec args, so chop off both params and args
        // at the next ParamSpec when computing how many args the TypeVarTuple should consume.
        let paramspec_param_idx = Self::peek_next_paramspec_param(param_idx + 1, tparams);
        let nparams_for_tvt = paramspec_param_idx.unwrap_or(tparams.len());
        let n_remaining_params = nparams_for_tvt - param_idx - 1;

        let paramspec_arg_idx = self.peek_next_paramspec_arg();
        let nargs_for_tvt = paramspec_arg_idx.unwrap_or(self.targs.len());
        let n_remaining_args = self.nargs_unconsumed(nargs_for_tvt);

        n_remaining_args.saturating_sub(n_remaining_params)
    }
}
