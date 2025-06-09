/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Used to guard computation that could potentially loop forever.

pub struct Gas(isize);

impl Gas {
    /// Create a new `Gas` with the given number of units of gas.
    /// You will be able to call `decrement` that many times successfully.
    pub const fn new(available: isize) -> Self {
        Self(available)
    }

    /// Use up a unit of gas. Returns true if you should stop.
    pub fn stop(&mut self) -> bool {
        self.0 -= 1;
        self.0 < 0
    }

    /// Restore a unit of gas.
    pub fn restore(&mut self) {
        self.0 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_precise() {
        let mut g = Gas::new(3);
        assert!(!g.stop());
        assert!(!g.stop());
        assert!(!g.stop());
        assert!(g.stop());
        assert!(g.stop());
        assert!(g.stop());
    }

    #[test]
    fn test_gas_restore() {
        let mut g = Gas::new(2);
        assert!(!g.stop());
        g.restore();
        assert!(!g.stop());
        assert!(!g.stop());
        assert!(g.stop());
    }
}
