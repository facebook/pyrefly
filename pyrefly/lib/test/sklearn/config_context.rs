/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

// No inline `sklearn` module is added, so `import sklearn` resolves against the real
// bundled third-party stubs and exercises the actual sklearn-stubs rather than a
// hand-written copy. The `missing-source-for-stubs` error is ignored, because there is
// no sklearn installed in the test env.
testcase!(
    test_sklearn_config_context_is_context_manager,
    TestEnv::new(),
    r#"
import sklearn # pyrefly: ignore[missing-source-for-stubs]

with sklearn.config_context(transform_output="pandas"):
    pass
"#,
);
