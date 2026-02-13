# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from ._runners import test as test
from ._testing import (
    IS_CI as IS_CI,
    IS_TRAVIS_CI as IS_TRAVIS_CI,
    SkipTest as SkipTest,
    TestingCanvas as TestingCanvas,
    assert_equal as assert_equal,
    assert_in as assert_in,
    assert_is as assert_is,
    assert_not_equal as assert_not_equal,
    assert_not_in as assert_not_in,
    assert_raises as assert_raises,
    assert_true as assert_true,
    has_pyopengl as has_pyopengl,
    raises as raises,
    requires_application as requires_application,
    requires_img_lib as requires_img_lib,
    requires_ipython as requires_ipython,
    requires_numpydoc as requires_numpydoc,
    requires_pyopengl as requires_pyopengl,
    requires_scipy as requires_scipy,
    requires_ssl as requires_ssl,
    run_tests_if_main as run_tests_if_main,
    save_testing_image as save_testing_image,
)
