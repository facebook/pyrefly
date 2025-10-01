/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn django_env() -> TestEnv {
    let path = std::env::var("DJANGO_TEST_PATH").expect("DJANGO_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

testcase!(
    test_view,
    django_env(),
    r#"
from django.db import models
from django.views.generic.detail import SingleObjectMixin 
from typing_extensions import assert_type

class MyModel(models.Model): ...

class MyDetailView(SingleObjectMixin[MyModel]): ...

detail_view = MyDetailView()
assert_type(detail_view.model, type[MyModel]) 
assert_type(detail_view.queryset, models.QuerySet[MyModel, MyModel] | None) 
assert_type(detail_view.get_context_object_name(MyModel()), str) 
assert_type(detail_view.get_context_object_name(1), str | None) 
"#,
);

testcase!(
    test_list_view,
    django_env(),
    r#"
from django.db import models
from django.views.generic.list import ListView 
from typing_extensions import assert_type

class MyModel(models.Model): ...

class MyListView(ListView[MyModel]): ...

list_view = MyListView()
assert_type(list_view.model, type[MyModel] | None) 
assert_type(list_view.queryset, models.QuerySet[MyModel, MyModel] | None) 
assert_type(list_view.get_context_object_name(models.QuerySet[MyModel]()), str) 
assert_type(list_view.get_context_object_name(MyModel()), str | None) 
assert_type(list_view.get_context_object_name(1), str | None)
"#,
);

testcase!(
    bug = "Example should typecheck.",
    test_user_passes_test_decorator,
    django_env(),
    r#"
from django.contrib.auth.decorators import user_passes_test 
from django.http import HttpRequest, HttpResponse 
from django.urls import reverse, reverse_lazy

reversed_url = reverse("url")

@user_passes_test(lambda user: user.is_active, login_url=reversed_url) # E: Argument `(user: AbstractBaseUser | AnonymousUser) -> BooleanField[Combinable | bool, bool] | bool` is not assignable to parameter `test_func` with type `(AbstractBaseUser | AnonymousUser) -> bool` in function `django.contrib.auth.decorators.user_passes_test`
def my_view1(request: HttpRequest) -> HttpResponse: ...
"#,
);
