/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_view,
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

django_testcase!(
    test_list_view,
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

django_testcase!(
    test_user_passes_test_decorator,
    r#"
from django.contrib.auth.decorators import user_passes_test
from django.http import HttpRequest, HttpResponse
from django.urls import reverse, reverse_lazy

reversed_url = reverse("url")

@user_passes_test(lambda user: user.is_active, login_url=reversed_url)
def my_view1(request: HttpRequest) -> HttpResponse: ...
"#,
);
