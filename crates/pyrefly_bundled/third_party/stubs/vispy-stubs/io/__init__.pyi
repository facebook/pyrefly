# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from os import path as _op

from .datasets import (
    load_crate as load_crate,
    load_data_file as load_data_file,
    load_iris as load_iris,
    load_spatial_filters as load_spatial_filters,
)
from .image import (
    _check_img_lib as _check_img_lib,
    _make_png as _make_png,
    imread as imread,
    imsave as imsave,
    read_png as read_png,
    write_png as write_png,
)
from .mesh import read_mesh as read_mesh, write_mesh as write_mesh

_data_dir = ...

__all__ = [
    "imread",
    "imsave",
    "load_iris",
    "load_crate",
    "load_spatial_filters",
    "load_data_file",
    "read_mesh",
    "read_png",
    "write_mesh",
    "write_png",
]
