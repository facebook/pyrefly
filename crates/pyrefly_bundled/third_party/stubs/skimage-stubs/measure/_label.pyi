from .._shared.utils import deprecate_kwarg

def _label_bool(image, background=None, return_num=False, connectivity=None): ...
@deprecate_kwarg({"input": "label_image"}, deprecated_version="0.19", removed_version="1.0")
def label(
    label_image,
    background: int | None = None,
    return_num: bool = False,
    connectivity: int | None = None,
): ...
