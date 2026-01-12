from _typeshed import Incomplete
from conans.client.graph.graph import RECIPE_DOWNLOADED as RECIPE_DOWNLOADED, RECIPE_EDITABLE as RECIPE_EDITABLE, RECIPE_INCACHE as RECIPE_INCACHE, RECIPE_NEWER as RECIPE_NEWER, RECIPE_NOT_IN_REMOTE as RECIPE_NOT_IN_REMOTE, RECIPE_NO_REMOTE as RECIPE_NO_REMOTE, RECIPE_UPDATEABLE as RECIPE_UPDATEABLE, RECIPE_UPDATED as RECIPE_UPDATED
from conans.client.output import ScopedOutput as ScopedOutput
from conans.client.recorder.action_recorder import INSTALL_ERROR_MISSING as INSTALL_ERROR_MISSING, INSTALL_ERROR_NETWORK as INSTALL_ERROR_NETWORK
from conans.client.remover import DiskRemover as DiskRemover
from conans.errors import ConanException as ConanException, NotFoundException as NotFoundException, RecipeNotFoundException as RecipeNotFoundException
from conans.paths.package_layouts.package_editable_layout import PackageEditableLayout as PackageEditableLayout
from conans.util.tracer import log_recipe_got_from_local_cache as log_recipe_got_from_local_cache

DEPRECATED_CONAN_CENTER_BINTRAY_URL: str

class ConanProxy:
    _cache: Incomplete
    _out: Incomplete
    _remote_manager: Incomplete
    def __init__(self, cache, output, remote_manager) -> None: ...
    def get_recipe(self, ref, check_updates, update, remotes, recorder): ...
    def _get_recipe(self, layout, ref, check_updates, update, remotes, recorder): ...
    def _download_recipe(self, layout, ref, output, remotes, remote, recorder): ...
