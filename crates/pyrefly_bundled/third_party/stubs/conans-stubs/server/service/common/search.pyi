from _typeshed import Incomplete
from conans import load as load
from conans.errors import ForbiddenException as ForbiddenException, NotFoundException as NotFoundException, RecipeNotFoundException as RecipeNotFoundException
from conans.model.info import ConanInfo as ConanInfo
from conans.model.ref import ConanFileReference as ConanFileReference, PackageReference as PackageReference
from conans.paths import CONANINFO as CONANINFO
from conans.search.search import _partial_match as _partial_match, filter_packages as filter_packages
from conans.util.files import list_folder_subdirs as list_folder_subdirs
from conans.util.log import logger as logger

def _get_local_infos_min(server_store, ref, look_in_all_rrevs): ...
def search_packages(server_store, ref, query, look_in_all_rrevs): ...

class SearchService:
    _authorizer: Incomplete
    _server_store: Incomplete
    _auth_user: Incomplete
    def __init__(self, authorizer, server_store, auth_user) -> None: ...
    def search_packages(self, reference, query, look_in_all_rrevs: bool = ...): ...
    def _search_recipes(self, pattern: Incomplete | None = ..., ignorecase: bool = ...): ...
    def search(self, pattern: Incomplete | None = ..., ignorecase: bool = ...): ...
