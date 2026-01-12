from conans.client.output import ScopedOutput as ScopedOutput
from conans.client.source import retrieve_exports_sources as retrieve_exports_sources
from conans.errors import NotFoundException as NotFoundException, RecipeNotFoundException as RecipeNotFoundException
from conans.model.ref import ConanFileReference as ConanFileReference, PackageReference as PackageReference

def download(app, ref, package_ids, remote, recipe, recorder, remotes) -> None: ...
def _download_binaries(conanfile, ref, package_ids, cache, remote_manager, remote, output, recorder, parallel) -> None: ...
