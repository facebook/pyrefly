from _typeshed import Incomplete
from conans.model import Generator as Generator
from conans.model.manifest import FileTreeManifest as FileTreeManifest
from conans.paths import BUILD_INFO_DEPLOY as BUILD_INFO_DEPLOY
from conans.util.dates import timestamp_now as timestamp_now
from conans.util.files import md5sum as md5sum, mkdir as mkdir

FILTERED_FILES: Incomplete

class DeployGenerator(Generator):
    def deploy_manifest_content(self, copied_files): ...
    @property
    def filename(self): ...
    @property
    def content(self): ...
