from _typeshed import Incomplete
from conans.errors import ConanException as ConanException

travis: str
linux_config: str
linux_config_gcc: Incomplete
linux_config_clang: Incomplete
osx_config: str
build_py: str
travis_install: str
travis_run: str
appveyor: str
gitlab: str
gitlab_config_gcc: str
gitlab_config_clang: str
circleci: str
circleci_config_gcc: str
circleci_config_clang: str
circleci_config_osx: str
circleci_install: str
circleci_run: str
circleci_workflow: str
circleci_job: str

def get_build_py(name, shared): ...
def get_travis(name, version, user, channel, linux_gcc_versions, linux_clang_versions, osx_clang_versions, upload_url): ...
def get_appveyor(name, version, user, channel, visual_versions, upload_url): ...
def get_gitlab(name, version, user, channel, linux_gcc_versions, linux_clang_versions, upload_url): ...
def get_circleci(name, version, user, channel, linux_gcc_versions, linux_clang_versions, osx_clang_versions, upload_url): ...
def ci_get_files(name, version, user, channel, visual_versions, linux_gcc_versions, linux_clang_versions, osx_clang_versions, shared, upload_url, gitlab_gcc_versions, gitlab_clang_versions, circleci_gcc_versions, circleci_clang_versions, circleci_osx_versions): ...
