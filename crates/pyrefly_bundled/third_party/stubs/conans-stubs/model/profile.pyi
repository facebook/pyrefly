from _typeshed import Incomplete
from conans.client import settings_preprocessor as settings_preprocessor
from conans.errors import ConanException as ConanException
from conans.model.conf import ConfDefinition as ConfDefinition
from conans.model.env_info import EnvValues as EnvValues
from conans.model.options import OptionsValues as OptionsValues
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.model.values import Values as Values

class Profile:
    settings: Incomplete
    package_settings: Incomplete
    env_values: Incomplete
    options: Incomplete
    build_requires: Incomplete
    conf: Incomplete
    buildenv: Incomplete
    runenv: Incomplete
    processed_settings: Incomplete
    _user_options: Incomplete
    _package_settings_values: Incomplete
    dev_reference: Incomplete
    def __init__(self) -> None: ...
    @property
    def user_options(self): ...
    @property
    def package_settings_values(self): ...
    def process_settings(self, cache, preprocess: bool = ...) -> None: ...
    def dumps(self): ...
    def compose_profile(self, other) -> None: ...
    def update_settings(self, new_settings) -> None: ...
    def update_package_settings(self, package_settings) -> None: ...
