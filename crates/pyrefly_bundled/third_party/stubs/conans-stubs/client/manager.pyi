from _typeshed import Incomplete
from conans.client.generators import write_toolchain as write_toolchain
from conans.client.graph.build_mode import BuildMode as BuildMode
from conans.client.graph.graph import RECIPE_CONSUMER as RECIPE_CONSUMER, RECIPE_VIRTUAL as RECIPE_VIRTUAL
from conans.client.graph.printer import print_graph as print_graph
from conans.client.importer import run_deploy as run_deploy, run_imports as run_imports
from conans.client.installer import BinaryInstaller as BinaryInstaller, call_system_requirements as call_system_requirements
from conans.client.manifest_manager import ManifestManager as ManifestManager
from conans.client.output import Color as Color
from conans.client.source import retrieve_exports_sources as retrieve_exports_sources
from conans.client.tools import cross_building as cross_building, get_cross_building_settings as get_cross_building_settings
from conans.errors import ConanException as ConanException
from conans.model.conan_file import ConanFile as ConanFile
from conans.model.graph_lock import GraphLockFile as GraphLockFile
from conans.model.ref import ConanFileReference as ConanFileReference
from conans.paths import CONANINFO as CONANINFO
from conans.util.files import normalize as normalize, save as save

def deps_install(app, ref_or_path, install_folder, base_folder, graph_info, remotes: Incomplete | None = ..., build_modes: Incomplete | None = ..., update: bool = ..., manifest_folder: Incomplete | None = ..., manifest_verify: bool = ..., manifest_interactive: bool = ..., generators: Incomplete | None = ..., no_imports: bool = ..., create_reference: Incomplete | None = ..., keep_build: bool = ..., recorder: Incomplete | None = ..., lockfile_node_id: Incomplete | None = ..., is_build_require: bool = ..., add_txt_generator: bool = ..., require_overrides: Incomplete | None = ..., conanfile_path: Incomplete | None = ..., test: Incomplete | None = ..., output_folder: Incomplete | None = ...): ...
