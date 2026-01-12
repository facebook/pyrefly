from conans.client.tools.env import *
from conans.client.tools.pkg_config import *
from conans.client.tools.scm import *
from conans.client.tools.settings import *
from conans.client.tools.intel import *
from conans.client.tools.apple import *
from conans.client.tools.android import *
from _typeshed import Incomplete
from collections.abc import Generator
from conans.client.output import ConanOutput as ConanOutput
from conans.client.tools import files as tools_files, net as tools_net, oss as tools_oss, system_pm as tools_system_pm, win as tools_win
from conans.client.tools.version import Version as Version
from conans.util.env_reader import get_env as get_env
from conans.util.files import _generic_algorithm_sum as _generic_algorithm_sum, load as load, md5 as md5, md5sum as md5sum, mkdir as mkdir, relative_dirs as relative_dirs, rmdir as rmdir, save_append as save_append, sha1sum as sha1sum, sha256sum as sha256sum, to_file_bytes as to_file_bytes, touch as touch
from conans.util.log import logger as logger

_global_output: Incomplete
_global_requester: Incomplete
_global_config: Incomplete

def set_global_instances(the_output, the_requester, config) -> None: ...
def get_global_instances(): ...
def save(path, content, append: bool = ...) -> None: ...
ftp_download = tools_net.ftp_download

def download(*args, **kwargs): ...
def get(*args, **kwargs): ...
chdir = tools_files.chdir
human_size = tools_files.human_size
untargz = tools_files.untargz
check_with_algorithm_sum = tools_files.check_with_algorithm_sum
check_sha1 = tools_files.check_sha1
check_md5 = tools_files.check_md5
check_sha256 = tools_files.check_sha256
patch = tools_files.patch
replace_prefix_in_pc_file = tools_files.replace_prefix_in_pc_file
collect_libs = tools_files.collect_libs
which = tools_files.which
unix2dos = tools_files.unix2dos
dos2unix = tools_files.dos2unix
rename = tools_files.rename
fix_symlinks = tools_files.fix_symlinks
remove_files_by_mask = tools_files.remove_files_by_mask

def unzip(*args, **kwargs): ...
def replace_in_file(*args, **kwargs): ...
def replace_path_in_file(*args, **kwargs): ...
args_to_string = tools_oss.args_to_string
detected_architecture = tools_oss.detected_architecture
detected_os = tools_oss.detected_os
OSInfo = tools_oss.OSInfo
cross_building = tools_oss.cross_building
get_cross_building_settings = tools_oss.get_cross_building_settings
get_gnu_triplet = tools_oss.get_gnu_triplet

def cpu_count(*args, **kwargs): ...

class SystemPackageTool(tools_system_pm.SystemPackageTool):
    def __init__(self, *args, **kwargs) -> None: ...

class NullTool(tools_system_pm.NullTool):
    def __init__(self, *args, **kwargs) -> None: ...

class AptTool(tools_system_pm.AptTool):
    def __init__(self, *args, **kwargs) -> None: ...

class DnfTool(tools_system_pm.DnfTool):
    def __init__(self, *args, **kwargs) -> None: ...

class YumTool(tools_system_pm.YumTool):
    def __init__(self, *args, **kwargs) -> None: ...

class BrewTool(tools_system_pm.BrewTool):
    def __init__(self, *args, **kwargs) -> None: ...

class PkgTool(tools_system_pm.PkgTool):
    def __init__(self, *args, **kwargs) -> None: ...

class ChocolateyTool(tools_system_pm.ChocolateyTool):
    def __init__(self, *args, **kwargs) -> None: ...

class PkgUtilTool(tools_system_pm.PkgUtilTool):
    def __init__(self, *args, **kwargs) -> None: ...

class PacManTool(tools_system_pm.PacManTool):
    def __init__(self, *args, **kwargs) -> None: ...

class ZypperTool(tools_system_pm.ZypperTool):
    def __init__(self, *args, **kwargs) -> None: ...
vs_installation_path = tools_win.vs_installation_path
vswhere = tools_win.vswhere
vs_comntools = tools_win.vs_comntools
find_windows_10_sdk = tools_win.find_windows_10_sdk
escape_windows_cmd = tools_win.escape_windows_cmd
get_cased_path = tools_win.get_cased_path
MSYS2: Incomplete
MSYS: Incomplete
CYGWIN: Incomplete
WSL: Incomplete
SFU: Incomplete
unix_path = tools_win.unix_path
run_in_windows_bash = tools_win.run_in_windows_bash
msvs_toolset = tools_win.msvs_toolset

def vcvars(*args, **kwargs) -> Generator[None, None, None]: ...
def msvc_build_command(*args, **kwargs): ...
def build_sln_command(*args, **kwargs): ...
def vcvars_command(*args, **kwargs): ...
def vcvars_dict(*args, **kwargs): ...
def latest_vs_version_installed(*args, **kwargs): ...

os_info: Incomplete
