from _typeshed import Incomplete
from conans.client.tools import environment_append as environment_append
from conans.errors import ConanException as ConanException
from conans.util.files import decode_text as decode_text
from conans.util.runners import pyinstaller_bundle_env_cleaned as pyinstaller_bundle_env_cleaned

class _UnbufferedWrite:
    _stream: Incomplete
    def __init__(self, stream) -> None: ...
    def write(self, *args, **kwargs) -> None: ...

class ConanRunner:
    _print_commands_to_output: Incomplete
    _generate_run_log_file: Incomplete
    _log_run_to_output: Incomplete
    _output: Incomplete
    def __init__(self, print_commands_to_output: bool = ..., generate_run_log_file: bool = ..., log_run_to_output: bool = ..., output: Incomplete | None = ...) -> None: ...
    def __call__(self, command, output: bool = ..., log_filepath: Incomplete | None = ..., cwd: Incomplete | None = ..., subprocess: bool = ...): ...
    def _pipe_os_call(self, command, stream_output, log_handler, cwd, user_output): ...
    @staticmethod
    def _simple_os_call(command, cwd): ...
