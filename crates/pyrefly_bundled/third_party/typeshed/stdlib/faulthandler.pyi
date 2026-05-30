"""faulthandler module."""

import sys
from _typeshed import FileDescriptorLike

def cancel_dump_traceback_later() -> None:
    """
    cancel_dump_traceback_later():
    cancel the previous call to dump_traceback_later().
    """
    ...
def disable() -> None:
    """disable(): disable the fault handler"""
    ...

if sys.version_info >= (3, 15):
    def dump_traceback(
        file: FileDescriptorLike = sys.stderr, all_threads: bool = True, *, max_threads: int | None = None
    ) -> None: ...

else:
    def dump_traceback(file: FileDescriptorLike = sys.stderr, all_threads: bool = True) -> None:
        """dump_traceback(file=sys.stderr, all_threads=True): dump the traceback of the current thread, or of all threads if all_threads is True, into file"""
        ...

if sys.version_info >= (3, 14):
    def dump_c_stack(file: FileDescriptorLike = sys.stderr) -> None: ...

if sys.version_info >= (3, 15):
    def dump_traceback_later(
        timeout: float,
        repeat: bool = False,
        file: FileDescriptorLike = sys.stderr,
        exit: bool = False,
        *,
        max_threads: int | None = None,
    ) -> None: ...

else:
    def dump_traceback_later(
        timeout: float, repeat: bool = False, file: FileDescriptorLike = sys.stderr, exit: bool = False
    ) -> None:
        """
        dump_traceback_later(timeout, repeat=False, file=sys.stderr, exit=False):
        dump the traceback of all threads in timeout seconds,
        or each timeout seconds if repeat is True. If exit is True, call _exit(1) which is not safe.
        """
        ...

if sys.version_info >= (3, 15):
    def enable(
        file: FileDescriptorLike = sys.stderr, all_threads: bool = True, c_stack: bool = True, *, max_threads: int | None = None
    ) -> None: ...

elif sys.version_info >= (3, 14):
    def enable(file: FileDescriptorLike = sys.stderr, all_threads: bool = True, c_stack: bool = True) -> None: ...

else:
    def enable(file: FileDescriptorLike = sys.stderr, all_threads: bool = True) -> None:
        """enable(file=sys.stderr, all_threads=True): enable the fault handler"""
        ...

def is_enabled() -> bool:
    """is_enabled()->bool: check if the handler is enabled"""
    ...

if sys.platform != "win32":
    if sys.version_info >= (3, 15):
        def register(
            signum: int,
            file: FileDescriptorLike = sys.stderr,
            all_threads: bool = True,
            chain: bool = False,
            *,
            max_threads: int | None = None,
        ) -> None: ...
    else:
        def register(
            signum: int, file: FileDescriptorLike = sys.stderr, all_threads: bool = True, chain: bool = False
        ) -> None:
            """register(signum, file=sys.stderr, all_threads=True, chain=False): register a handler for the signal 'signum': dump the traceback of the current thread, or of all threads if all_threads is True, into file"""
            ...

    def unregister(signum: int, /) -> None:
        """unregister(signum): unregister the handler of the signal 'signum' registered by register()"""
        ...
