import _thread
import sys
from _thread import _ExceptHookArgs, get_native_id as get_native_id
from _typeshed import ProfileFunction, TraceFunction
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from types import TracebackType
from typing import Any, Final, TypeVar, final
from typing_extensions import deprecated

_T = TypeVar("_T")

__all__ = [
    "get_ident",
    "active_count",
    "Condition",
    "current_thread",
    "enumerate",
    "main_thread",
    "TIMEOUT_MAX",
    "Event",
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Thread",
    "Barrier",
    "BrokenBarrierError",
    "Timer",
    "ThreadError",
    "ExceptHookArgs",
    "setprofile",
    "settrace",
    "local",
    "stack_size",
    "excepthook",
    "get_native_id",
]

if sys.version_info >= (3, 10):
    __all__ += ["getprofile", "gettrace"]

if sys.version_info >= (3, 12):
    __all__ += ["setprofile_all_threads", "settrace_all_threads"]

_profile_hook: ProfileFunction | None

def active_count() -> int: ...
@deprecated("Deprecated since Python 3.10. Use `active_count()` instead.")
def activeCount() -> int: ...
def current_thread() -> Thread: ...
@deprecated("Deprecated since Python 3.10. Use `current_thread()` instead.")
def currentThread() -> Thread: ...
def get_ident() -> int:
    """
    get_ident() -> integer

    Return a non-zero integer that uniquely identifies the current thread
    amongst other threads that exist simultaneously.
    This may be used to identify per-thread resources.
    Even though on some platforms threads identities may appear to be
    allocated consecutive numbers starting at 1, this behavior should not
    be relied upon, and the number should be seen purely as a magic cookie.
    A thread's identity may be reused for another thread after it exits.
    """
    ...
def enumerate() -> list[Thread]: ...
def main_thread() -> Thread: ...
def settrace(func: TraceFunction) -> None: ...
def setprofile(func: ProfileFunction | None) -> None: ...

if sys.version_info >= (3, 12):
    def setprofile_all_threads(func: ProfileFunction | None) -> None: ...
    def settrace_all_threads(func: TraceFunction) -> None: ...

if sys.version_info >= (3, 10):
    def gettrace() -> TraceFunction | None: ...
    def getprofile() -> ProfileFunction | None: ...

def stack_size(size: int = 0, /) -> int:
    """
    stack_size([size]) -> size

    Return the thread stack size used when creating new threads.  The
    optional size argument specifies the stack size (in bytes) to be used
    for subsequently created threads, and must be 0 (use platform or
    configured default) or a positive integer value of at least 32,768 (32k).
    If changing the thread stack size is unsupported, a ThreadError
    exception is raised.  If the specified size is invalid, a ValueError
    exception is raised, and the stack size is unmodified.  32k bytes
     currently the minimum supported stack size value to guarantee
    sufficient stack space for the interpreter itself.

    Note that some platforms may have particular restrictions on values for
    the stack size, such as requiring a minimum stack size larger than 32 KiB or
    requiring allocation in multiples of the system memory page size
    - platform documentation should be referred to for more information
    (4 KiB pages are common; using multiples of 4096 for the stack size is
    the suggested approach in the absence of more specific information).
    """
    ...

TIMEOUT_MAX: Final[float]

ThreadError = _thread.error
local = _thread._local

class Thread:
    name: str
    @property
    def ident(self) -> int | None:
        """
        Thread identifier of this thread or None if it has not been started.

        This is a nonzero integer. See the get_ident() function. Thread
        identifiers may be recycled when a thread exits and another thread is
        created. The identifier is available even after the thread has exited.
        """
        ...
    daemon: bool
    if sys.version_info >= (3, 14):
        def __init__(
            self,
            group: None = None,
            target: Callable[..., object] | None = None,
            name: str | None = None,
            args: Iterable[Any] = (),
            kwargs: Mapping[str, Any] | None = None,
            *,
            daemon: bool | None = None,
            context: ContextVar[Any] | None = None,
        ) -> None: ...
    else:
        def __init__(
            self,
            group: None = None,
            target: Callable[..., object] | None = None,
            name: str | None = None,
            args: Iterable[Any] = (),
            kwargs: Mapping[str, Any] | None = None,
            *,
            daemon: bool | None = None,
        ) -> None: ...

    def start(self) -> None: ...
    def run(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    @property
    def native_id(self) -> int | None:
        """
        Native integral thread ID of this thread, or None if it has not been started.

        This is a non-negative integer. See the get_native_id() function.
        This represents the Thread ID as reported by the kernel.
        """
        ...
    def is_alive(self) -> bool: ...
    @deprecated("Deprecated since Python 3.10. Read the `daemon` attribute instead.")
    def isDaemon(self) -> bool: ...
    @deprecated("Deprecated since Python 3.10. Set the `daemon` attribute instead.")
    def setDaemon(self, daemonic: bool) -> None: ...
    @deprecated("Deprecated since Python 3.10. Read the `name` attribute instead.")
    def getName(self) -> str: ...
    @deprecated("Deprecated since Python 3.10. Set the `name` attribute instead.")
    def setName(self, name: str) -> None: ...

class _DummyThread(Thread):
    def __init__(self) -> None: ...

# This is actually the function _thread.allocate_lock for <= 3.12
Lock = _thread.LockType

# Python implementation of RLock.
@final
class _RLock:
    _count: int
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    __enter__ = acquire
    def __exit__(self, t: type[BaseException] | None, v: BaseException | None, tb: TracebackType | None) -> None: ...

    if sys.version_info >= (3, 14):
        def locked(self) -> bool: ...

RLock = _thread.RLock  # Actually a function at runtime.

class Condition:
    def __init__(self, lock: Lock | _RLock | RLock | None = None) -> None: ...
    def __enter__(self) -> bool: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    if sys.version_info >= (3, 14):
        def locked(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...
    def wait_for(self, predicate: Callable[[], _T], timeout: float | None = None) -> _T: ...
    def notify(self, n: int = 1) -> None: ...
    def notify_all(self) -> None: ...
    @deprecated("Deprecated since Python 3.10. Use `notify_all()` instead.")
    def notifyAll(self) -> None: ...

class Semaphore:
    _value: int
    def __init__(self, value: int = 1) -> None: ...
    def __exit__(self, t: type[BaseException] | None, v: BaseException | None, tb: TracebackType | None) -> None: ...
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...
    def __enter__(self, blocking: bool = True, timeout: float | None = None) -> bool: ...
    def release(self, n: int = 1) -> None: ...

class BoundedSemaphore(Semaphore): ...

class Event:
    def is_set(self) -> bool: ...
    @deprecated("Deprecated since Python 3.10. Use `is_set()` instead.")
    def isSet(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool: ...

excepthook: Callable[[_ExceptHookArgs], object]
if sys.version_info >= (3, 10):
    __excepthook__: Callable[[_ExceptHookArgs], object]
ExceptHookArgs = _ExceptHookArgs

class Timer(Thread):
    args: Iterable[Any]  # undocumented
    finished: Event  # undocumented
    function: Callable[..., Any]  # undocumented
    interval: float  # undocumented
    kwargs: Mapping[str, Any]  # undocumented

    def __init__(
        self,
        interval: float,
        function: Callable[..., object],
        args: Iterable[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None: ...
    def cancel(self) -> None: ...

class Barrier:
    @property
    def parties(self) -> int:
        """Return the number of threads required to trip the barrier."""
        ...
    @property
    def n_waiting(self) -> int:
        """Return the number of threads currently waiting at the barrier."""
        ...
    @property
    def broken(self) -> bool:
        """Return True if the barrier is in a broken state."""
        ...
    def __init__(self, parties: int, action: Callable[[], None] | None = None, timeout: float | None = None) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def reset(self) -> None: ...
    def abort(self) -> None: ...

class BrokenBarrierError(RuntimeError): ...
