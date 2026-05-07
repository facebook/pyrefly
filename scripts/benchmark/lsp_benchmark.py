#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark `textDocument/definition` between LSP servers on a codebase.

This script is intentionally self-contained (std-lib only) so it can run in
restricted environments.

Two modes of operation:

1. **Single-repo mode** (--root): benchmark an already-cloned repo.
2. **Multi-package mode** (--install-envs): auto-clone packages from
   install_envs.json and benchmark each one, producing aggregate results.

What it does
------------
1) Pick a random Python file under the given repo root.
2) Pick a random *identifier token* position in that file.
3) Start each LSP server (pyrefly, ty, pyright, zuban) over stdio.
4) Initialize, open the document, and request `textDocument/definition`.
5) Measure latency and check whether the returned location looks valid.

Notes
-----
- You must provide the actual server commands (or use --install-envs mode
  which auto-detects installed checkers).
- The benchmark is about "Go to definition" wiring, not type-checking.
- LSP servers can return either `Location` or `Location[]` or
  `LocationLink[]`. This script supports all of them.

Usage:
    # Single-repo mode
    python3 lsp_benchmark.py --root <REPO> --pyrefly-cmd "pyrefly lsp"

    # Multi-package mode (auto-clones from install_envs.json)
    python3 lsp_benchmark.py --install-envs install_envs.json -r 10

Examples:
    # Benchmark pyrefly and ty on a repo
    python3 lsp_benchmark.py --root ~/projects/myrepo \\
        --pyrefly-cmd "pyrefly lsp" --ty-cmd "ty server" --runs 10

    # Auto-clone and benchmark all packages
    python3 lsp_benchmark.py --install-envs install_envs.json -c pyrefly ty -r 10

    # Benchmark only 3 packages with 20 runs each
    python3 lsp_benchmark.py --install-envs install_envs.json -p 3 -r 20

    # Benchmark specific packages by name
    python3 lsp_benchmark.py --install-envs install_envs.json -n requests flask
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


JsonObj = Dict[str, Any]

SCRIPT_DIR: Path = Path(__file__).parent

# Type checker LSP commands for multi-package mode
TYPE_CHECKER_COMMANDS: Dict[str, str] = {
    "pyright": "pyright-langserver --stdio",
    "pyrefly": "pyrefly lsp",
    "ty": "ty server",
    "zuban": "zubanls",
}

DEFAULT_TYPE_CHECKERS: List[str] = ["pyright", "pyrefly", "ty", "zuban"]


@dataclasses.dataclass
class Position:
    line: int
    character: int


@dataclasses.dataclass
class Range:
    start: Position
    end: Position


@dataclasses.dataclass
class Location:
    uri: str
    range: Range


@dataclasses.dataclass
class DefinitionResult:
    # Backward compatible: "ok" indicates the LSP request succeeded (no timeout / protocol error).
    # Use "found" to check whether any definition locations were returned.
    ok: bool
    found: bool
    n_locations: int
    latency_ms: Optional[float]
    error: Optional[str]
    raw_result: Any
    locations: List[Location]


@dataclasses.dataclass
class BenchmarkCase:
    file_path: Path
    uri: str
    position: Position
    token: str
    line_text: str
    kind: str = "unknown"


def _path_to_uri(path: Path) -> str:
    # Use file:// URI with forward slashes.
    # Path.as_uri() requires absolute paths.
    return path.resolve().as_uri()


def _uri_to_path(uri: str) -> Path:
    # Minimal file URI decoding (good enough for local Windows paths)
    if uri.startswith("file:///"):
        # file:///C:/...
        path = uri[len("file:///") :]
        return Path(path.replace("/", "\\"))
    if uri.startswith("file://"):
        path = uri[len("file://") :]
        return Path(path)
    return Path(uri)


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@dataclasses.dataclass
class _AstOccurrence:
    line_1b: int
    col_0b: int
    token: str
    kind: str


def _safe_line(lines: List[str], idx0: int) -> str:
    if 0 <= idx0 < len(lines):
        return lines[idx0]
    return ""


def _token_from_line_at(lines: List[str], line_1b: int, col_0b: int) -> Optional[str]:
    # Best-effort: extract an identifier token from the given line at/after col.
    line0 = line_1b - 1
    if not (0 <= line0 < len(lines)):
        return None
    s = lines[line0]
    if col_0b < 0 or col_0b >= len(s):
        return None
    m = _IDENTIFIER_RE.search(s, pos=col_0b)
    if not m:
        return None
    # Ensure the match actually covers the caret position (common for ast col offsets).
    if not (m.start() <= col_0b <= m.end()):
        # fall back to nearest identifier that starts at col
        if m.start() != col_0b:
            return None
    return m.group(0)


def _collect_ast_occurrences(src: str) -> List[_AstOccurrence]:
    """Collect LSP-relevant symbol occurrences via AST.

    We prefer nodes that typically have a useful go-to-definition:
    - ast.Name (variable/reference)
    - ast.Attribute (x.y -> focus on `y`)
    - ast.Call (callee -> focus on function name / attribute)

    We intentionally exclude obvious builtins/typing primitives to reduce noise.
    """

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    # Collect imported names so we can bias toward "clickable" symbols.
    imported_names: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                asname = alias.asname or alias.name.split(".")[0]
                imported_names.add(asname)
                imported_modules.add(asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                asname = alias.asname or alias.name
                imported_names.add(asname)

    banned = {
        "True",
        "False",
        "None",
        "self",
        "cls",
        "int",
        "str",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "object",
    }

    occ: List[_AstOccurrence] = []

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if (
                node.id not in banned
                and hasattr(node, "lineno")
                and hasattr(node, "col_offset")
            ):
                kind = "imported_name" if node.id in imported_names else "name"
                occ.append(
                    _AstOccurrence(
                        line_1b=int(node.lineno),
                        col_0b=int(node.col_offset),
                        token=node.id,
                        kind=kind,
                    )
                )
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # For attribute `x.y`, col_offset typically points at `x`.
            # We don't have end offsets on all Python versions, so compute `y` column
            # by searching within the source line.
            if (
                node.attr in banned
                or not hasattr(node, "lineno")
                or not hasattr(node, "col_offset")
            ):
                self.generic_visit(node)
                return
            lineno = int(node.lineno)
            col0 = int(node.col_offset)
            # We'll patch the column later using the raw line when we build cases.
            base_is_imported_module = (
                isinstance(node.value, ast.Name) and node.value.id in imported_modules
            )
            kind = "imported_attr" if base_is_imported_module else "attr"
            occ.append(
                _AstOccurrence(line_1b=lineno, col_0b=col0, token=node.attr, kind=kind)
            )
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Prefer the callee position.
            fn = node.func
            if (
                isinstance(fn, ast.Name)
                and fn.id not in banned
                and hasattr(fn, "lineno")
                and hasattr(fn, "col_offset")
            ):
                kind = "imported_call" if fn.id in imported_names else "call"
                occ.append(
                    _AstOccurrence(
                        line_1b=int(fn.lineno),
                        col_0b=int(fn.col_offset),
                        token=fn.id,
                        kind=kind,
                    )
                )
            elif (
                isinstance(fn, ast.Attribute)
                and fn.attr not in banned
                and hasattr(fn, "lineno")
                and hasattr(fn, "col_offset")
            ):
                base_is_imported_module = (
                    isinstance(fn.value, ast.Name) and fn.value.id in imported_modules
                )
                kind = "imported_attr_call" if base_is_imported_module else "attr_call"
                occ.append(
                    _AstOccurrence(
                        line_1b=int(fn.lineno),
                        col_0b=int(fn.col_offset),
                        token=fn.attr,
                        kind=kind,
                    )
                )
            self.generic_visit(node)

    V().visit(tree)
    return occ


def pick_random_python_file(root: Path, *, rng: random.Random) -> Path:
    # Generic, cross-project discovery: pick any Python file under root.
    # Exclude common virtualenv/build/cache folders to avoid huge scans and noise.
    candidates: List[Path] = []

    excluded_dir_names = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".tox",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        ".idea",
        ".vscode",
    }

    for p in root.rglob("*.py"):
        parts_lower = {s.lower() for s in p.parts}
        if any(excl in parts_lower for excl in excluded_dir_names):
            continue
        candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "No .py files found under root. "
            "Run from a Python project root or pass --root to a folder that contains Python files."
        )

    return rng.choice(candidates)


def pick_random_case(
    root: Path, *, rng: random.Random, max_file_tries: int = 50
) -> BenchmarkCase:
    """Pick a random BenchmarkCase, retrying across files if needed.

    Some Python files (e.g. empty `__init__.py` stubs or files full of comments)
    might not yield any usable symbol occurrences.
    """

    last_err: Optional[Exception] = None
    for _ in range(max_file_tries):
        file_path = pick_random_python_file(root, rng=rng)
        try:
            return pick_random_identifier_case(file_path, rng=rng)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to pick a usable symbol after {max_file_tries} files; last error: {last_err}"
    )


def pick_random_identifier_case(
    file_path: Path, *, rng: random.Random
) -> BenchmarkCase:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Prefer AST-derived occurrences so we target a *real* symbol.
    ast_occ = _collect_ast_occurrences(text)
    candidates: List[Tuple[int, int, str, str]] = []
    for o in ast_occ:
        line0 = o.line_1b - 1
        if not (0 <= line0 < len(lines)):
            continue

        # For Attribute nodes we recorded col_offset of the base; try to locate the attribute token on the line.
        col0 = o.col_0b
        if o.token and o.token != "":
            # Search forward from AST col_offset first (preferred — avoids
            # matching an earlier occurrence of the same token on the line).
            idx = lines[line0].find(o.token, o.col_0b)
            if idx == -1:
                # Fallback: search from start (Attribute col_offset points to
                # base, not attr, so the token may precede col_offset).
                idx = lines[line0].find(o.token)
            if idx != -1:
                col0 = idx

        # Final sanity: verify an identifier exists at that position.
        tok = _token_from_line_at(lines, o.line_1b, col0) or o.token
        if not tok:
            continue

        candidates.append((line0, col0, tok, o.kind))

    # Fallback: regex scan if AST yields nothing (syntax errors, doc-only files, etc.)
    if not candidates:
        for i, line in enumerate(lines):
            for m in _IDENTIFIER_RE.finditer(line):
                tok = m.group(0)
                if tok in {
                    "True",
                    "False",
                    "None",
                    "self",
                    "cls",
                    "int",
                    "str",
                    "float",
                    "bool",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "object",
                }:
                    continue
                candidates.append((i, m.start(), tok, "regex"))

    if not candidates:
        raise RuntimeError(f"No identifier tokens found in {file_path}")

    # Bias towards imported symbols first (much more likely to have a definition).
    preferred_kinds = {
        "imported_name",
        "imported_attr",
        "imported_call",
        "imported_attr_call",
    }
    preferred = [c for c in candidates if c[3] in preferred_kinds]
    pool = preferred if preferred else candidates

    line, col, tok, kind = rng.choice(pool)
    uri = _path_to_uri(file_path)
    return BenchmarkCase(
        file_path=file_path,
        uri=uri,
        position=Position(line=line, character=col),
        token=tok,
        line_text=_safe_line(lines, line),
        kind=kind,
    )


class LspProtocolError(RuntimeError):
    pass


class LspClient:
    def __init__(self, name: str, argv: List[str], root: Path, *, trace: bool = False):
        self.name = name
        self.argv = argv
        self.root = root
        self.trace = trace

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._rx_queue: "queue.Queue[JsonObj]" = queue.Queue()
        self._pending: Dict[Union[int, str], "queue.Queue[JsonObj]"] = {}
        self._next_id = 1
        self._shutdown = False
        self._stderr_tail: "queue.Queue[str]" = queue.Queue(maxsize=200)

    def __enter__(self) -> "LspClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        except Exception:
            if exc is None:
                raise

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self.argv,
            cwd=str(self.root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self._proc.stdout is not None
        assert self._proc.stdin is not None
        assert self._proc.stderr is not None

        self._rx_thread = threading.Thread(
            target=self._rx_loop, name=f"{self.name}-lsp-rx", daemon=True
        )
        self._rx_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, name=f"{self.name}-lsp-stderr", daemon=True
        )
        self._stderr_thread.start()

    def _stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        stream = self._proc.stderr
        try:
            while True:
                line = stream.readline()
                if not line:
                    return
                s = line.decode("utf-8", errors="replace").rstrip("\r\n")
                try:
                    if self._stderr_tail.full():
                        _ = self._stderr_tail.get_nowait()
                    self._stderr_tail.put_nowait(s)
                except Exception:
                    pass
        except Exception:
            return

    def _stderr_tail_text(self, max_lines: int = 40) -> str:
        lines: List[str] = []
        try:
            while True:
                lines.append(self._stderr_tail.get_nowait())
        except queue.Empty:
            pass
        for s in lines:
            try:
                self._stderr_tail.put_nowait(s)
            except Exception:
                pass
        tail = lines[-max_lines:]
        return "\n".join(tail)

    def stop(self) -> None:
        if self._proc is None:
            return
        if not self._shutdown:
            try:
                self.request("shutdown", {})
            except Exception:
                pass
            try:
                self.notify("exit", {})
            except Exception:
                pass
            self._shutdown = True

        try:
            self._proc.terminate()
        except Exception:
            pass

        try:
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

        self._proc = None

    def initialize(self) -> None:
        root_uri = _path_to_uri(self.root)
        params = {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False, "linkSupport": True},
                },
                "workspace": {
                    "workspaceFolders": True,
                },
            },
            "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
            "clientInfo": {"name": "lsp-bench", "version": "0.1"},
            "trace": "verbose" if self.trace else "off",
        }
        self.request("initialize", params, timeout_s=120)
        self.notify("initialized", {})

    def change_configuration(self, settings: Any) -> None:
        self.notify("workspace/didChangeConfiguration", {"settings": settings})

    def open_document(
        self, uri: str, text: str, *, language_id: str = "python", version: int = 1
    ) -> None:
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": version,
                    "text": text,
                }
            },
        )

    def definition(
        self, uri: str, pos: Position, *, timeout_s: float = 60.0
    ) -> DefinitionResult:
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": pos.line, "character": pos.character},
        }
        t0 = time.perf_counter()
        try:
            resp = self.request("textDocument/definition", params, timeout_s=timeout_s)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            result = resp.get("result")
            locs = _parse_definition_result(result)
            found = len(locs) > 0
            return DefinitionResult(
                ok=True,
                found=found,
                n_locations=len(locs),
                latency_ms=dt_ms,
                error=None,
                raw_result=result,
                locations=locs,
            )
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return DefinitionResult(
                ok=False,
                found=False,
                n_locations=0,
                latency_ms=dt_ms,
                error=str(e),
                raw_result=None,
                locations=[],
            )

    def notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, *, timeout_s: float = 30.0) -> JsonObj:
        req_id = self._next_id
        self._next_id += 1

        waiter: "queue.Queue[JsonObj]" = queue.Queue(maxsize=1)
        self._pending[req_id] = waiter
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

        try:
            resp = waiter.get(timeout=timeout_s)
        except queue.Empty as e:
            tail = self._stderr_tail_text()
            extra = f"\n--- {self.name} stderr (tail) ---\n{tail}" if tail else ""
            raise TimeoutError(
                f"{self.name}: timeout waiting for response to {method}{extra}"
            ) from e
        finally:
            self._pending.pop(req_id, None)

        if "error" in resp:
            raise LspProtocolError(
                f"{self.name}: LSP error for {method}: {resp['error']}"
            )
        return resp

    def _send(self, msg: JsonObj) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(f"{self.name}: process not started")

        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")

        if self.trace:
            sys.stderr.write(f"[{self.name} ->] {msg.get('method', 'response')}\n")

        self._proc.stdin.write(header)
        self._proc.stdin.write(body)
        self._proc.stdin.flush()

    def _rx_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None

        stream = self._proc.stdout
        try:
            while True:
                headers: Dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    try:
                        k, v = line.decode("ascii", errors="replace").split(":", 1)
                    except ValueError:
                        continue
                    headers[k.strip().lower()] = v.strip()

                if "content-length" not in headers:
                    continue

                try:
                    length = int(headers["content-length"])
                except ValueError:
                    continue

                body = stream.read(length)
                if not body:
                    return

                try:
                    msg = json.loads(body.decode("utf-8", errors="replace"))
                except Exception:
                    continue

                if self.trace:
                    if "method" in msg:
                        sys.stderr.write(f"[{self.name} <-] notify {msg['method']}\n")
                    else:
                        sys.stderr.write(
                            f"[{self.name} <-] response id={msg.get('id')}\n"
                        )

                if "id" in msg and msg.get("id") in self._pending:
                    self._pending[msg["id"]].put(msg)
                else:
                    self._rx_queue.put(msg)
        except Exception:
            if self.trace:
                traceback.print_exc()


def _parse_definition_result(result: Any) -> List[Location]:
    if result is None:
        return []

    def loc_from(obj: Any) -> Optional[Location]:
        if not isinstance(obj, dict):
            return None
        if "targetUri" in obj and "targetRange" in obj:
            # LocationLink
            uri = obj["targetUri"]
            r = obj["targetRange"]
        elif "uri" in obj and "range" in obj:
            uri = obj["uri"]
            r = obj["range"]
        else:
            return None

        try:
            return Location(
                uri=str(uri),
                range=Range(
                    start=Position(
                        line=int(r["start"]["line"]),
                        character=int(r["start"]["character"]),
                    ),
                    end=Position(
                        line=int(r["end"]["line"]), character=int(r["end"]["character"])
                    ),
                ),
            )
        except Exception:
            return None

    locs: List[Location] = []
    if isinstance(result, list):
        for item in result:
            loc = loc_from(item)
            if loc:
                locs.append(loc)
    else:
        loc = loc_from(result)
        if loc:
            locs.append(loc)

    return locs


def _looks_like_valid_location(loc: Location, repo_root: Path) -> bool:
    """Check whether a definition location looks valid.

    Verifies the resolved path exists on disk and that the start line is
    within the file's line count.
    """
    p = _uri_to_path(loc.uri)
    try:
        p = p.resolve()
    except Exception:
        return False

    if not p.exists():
        return False

    if loc.range.start.line < 0 or loc.range.start.character < 0:
        return False
    if loc.range.end.line < 0 or loc.range.end.character < 0:
        return False

    # Verify line is within file bounds
    try:
        line_count = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
        if loc.range.start.line >= line_count:
            return False
    except OSError:
        return False

    return True


def wait_for_ready(
    lsp: LspClient,
    root: Path,
    *,
    timeout_s: float = 120.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Wait for an LSP server to finish indexing and become ready.

    Opens a trivial Python file and sends throwaway definition requests
    until the server responds without error, indicating it has finished
    initial indexing.
    """
    # Create a small probe file
    probe_path = root / "__lsp_bench_probe__.py"
    probe_text = "import os\nos.path.join\n"
    try:
        probe_path.write_text(probe_text, encoding="utf-8")
        probe_uri = _path_to_uri(probe_path)
        lsp.open_document(probe_uri, probe_text)

        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            res = lsp.definition(
                probe_uri, Position(line=1, character=8), timeout_s=poll_interval_s
            )
            if res.ok:
                return
            time.sleep(poll_interval_s)

        print(
            f"    Warning: {lsp.name} did not become ready within {timeout_s}s, proceeding anyway"
        )
    finally:
        probe_path.unlink(missing_ok=True)


def start_server(
    name: str,
    cmd: str,
    root: Path,
    *,
    trace: bool = False,
    settings: Any = None,
    init_timeout_s: float = 120.0,
) -> LspClient:
    """Start an LSP server, initialize it, and wait for it to be ready.

    Returns a live LspClient that can be reused for multiple definition
    requests.  The caller is responsible for calling ``stop()`` when done.
    """
    argv = _split_command(cmd)
    lsp = LspClient(name=name, argv=argv, root=root, trace=trace)
    lsp.start()
    lsp.initialize()
    if settings is not None:
        lsp.change_configuration(settings)
    print(f"    Waiting for {name} to finish indexing...")
    wait_for_ready(lsp, root, timeout_s=init_timeout_s)
    print(f"    {name} ready.")
    return lsp


def run_one_server(
    name: str,
    cmd: str,
    case: BenchmarkCase,
    root: Path,
    *,
    trace: bool = False,
    settings: Any = None,
    timeout_s: float = 10.0,
) -> DefinitionResult:
    """Run a single LSP server and measure Go to Definition latency.

    This is the legacy per-request startup path.  Prefer ``start_server()``
    + ``lsp.definition()`` for benchmarking steady-state latency.

    Args:
        name: Server name for logging.
        cmd: Command to start the LSP server.
        case: The benchmark case (file, position, token).
        root: Repository root path.
        trace: Enable verbose LSP tracing.
        settings: Optional LSP settings to apply.
        timeout_s: Timeout in seconds for the definition request (default: 10s).
                   Timeouts are counted as errors and do NOT contribute to latency stats.

    Returns:
        DefinitionResult with latency and location info.
    """
    argv = _split_command(cmd)
    with LspClient(name=name, argv=argv, root=root, trace=trace) as lsp:
        lsp.initialize()
        if settings is not None:
            lsp.change_configuration(settings)

        text = case.file_path.read_text(encoding="utf-8", errors="replace")
        lsp.open_document(case.uri, text)
        return lsp.definition(case.uri, case.position, timeout_s=timeout_s)


def _split_command(cmd: str) -> List[str]:
    argv = shlex.split(cmd, posix=os.name != "nt")

    if os.name == "nt" and argv:
        first = argv[0]
        normalized = first.replace("/", "\\")
        if normalized.lower().endswith("node_modules\\.bin\\pyright-langserver"):
            cmd_path = Path(normalized + ".cmd")
            if cmd_path.exists():
                argv[0] = str(cmd_path)

    return argv


# ---------------------------------------------------------------------------
# Multi-package orchestration helpers
# ---------------------------------------------------------------------------


def _load_packages(
    install_envs_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load packages from install_envs.json. Returns packages that have a github_url."""
    if install_envs_file is None:
        install_envs_file = SCRIPT_DIR / "install_envs.json"

    if not install_envs_file.exists():
        print(f"Error: {install_envs_file} not found")
        return []

    with open(install_envs_file, encoding="utf-8") as f:
        data = json.load(f)

    packages: List[Dict[str, Any]] = []
    for pkg in data.get("packages", []):
        github_url = pkg.get("github_url", "")
        if not github_url:
            continue
        name = pkg.get("name") or github_url.rstrip("/").split("/")[-1]
        packages.append({**pkg, "name": name})

    return packages


def _clone_package(
    github_url: str,
    name: str,
    dest: Path,
    timeout: int = 180,
) -> Optional[Path]:
    """Shallow-clone a GitHub repository."""
    target = dest / name
    try:
        print(f"  Cloning {github_url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", github_url, str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  Failed to clone: {result.stderr[:200]}")
            return None
        return target
    except subprocess.TimeoutExpired:
        print(f"  Timeout cloning {github_url}")
        return None
    except Exception as e:
        print(f"  Error cloning: {e}")
        return None


def _find_type_checker_command(checker: str) -> Optional[str]:
    """Find the command to run a type checker's LSP server, if available."""
    if checker not in TYPE_CHECKER_COMMANDS:
        return None

    cmd_parts = TYPE_CHECKER_COMMANDS[checker].split()
    executable = cmd_parts[0]

    which_cmd = "where" if sys.platform == "win32" else "which"
    result = subprocess.run([which_cmd, executable], capture_output=True)

    if result.returncode == 0:
        return TYPE_CHECKER_COMMANDS[checker]

    return None


def _get_type_checker_versions(type_checkers: List[str]) -> Dict[str, str]:
    """Get version strings for the requested type checkers."""
    versions: Dict[str, str] = {}
    version_commands: Dict[str, List[str]] = {
        "pyright": ["pyright", "--version"],
        "pyrefly": ["pyrefly", "--version"],
        "ty": ["ty", "--version"],
        "zuban": ["zuban", "--version"],
    }
    for name in type_checkers:
        cmd = version_commands.get(name)
        if not cmd:
            versions[name] = "unknown"
            continue
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                match = re.search(r"\d+\.\d+\.\d+", output)
                if match:
                    versions[name] = match.group(0)
                else:
                    parts = output.split()
                    versions[name] = parts[1] if len(parts) > 1 else parts[0]
            else:
                versions[name] = "unknown"
        except (subprocess.TimeoutExpired, OSError):
            versions[name] = "not installed"
    return versions


def _run_benchmark_for_package(
    package_path: Path,
    type_checkers: List[str],
    runs: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the LSP benchmark for a package across all type checkers.

    All type checkers are run in a single invocation to ensure they are
    tested on the exact same files and positions for fairness.
    """
    results: Dict[str, Dict[str, Any]] = {}

    available: List[Tuple[str, str]] = []
    for checker in type_checkers:
        cmd = _find_type_checker_command(checker)
        if not cmd:
            print(f"    Skipping {checker}: command not found")
            results[checker] = {
                "ok": False,
                "error": "Type checker not installed",
                "latency_ms": None,
            }
        else:
            available.append((checker, cmd))

    if not available:
        return results

    print(f"    Running {', '.join(c[0] for c in available)} together...")

    checker_names = [name for name, _ in available]
    args = [
        "--root",
        str(package_path),
        "--servers",
        ",".join(checker_names),
        "--runs",
        str(runs),
        "--timeout",
        "10",
    ]

    for checker, cmd in available:
        args.extend([f"--{checker}-cmd", cmd])

    if any(name == "pyright" for name, _ in available):
        args.append("--pyright-disable-indexing")

    if seed is not None:
        args.extend(["--seed", str(seed)])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    args.extend(["--json", str(tmp_path)])

    try:
        main(args)

        if tmp_path.exists():
            try:
                with open(tmp_path, encoding="utf-8") as f:
                    benchmark_data: Dict[str, Any] = json.load(f)

                for checker, _ in available:
                    summary = benchmark_data.get("summary", {}).get(checker, {})
                    latency = summary.get("latency_ms", {})
                    results[checker] = {
                        "ok": True,
                        "runs": runs,
                        "ok_count": summary.get("ok", 0),
                        "ok_pct": summary.get("ok_pct", 0.0),
                        "found_count": summary.get("found", 0),
                        "found_pct": summary.get("found_pct", 0.0),
                        "valid_count": summary.get("valid", 0),
                        "valid_pct": summary.get("valid_pct", 0.0),
                        "errors": summary.get("errors", 0),
                        "latency_ms": {
                            "p50": latency.get("p50"),
                            "p95": latency.get("p95"),
                            "min": latency.get("min"),
                            "max": latency.get("max"),
                            "mean": latency.get("mean"),
                        },
                    }
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            for checker, _ in available:
                results[checker] = {
                    "ok": False,
                    "error": "No output file generated",
                    "latency_ms": None,
                }
    except Exception as e:
        print(f"    Error running benchmark: {e}")
        for checker, _ in available:
            results[checker] = {
                "ok": False,
                "error": str(e),
                "latency_ms": None,
            }

    return results


def _compute_aggregate_stats(
    results: List[Dict[str, Any]],
    type_checkers: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate statistics across all packages."""
    stats: Dict[str, Dict[str, Any]] = {}

    for checker in type_checkers:
        latencies: List[float] = []
        valid_counts: List[int] = []
        found_counts: List[int] = []
        ok_counts: List[int] = []
        total_runs = 0
        packages_tested = 0

        for result in results:
            if result.get("error"):
                continue
            metrics = result.get("metrics", {}).get(checker, {})
            if not metrics.get("ok"):
                continue

            packages_tested += 1
            runs = metrics.get("runs", 0)
            total_runs += runs
            ok_counts.append(metrics.get("ok_count", 0))
            found_counts.append(metrics.get("found_count", 0))
            valid_counts.append(metrics.get("valid_count", 0))

            latency = metrics.get("latency_ms") or {}
            mean_latency = latency.get("mean")
            if mean_latency is not None:
                latencies.append(float(mean_latency))

        avg_latency: Optional[float] = None
        min_latency: Optional[float] = None
        max_latency: Optional[float] = None

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

        ok_rate = (sum(ok_counts) / total_runs * 100) if total_runs > 0 else 0.0
        success_rate = (sum(valid_counts) / total_runs * 100) if total_runs > 0 else 0.0

        stats[checker] = {
            "packages_tested": packages_tested,
            "total_runs": total_runs,
            "total_ok": sum(ok_counts),
            "total_found": sum(found_counts),
            "total_valid": sum(valid_counts),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "ok_rate": ok_rate,
            "success_rate": success_rate,
        }

    return stats


def _unresolved_entry(
    case: BenchmarkCase,
    *,
    reason: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an unresolved-case payload for the JSON report."""
    entry: Dict[str, Any] = {
        "file": str(case.file_path),
        "uri": case.uri,
        "line": case.position.line,
        "character": case.position.character,
        "line_1b": case.position.line + 1,
        "character_1b": case.position.character + 1,
        "token": case.token,
        "kind": case.kind,
        "line_text": case.line_text,
        "reason": reason,
    }
    if error is not None:
        entry["error"] = error
    return entry


def _run_multi_package(args: argparse.Namespace) -> int:
    """Run multi-package mode: clone packages and benchmark each one."""
    type_checkers = list(args.checkers)
    output_dir = args.output or SCRIPT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_packages = _load_packages(args.install_envs)
    if not all_packages:
        print("Error: No packages found in install_envs.json")
        return 1

    if args.package_names:
        name_set = set(args.package_names)
        packages = [p for p in all_packages if p["name"] in name_set]
        if not packages:
            print(
                f"Warning: None of the specified packages found: {args.package_names}"
            )
            return 1
    elif args.packages:
        packages = all_packages[: args.packages]
    else:
        packages = all_packages

    runs_per_package = max(1, args.runs)

    print("=" * 70)
    print("LSP Benchmark (multi-package mode)")
    print("=" * 70)
    print(f"Packages to benchmark: {len(packages)}")
    print(f"Type checkers: {', '.join(type_checkers)}")
    print(f"Runs per package: {runs_per_package}")
    print("=" * 70)

    versions = _get_type_checker_versions(type_checkers)
    print("\nType Checker Versions:")
    for name, version in versions.items():
        print(f"  {name}: {version}")
    print()

    all_results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, pkg in enumerate(packages, 1):
            name = pkg["name"]
            github_url = pkg.get("github_url", "")

            print(f"\n[{i}/{len(packages)}] {name}")

            if not github_url:
                print("  Skipping: No GitHub URL")
                all_results.append(
                    {
                        "package_name": name,
                        "github_url": None,
                        "error": "No GitHub URL",
                        "metrics": {},
                    }
                )
                continue

            package_path = _clone_package(github_url, name, temp_path)
            if not package_path:
                all_results.append(
                    {
                        "package_name": name,
                        "github_url": github_url,
                        "error": "Failed to clone",
                        "metrics": {},
                    }
                )
                continue

            print(f"  Running benchmarks ({runs_per_package} runs each)...")
            try:
                metrics = _run_benchmark_for_package(
                    package_path, type_checkers, runs=runs_per_package, seed=args.seed
                )
                all_results.append(
                    {
                        "package_name": name,
                        "github_url": github_url,
                        "error": None,
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                print(f"  Error running benchmarks: {e}")
                all_results.append(
                    {
                        "package_name": name,
                        "github_url": github_url,
                        "error": f"Benchmark failed: {e}",
                        "metrics": {},
                    }
                )
            finally:
                shutil.rmtree(package_path, ignore_errors=True)

    aggregate = _compute_aggregate_stats(all_results, type_checkers)

    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y-%m-%d")

    os_name = args.os_name
    if os_name:
        output_file = output_dir / f"lsp_benchmark_{date_str}_{os_name}.json"
        latest_file = output_dir / f"lsp_latest-{os_name}.json"
    else:
        output_file = output_dir / f"lsp_benchmark_{date_str}.json"
        latest_file = output_dir / "lsp_latest.json"

    output_data: Dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "date": date_str,
        "type_checkers": type_checkers,
        "type_checker_versions": versions,
        "package_count": len(packages),
        "runs_per_package": runs_per_package,
        "aggregate": aggregate,
        "results": all_results,
    }
    if os_name:
        output_data["os"] = os_name

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print("\nAggregate Results:")
    print("-" * 70)
    for checker in type_checkers:
        s = aggregate.get(checker, {})
        if s.get("packages_tested", 0) == 0:
            print(f"  {checker}: No successful benchmarks")
            continue
        latency = s.get("avg_latency_ms")
        latency_str = f"{latency:.1f}ms" if latency else "N/A"
        print(f"  {checker}:")
        print(f"    Packages tested: {s.get('packages_tested', 0)}")
        print(f"    Total runs: {s.get('total_runs', 0)}")
        print(
            f"    Valid definitions: {s.get('total_valid', 0)} "
            f"({s.get('success_rate', 0):.1f}%)"
        )
        print(f"    Avg latency: {latency_str}")

    print(f"\nResults saved to: {output_file}")
    return 0


def run_benchmark_loop(
    servers: List[Tuple[str, LspClient]],
    root: Path,
    runs: int,
    rng: random.Random,
    *,
    timeout_s: float = 10.0,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run N definition queries against pre-initialized servers.

    Servers are queried sequentially for each case to avoid CPU/memory
    contention that would inflate and destabilize latency measurements.

    Returns (report_dict, agg_dict).
    """
    report: Dict[str, Any] = {
        "root": str(root),
        "runs": runs,
        "servers": [name for name, _ in servers],
        "cases": [],
        "summary": {},
    }

    agg: Dict[str, Dict[str, Any]] = {
        name: {
            "ok": 0,
            "found": 0,
            "valid": 0,
            "latencies_ms": [],
            "errors": 0,
            "timeouts": 0,
        }
        for name, _ in servers
    }

    for run_idx in range(runs):
        case = pick_random_case(root, rng=rng)

        case_payload: Dict[str, Any] = {
            "run": run_idx,
            "picked": {
                "file": str(case.file_path),
                "uri": case.uri,
                "line": case.position.line,
                "character": case.position.character,
                "line_1b": case.position.line + 1,
                "character_1b": case.position.character + 1,
                "token": case.token,
                "kind": case.kind,
                "line_text": case.line_text,
            },
            "results": {},
            "unresolved": {},
        }

        print(
            f"Run {run_idx + 1}/{runs}: {case.file_path}:{case.position.line + 1}:{case.position.character + 1} token={case.token} kind={case.kind}"
        )

        # Open the file on each server
        text = case.file_path.read_text(encoding="utf-8", errors="replace")
        for _server_name, lsp in servers:
            lsp.open_document(case.uri, text)

        # Query each server sequentially to avoid contention
        for server_name, lsp in servers:
            try:
                res = lsp.definition(case.uri, case.position, timeout_s=timeout_s)
                locations_payload = [
                    {
                        "uri": loc.uri,
                        "start": dataclasses.asdict(loc.range.start),
                        "end": dataclasses.asdict(loc.range.end),
                        "valid": _looks_like_valid_location(loc, root),
                    }
                    for loc in res.locations
                ]
                any_valid = any(r.get("valid") for r in locations_payload)

                case_payload["results"][server_name] = {
                    "ok": res.ok,
                    "found": res.found,
                    "n_locations": res.n_locations,
                    "latency_ms": res.latency_ms,
                    "error": res.error,
                    "locations": locations_payload,
                }

                if res.ok:
                    agg[server_name]["ok"] += 1
                if res.found:
                    agg[server_name]["found"] += 1
                if any_valid:
                    agg[server_name]["valid"] += 1
                if res.ok and res.latency_ms is not None:
                    agg[server_name]["latencies_ms"].append(res.latency_ms)

                if not locations_payload:
                    case_payload["unresolved"][server_name] = _unresolved_entry(
                        case, reason="no_definition_locations"
                    )
            except Exception as e:
                agg[server_name]["errors"] += 1
                case_payload["results"][server_name] = {
                    "ok": False,
                    "found": False,
                    "n_locations": 0,
                    "latency_ms": None,
                    "error": str(e),
                    "locations": [],
                }
                case_payload["unresolved"][server_name] = _unresolved_entry(
                    case, reason="server_exception", error=str(e)
                )

        report["cases"].append(case_payload)

    # Compute summary
    def _pct(n: int) -> float:
        return (100.0 * n / runs) if runs else 0.0

    for server_name, _ in servers:
        lats = agg[server_name]["latencies_ms"]
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted) // 2] if lats_sorted else None
        p95 = lats_sorted[int(len(lats_sorted) * 0.95)] if lats_sorted else None
        report["summary"][server_name] = {
            "ok": agg[server_name]["ok"],
            "ok_pct": _pct(agg[server_name]["ok"]),
            "found": agg[server_name]["found"],
            "found_pct": _pct(agg[server_name]["found"]),
            "valid": agg[server_name]["valid"],
            "valid_pct": _pct(agg[server_name]["valid"]),
            "errors": agg[server_name]["errors"],
            "latency_ms": {
                "count": len(lats),
                "p50": p50,
                "p95": p95,
                "min": min(lats) if lats else None,
                "max": max(lats) if lats else None,
                "mean": (sum(lats) / len(lats)) if lats else None,
            },
        }

    return report, agg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark Go to Definition latency across LSP servers",
    )

    ap.add_argument(
        "--root", type=Path, default=None, help="Repo root (workspace folder)"
    )
    ap.add_argument(
        "--servers",
        type=str,
        default=None,
        help=(
            "Comma-separated server names to run. "
            "Supported: pyrefly,ty,zuban,pyright. "
            "If omitted, runs all servers for which a --*-cmd was provided."
        ),
    )
    ap.add_argument(
        "--pyrefly-cmd",
        type=str,
        default=None,
        help="Command to start pyrefly LSP over stdio",
    )
    ap.add_argument(
        "--ty-cmd", type=str, default=None, help="Command to start ty LSP over stdio"
    )
    ap.add_argument(
        "--zuban-cmd",
        type=str,
        default=None,
        help="Command to start zuban LSP over stdio (optional)",
    )
    ap.add_argument(
        "--pyright-cmd",
        type=str,
        default=None,
        help="Command to start pyright LSP over stdio (optional)",
    )
    ap.add_argument(
        "--trace", action="store_true", help="Verbose LSP wire trace to stderr"
    )
    ap.add_argument(
        "--settings-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object to send via workspace/didChangeConfiguration after initialized. "
            "Applied to all selected servers."
        ),
    )
    ap.add_argument(
        "--pyright-disable-indexing",
        action="store_true",
        help=(
            "If set, sends Pyright configuration to disable indexing and related scanning "
            + "(indexing=false, autoSearchPaths=false, useLibraryCodeForTypes=false)."
        ),
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="Write machine-readable JSON report",
    )

    ap.add_argument(
        "--install-envs",
        type=Path,
        default=None,
        help="Path to install_envs.json (enables multi-package mode)",
    )
    ap.add_argument(
        "--packages",
        "-p",
        type=int,
        default=None,
        help="Max number of packages to benchmark (multi-package mode)",
    )
    ap.add_argument(
        "--package-names",
        "-n",
        nargs="+",
        default=None,
        help="Specific package names to benchmark (multi-package mode)",
    )
    ap.add_argument(
        "--checkers",
        "-c",
        nargs="+",
        default=DEFAULT_TYPE_CHECKERS,
        help=f"Type checkers to benchmark (multi-package mode, default: {' '.join(DEFAULT_TYPE_CHECKERS)})",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for results (multi-package mode)",
    )
    ap.add_argument(
        "--os-name",
        type=str,
        default=None,
        help="OS name for output filename (e.g., macos)",
    )

    ap.add_argument(
        "--seed", "-s", type=int, default=None, help="RNG seed for repeatability"
    )
    ap.add_argument(
        "--runs",
        "-r",
        type=int,
        default=None,
        help="Number of random symbol queries per run (default: 1 in single-repo, 100 in multi-package)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        dest="timeout_s",
        help=(
            "Timeout in seconds for each Go to Definition request (default: 10s). "
            "Requests that timeout are counted as errors and do NOT contribute to latency statistics."
        ),
    )
    ap.add_argument(
        "--init-timeout",
        type=float,
        default=120.0,
        dest="init_timeout_s",
        help=(
            "Timeout in seconds to wait for each server to finish indexing "
            "before starting benchmark runs (default: 120s)."
        ),
    )
    args = ap.parse_args(argv)

    if args.install_envs is not None:
        if args.runs is None:
            args.runs = 100
        return _run_multi_package(args)

    if args.runs is None:
        args.runs = 1

    root = (args.root or Path.cwd()).resolve()
    rng = random.Random(args.seed)

    runs = max(1, int(args.runs))

    settings_payload: Any = None
    if args.settings_json is not None:
        try:
            settings_payload = json.loads(args.settings_json)
        except Exception as e:
            raise SystemExit(f"--settings-json must be valid JSON: {e}")

    if args.pyright_disable_indexing:
        pyright_settings = {
            "python": {
                "analysis": {
                    "indexing": False,
                    "autoSearchPaths": False,
                    "useLibraryCodeForTypes": False,
                }
            }
        }
        if settings_payload is None:
            settings_payload = pyright_settings
        elif isinstance(settings_payload, dict) and isinstance(pyright_settings, dict):
            settings_payload = {**pyright_settings, **settings_payload}

    supported = {"pyrefly", "ty", "zuban", "pyright"}

    def _provided(cmd: Optional[str]) -> bool:
        return cmd is not None and bool(str(cmd).strip())

    if args.servers is None:
        requested: List[str] = []
        if _provided(args.pyrefly_cmd):
            requested.append("pyrefly")
        if _provided(args.ty_cmd):
            requested.append("ty")
        if _provided(args.zuban_cmd):
            requested.append("zuban")
        if _provided(args.pyright_cmd):
            requested.append("pyright")
        if not requested:
            raise SystemExit(
                "No servers selected: either pass --servers, or provide at least one of "
                "--pyrefly-cmd/--ty-cmd/--zuban-cmd/--pyright-cmd."
            )
    else:
        requested = [
            s.strip().lower() for s in str(args.servers).split(",") if s.strip()
        ]
        unknown = [s for s in requested if s not in supported]
        if unknown:
            raise SystemExit(
                f"Unknown server(s) in --servers: {unknown}. Supported: {sorted(supported)}"
            )

    def _need(name: str, cmd: Optional[str]) -> str:
        if cmd is None or not str(cmd).strip():
            raise SystemExit(
                f"--{name}-cmd is required when --servers includes '{name}'"
            )
        return str(cmd)

    servers: List[Tuple[str, str]] = []
    if "pyrefly" in requested:
        servers.append(("pyrefly", _need("pyrefly", args.pyrefly_cmd)))
    if "ty" in requested:
        servers.append(("ty", _need("ty", args.ty_cmd)))
    if "zuban" in requested:
        servers.append(("zuban", _need("zuban", args.zuban_cmd)))
    if "pyright" in requested:
        servers.append(("pyright", _need("pyright", args.pyright_cmd)))

    # Start all servers once and wait for indexing to complete.
    # Servers are started sequentially to avoid startup contention.
    live_servers: List[Tuple[str, LspClient]] = []
    for server_name, cmd in servers:
        per_server_settings = settings_payload
        if (
            args.pyright_disable_indexing
            and server_name != "pyright"
            and args.settings_json is None
        ):
            per_server_settings = None

        try:
            lsp = start_server(
                server_name,
                cmd,
                root,
                trace=args.trace,
                settings=per_server_settings,
                init_timeout_s=float(args.init_timeout_s),
            )
            live_servers.append((server_name, lsp))
        except Exception as e:
            print(f"  Failed to start {server_name}: {e}")

    try:
        report, agg = run_benchmark_loop(
            live_servers,
            root,
            runs,
            rng,
            timeout_s=float(args.timeout_s),
        )
    finally:
        # Shut down all servers
        for _, lsp in live_servers:
            try:
                lsp.stop()
            except Exception:
                pass

    report["seed"] = args.seed
    report["ts"] = time.time()

    print("Summary:")
    for server_name, _ in live_servers:
        s = report["summary"].get(server_name, {})
        lat = s.get("latency_ms", {})
        if lat.get("count"):
            print(
                f"  {server_name}: ok={s['ok']}/{runs} valid={s['valid']}/{runs} errors={s['errors']} p50={lat['p50']:.1f}ms p95={lat['p95']:.1f}ms"
            )
        else:
            print(
                f"  {server_name}: ok={s.get('ok', 0)}/{runs} valid={s.get('valid', 0)}/{runs} errors={s.get('errors', 0)} (no latency samples)"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
