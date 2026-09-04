#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""Measure steady-state memory of Python language servers under a scripted IDE workload.

Each run starts one server over stdio, drives it through fixed checkpoints, and records
the memory of its process tree: physical footprint on macOS (what Activity Monitor
shows) and PSS on Linux. Projects, workloads, server versions and the dependency
resolution date are pinned in lsp_memory_projects.json.

The script is self-contained (std-lib only). It needs `git` and `uv` on PATH.

Usage:
    # Clone projects, create venvs, install the pinned servers
    python3 lsp_memory_benchmark.py setup

    # Run every project and configuration 3 times, after one warmup run each
    python3 lsp_memory_benchmark.py run --out results/lsp-memory-macbook

    # Print median/min/max per checkpoint
    python3 lsp_memory_benchmark.py summarize results/lsp-memory-macbook
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECTS = SCRIPT_DIR / "lsp_memory_projects.json"
DEFAULT_WORKDIR = Path.home() / "pyrefly-lsp-memory"

# Each configuration is (server, basedpyright diagnosticMode). Pyrefly has no
# equivalent setting: it indexes the project in the background regardless.
CONFIGS = {
    "pyrefly": ("pyrefly", None),
    "basedpyright": ("basedpyright", "openFilesOnly"),
    "basedpyright-workspace": ("basedpyright", "workspace"),
}
CHECKPOINTS = [
    "initialized",
    "files_open",
    "find_references",
    "edits_round_1",
    "edits_round_2",
]


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------


class _RusageInfoV0(ctypes.Structure):
    """`struct rusage_info_v0` from macOS <sys/resource.h>."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
        )
    ]


def memory_metric() -> str:
    return "phys_footprint" if sys.platform == "darwin" else "pss"


@functools.cache
def _libproc() -> ctypes.CDLL:
    return ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def process_memory_kb(pid: int) -> tuple[int, int] | None:
    """Return (memory_kb, rss_kb) for one process, or None if it has exited.

    The memory value is the physical footprint on macOS and PSS on Linux.
    """
    if sys.platform == "darwin":
        info = _RusageInfoV0()
        if _libproc().proc_pid_rusage(pid, 0, ctypes.byref(info)) != 0:
            return None
        return info.ri_phys_footprint // 1024, info.ri_resident_size // 1024
    pss = rss = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    pss = int(line.split()[1])
                elif line.startswith("Rss:"):
                    rss = int(line.split()[1])
    except OSError:
        return None
    return pss, rss


def process_tree(root: int) -> list[int]:
    """Return `root` and all of its descendants."""
    output = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True
    ).stdout
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        pid, ppid = map(int, line.split())
        children.setdefault(ppid, []).append(pid)
    pids, stack = [], [root]
    while stack:
        pid = stack.pop()
        pids.append(pid)
        stack.extend(children.get(pid, []))
    return pids


def check_memory_reader() -> None:
    """Cross-check our RSS reading against `ps`, to catch a wrong struct layout."""
    ours = process_memory_kb(os.getpid())
    ps_rss = int(
        subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if ours is None or not 0.75 <= ours[1] / ps_rss <= 1.25:
        sys.exit(f"error: memory reader disagrees with ps (ours={ours}, ps={ps_rss})")


class Sampler(threading.Thread):
    """Samples the memory of a process tree at a fixed interval."""

    def __init__(self, pid: int, interval: float) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.start_time = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            mem = rss = procs = 0
            for pid in process_tree(self.pid):
                reading = process_memory_kb(pid)
                if reading is not None:
                    mem += reading[0]
                    rss += reading[1]
                    procs += 1
            with self.lock:
                self.samples.append(
                    {
                        "t": round(self.now(), 3),
                        "mem_kb": mem,
                        "rss_kb": rss,
                        "procs": procs,
                    }
                )
            self.stopped.wait(self.interval)

    def since(self, t: float) -> list[dict[str, Any]]:
        with self.lock:
            return [s for s in self.samples if s["t"] >= t]

    def now(self) -> float:
        return time.monotonic() - self.start_time


# ---------------------------------------------------------------------------
# LSP client
# ---------------------------------------------------------------------------


class LspClient:
    """A minimal stdio LSP client that answers server requests from `settings`."""

    def __init__(
        self, command: list[str], cwd: Path, settings: dict[str, Any], log_path: Path
    ) -> None:
        self.settings = settings
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )
        stdin, stdout = self.proc.stdin, self.proc.stdout
        assert stdin is not None and stdout is not None, "Popen was given PIPE"
        self.stdin, self.stdout = stdin, stdout
        self.write_lock = threading.Lock()
        self.next_id = 0
        self.pending: dict[int, Queue] = {}
        self.diagnostics: dict[str, list[dict[str, Any]]] = {}
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _send(self, message: dict[str, Any]) -> None:
        body = json.dumps(message).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        with self.write_lock:
            self.stdin.write(header + body)
            self.stdin.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        with self.write_lock:
            self.next_id += 1
            msg_id = self.next_id
        queue: Queue = Queue()
        self.pending[msg_id] = queue
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            response = queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError(f"{method} timed out after {timeout}s") from None
        finally:
            del self.pending[msg_id]
        if "error" in response:
            raise RuntimeError(f"{method} failed: {response['error']}")
        return response.get("result")

    def _read_loop(self) -> None:
        stdout = self.stdout
        while True:
            length = None
            while True:
                line = stdout.readline()
                if not line:
                    return  # The server exited.
                line = line.strip()
                if not line:
                    break
                name, _, value = line.decode().partition(":")
                if name.lower() == "content-length":
                    length = int(value)
            if length is None:
                raise RuntimeError("LSP message without a Content-Length header")
            message = json.loads(stdout.read(length))
            if "method" in message and "id" in message:
                self._handle_server_request(message)
            elif "method" in message:
                self._handle_notification(message)
            else:
                # A response that arrives after its request timed out has no queue.
                queue = self.pending.get(message.get("id"))
                if queue is not None:
                    queue.put(message)

    def _handle_server_request(self, message: dict[str, Any]) -> None:
        if message["method"] == "workspace/configuration":
            items = message["params"]["items"]
            result = [self._section(item.get("section")) for item in items]
        else:
            # Registration, progress creation, refresh requests, etc. need no action.
            result = None
        self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _section(self, section: str | None) -> Any:
        value: Any = self.settings
        for part in section.split(".") if section else []:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _handle_notification(self, message: dict[str, Any]) -> None:
        method, params = message["method"], message.get("params", {})
        if method == "textDocument/publishDiagnostics":
            self.diagnostics[params["uri"]] = params["diagnostics"]
        elif method in ("window/logMessage", "window/showMessage"):
            self.log.write(f"[{method}] {params.get('message', '')}\n")
            self.log.flush()

    def shutdown(self) -> None:
        try:
            self.request("shutdown", {}, timeout=30)
            self.notify("exit", {})
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.log.close()


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class Document:
    """An open file whose edits only touch helper functions appended to its text.

    A body edit changes the return value of an annotated helper, so the module's
    interface is unchanged. An interface edit adds a new top-level function, so the
    module's exports change and its dependents must be rechecked. The body helper is
    present from `didOpen`, so the first body edit does not change the interface.
    Edits are only sent to the server; files on disk are never modified.
    """

    def __init__(self, path: Path) -> None:
        self.uri = path.as_uri()
        self.original = path.read_text()
        if not self.original.endswith("\n"):
            self.original += "\n"
        self.version = 0
        self.body_edits = 0
        self.interface_edits = 0

    def text(self) -> str:
        text = self.original
        text += f"\n\ndef _bench_body() -> int:\n    return {self.body_edits}\n"
        for i in range(1, self.interface_edits + 1):
            text += f"\n\ndef _bench_api_{i}(x: int) -> int:\n    return x\n"
        return text

    def apply(self, kind: str) -> str:
        """Apply one edit of `kind` and return the new full text."""
        if kind == "body":
            self.body_edits += 1
        elif kind == "interface":
            self.interface_edits += 1
        else:
            raise ValueError(f"unknown edit kind {kind!r}")
        self.version += 1
        return self.text()


def find_definition(path: Path, symbol: str) -> dict[str, int]:
    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+({re.escape(symbol)})\b")
    for line_no, line in enumerate(path.read_text().splitlines()):
        match = pattern.match(line)
        if match:
            return {"line": line_no, "character": match.start(1)}
    raise ValueError(f"no definition of {symbol!r} found in {path}")


def wait_steady(
    sampler: Sampler, since: float, window: float, tolerance: float, timeout: float
) -> dict[str, Any]:
    """Block until memory varies by at most `tolerance` over the last `window` seconds."""
    while True:
        elapsed = sampler.now() - since
        recent = sampler.since(sampler.now() - window)
        if elapsed >= window and recent:
            values = [s["mem_kb"] for s in recent]
            if max(values) - min(values) <= tolerance * max(values):
                last = recent[-1]
                return {
                    "steady": True,
                    "mem_kb": last["mem_kb"],
                    "rss_kb": last["rss_kb"],
                }
        if elapsed >= timeout:
            last = sampler.since(0)[-1]
            return {"steady": False, "mem_kb": last["mem_kb"], "rss_kb": last["rss_kb"]}
        time.sleep(1)


def run_workload(
    command: list[str],
    project: Path,
    settings: dict[str, Any],
    workload: dict[str, Any],
    log_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Drive one server through all checkpoints and return the measurements."""
    client = LspClient(command, project, settings, log_path)
    sampler = Sampler(client.proc.pid, args.interval)
    sampler.start()
    checkpoints: list[dict[str, Any]] = []

    def checkpoint(name: str, since: float, **extra: Any) -> None:
        result = wait_steady(
            sampler,
            since,
            args.steady_window,
            args.steady_tolerance,
            args.steady_timeout,
        )
        peak = max(s["mem_kb"] for s in sampler.since(since))
        result.update(name=name, t=round(sampler.now(), 1), peak_kb=peak, **extra)
        checkpoints.append(result)
        status = "steady" if result["steady"] else "NOT STEADY (timed out)"
        print(
            f"      [{result['t']:>7.1f}s] {name}: {result['mem_kb'] / 1024:.0f} MiB, "
            f"peak {peak / 1024:.0f} MiB ({status})",
            flush=True,
        )

    try:
        t = sampler.now()
        client.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": project.as_uri(),
                "workspaceFolders": [{"uri": project.as_uri(), "name": project.name}],
                "capabilities": {
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "didChangeConfiguration": {"dynamicRegistration": True},
                    },
                    "textDocument": {
                        "synchronization": {
                            "didSave": True,
                            "dynamicRegistration": True,
                        },
                        "publishDiagnostics": {},
                        "references": {},
                    },
                    "window": {"workDoneProgress": True},
                },
            },
            timeout=args.request_timeout,
        )
        client.notify("initialized", {})
        client.notify("workspace/didChangeConfiguration", {"settings": settings})
        checkpoint("initialized", t)

        t = sampler.now()
        documents = {rel: Document(project / rel) for rel in workload["open"]}
        for doc in documents.values():
            client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": doc.uri,
                        "languageId": "python",
                        "version": doc.version,
                        "text": doc.text(),
                    }
                },
            )
        checkpoint("files_open", t)
        checkpoints[-1]["unresolved_imports"] = [
            f"{Path(doc.uri).name}:{d['range']['start']['line'] + 1}: {d['message']}"
            for doc in documents.values()
            for d in client.diagnostics.get(doc.uri, [])
            if re.search(
                r"could not be resolved(?! from source)|cannot find module",
                d["message"],
                re.IGNORECASE,
            )
        ]

        t = sampler.now()
        ref = workload["references"]
        locations = client.request(
            "textDocument/references",
            {
                "textDocument": {"uri": documents[ref["file"]].uri},
                "position": find_definition(project / ref["file"], ref["symbol"]),
                "context": {"includeDeclaration": True},
            },
            timeout=args.request_timeout,
        )
        checkpoint("find_references", t, reference_count=len(locations or []))

        for round_no in (1, 2):
            t = sampler.now()
            for edit in workload["edits"]:
                doc = documents[edit["file"]]
                text = doc.apply(edit["kind"])
                client.notify(
                    "textDocument/didChange",
                    {
                        "textDocument": {"uri": doc.uri, "version": doc.version},
                        "contentChanges": [{"text": text}],
                    },
                )
                client.notify(
                    "textDocument/didSave", {"textDocument": {"uri": doc.uri}}
                )
                time.sleep(args.edit_delay)
            checkpoint(f"edits_round_{round_no}", t)
    finally:
        client.shutdown()
        sampler.stopped.set()
        sampler.join()

    return {"checkpoints": checkpoints, "samples": sampler.samples}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs: Any) -> str:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd, check=True, text=True, capture_output=True, **kwargs
    ).stdout


def _venv_bin(venv: Path, name: str) -> Path:
    return venv / "bin" / name


def _select_projects(spec: dict[str, Any], names: list[str] | None) -> list[dict]:
    projects = spec["projects"]
    if names:
        unknown = set(names) - {p["name"] for p in projects}
        if unknown:
            sys.exit(f"error: unknown projects: {', '.join(sorted(unknown))}")
        projects = [p for p in projects if p["name"] in names]
    return projects


def setup(args: argparse.Namespace) -> None:
    spec = json.loads(args.projects_file.read_text())
    workdir = args.workdir.resolve()
    uv_pin = ["--exclude-newer", spec["exclude_newer"]]

    print("Installing servers")
    servers_venv = workdir / "servers"
    _run(
        ["uv", "venv", "--clear", str(servers_venv), "--python", spec["python_version"]]
    )
    _run(
        ["uv", "pip", "install", "--python", str(servers_venv)]
        + [f"{name}=={version}" for name, version in spec["servers"].items()]
    )

    for project in _select_projects(spec, args.project):
        name = project["name"]
        print(f"Setting up {name}")
        checkout = workdir / "projects" / name
        if not checkout.exists():
            _run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    project["github_url"],
                    str(checkout),
                ]
            )
        _run(["git", "-C", str(checkout), "checkout", "--detach", project["commit"]])

        # Empty configs override any type checker settings the project ships (for
        # example pandas' `[tool.pyright]`), so both servers use their defaults
        # over the whole checkout.
        (checkout / "pyrefly.toml").write_text("")
        (checkout / "pyrightconfig.json").write_text("{}\n")

        # The venv lives outside the checkout so neither server indexes it.
        venv = workdir / "venvs" / name
        _run(["uv", "venv", "--clear", str(venv), "--python", spec["python_version"]])
        install = ["uv", "pip", "install", "--python", str(venv)] + uv_pin
        install += project.get("uv_args", [])
        env = {**os.environ, **project.get("install_env", {})}
        if project.get("install"):
            _run(install + ["-e", str(checkout)], env=env)
        if project.get("deps"):
            _run(install + project["deps"], env=env)
        freeze = _run(["uv", "pip", "freeze", "--python", str(venv)])
        (workdir / "venvs" / f"{name}.freeze.txt").write_text(freeze)
    print(f"Setup complete in {workdir}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _environment(workdir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    env: dict[str, Any] = {
        "date": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_metric": memory_metric(),
        "servers": spec["servers"],
    }
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            env[key] = _run(["sysctl", "-n", key]).strip()
    else:
        with open("/proc/meminfo") as f:
            env["MemTotal"] = f.readline().split(":")[1].strip()
    env["freezes"] = {
        path.name.removesuffix(".freeze.txt"): path.read_text().splitlines()
        for path in sorted((workdir / "venvs").glob("*.freeze.txt"))
    }
    return env


def lsp_settings(config: str, python: str) -> dict[str, Any]:
    """Settings returned for `workspace/configuration` under `config`."""
    settings: dict[str, Any] = {"python": {"pythonPath": python}}
    diagnostic_mode = CONFIGS[config][1]
    if diagnostic_mode is not None:
        settings["basedpyright"] = {"analysis": {"diagnosticMode": diagnostic_mode}}
    return settings


def run(args: argparse.Namespace) -> None:
    spec = json.loads(args.projects_file.read_text())
    workdir = args.workdir.resolve()
    servers_venv = workdir / "servers"
    pyrefly = _venv_bin(servers_venv, "pyrefly")
    if not pyrefly.exists():
        sys.exit(f"error: {pyrefly} not found; run `setup` first")
    # Launch basedpyright's Node process directly: the `basedpyright-langserver`
    # entry point is a Python wrapper that stays alive and would be measured too.
    node, langserver = _run(
        [
            str(_venv_bin(servers_venv, "python")),
            "-c",
            "import os, basedpyright, nodejs_wheel; "
            "print(os.path.join(os.path.dirname(nodejs_wheel.__file__), 'bin', 'node')); "
            "print(os.path.join(os.path.dirname(basedpyright.__file__), "
            "'langserver.index.js'))",
        ]
    ).split()
    commands = {
        "pyrefly": [str(pyrefly), "lsp"],
        "basedpyright": [node, langserver, "--stdio"],
    }
    check_memory_reader()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "environment.json").write_text(
        json.dumps(_environment(workdir, spec), indent=2) + "\n"
    )
    parameters = {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    warmup_args = argparse.Namespace(**{**vars(args), "steady_window": 10.0})
    failures: list[str] = []

    for project in _select_projects(spec, args.project):
        name = project["name"]
        checkout = workdir / "projects" / name
        python = str(_venv_bin(workdir / "venvs" / name, "python"))
        project_out = args.out / name
        project_out.mkdir(exist_ok=True)

        # One warmup run per configuration (label None), then interleaved
        # repetitions so slow drift in machine state affects all configurations.
        runs = [(None, config) for config in args.configs] + [
            (rep, config)
            for rep in range(1, args.repetitions + 1)
            for config in args.configs
        ]
        for rep, config in runs:
            label = "warmup" if rep is None else str(rep)
            print(f"{name}: {config} ({label})", flush=True)
            settings = lsp_settings(config, python)
            try:
                result = run_workload(
                    commands[CONFIGS[config][0]],
                    checkout,
                    settings,
                    project["workload"],
                    project_out / f"{config}-{label}.log",
                    warmup_args if rep is None else args,
                )
            except (TimeoutError, BrokenPipeError, RuntimeError) as e:
                # One hung or crashed server should not abandon the remaining runs.
                failures.append(f"{name}: {config} ({label}): {e}")
                print(f"  FAILED: {e}", flush=True)
                continue
            if rep is None:
                continue
            result.update(
                project=name,
                commit=project["commit"],
                config=config,
                repetition=rep,
                memory_metric=memory_metric(),
                settings=settings,
                parameters=parameters,
            )
            (project_out / f"{config}-{rep}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    print(f"Results written to {args.out}")
    if failures:
        sys.exit("Failed runs (see their .log files):\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


# Each summary table is (summary key, checkpoint field, title).
SUMMARY_TABLES = [
    ("steady", "mem_kb", "Steady value (last sample once memory stopped changing)"),
    ("peak", "peak_kb", "Peak during the step"),
]
# basedpyright logs this every time heap pressure makes it discard its type cache.
CACHE_CLEAR_MESSAGE = "Emptying type cache to avoid heap overflow"


def _stats(values: list[int]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_summary(results_dir: Path) -> dict[str, Any]:
    """Aggregate every run under `results_dir` into a median and range per checkpoint.

    The summary keeps no timelines, logs or local paths, so it is small enough to
    commit and safe to publish.
    """
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(results_dir.glob("*/*.json")):
        run = json.loads(path.read_text())
        log = path.with_suffix(".log")
        run["cache_clears"] = (
            log.read_text(errors="replace").count(CACHE_CLEAR_MESSAGE)
            if log.exists()
            else None
        )
        runs.setdefault(run["project"], {}).setdefault(run["config"], []).append(run)
    if not runs:
        sys.exit(f"error: no results found in {results_dir}")

    projects: dict[str, Any] = {}
    for project, configs in runs.items():
        summaries = {}
        for config, config_runs in configs.items():
            checkpoints = {}
            for name in CHECKPOINTS:
                points = [
                    c
                    for r in config_runs
                    for c in r["checkpoints"]
                    if c["name"] == name
                ]
                checkpoints[name] = {
                    key: _stats([c[field] for c in points])
                    for key, field, _ in SUMMARY_TABLES
                } | {"all_steady": all(c["steady"] for c in points)}
            config_summary: dict[str, Any] = {"repetitions": len(config_runs)}
            if CONFIGS[config][0] == "basedpyright":
                config_summary["cache_clears"] = [
                    r["cache_clears"] for r in config_runs
                ]
            config_summary["checkpoints"] = checkpoints
            summaries[config] = config_summary
        commit = next(iter(configs.values()))[0]["commit"]
        projects[project] = {"commit": commit, "configs": summaries}

    first = next(iter(next(iter(runs.values())).values()))[0]
    return {
        "memory_metric": first["memory_metric"],
        "unit": "KiB",
        "parameters": first["parameters"],
        "projects": projects,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"Memory metric: {summary['memory_metric']}, MiB, "
        "median (min-max) over repetitions\n"
    )
    for project, data in summary["projects"].items():
        print(f"## {project}\n")
        for key, _, title in SUMMARY_TABLES:
            print(f"{title}:\n")
            print("| config | " + " | ".join(CHECKPOINTS) + " |")
            print("|---" * (len(CHECKPOINTS) + 1) + "|")
            for config, config_data in data["configs"].items():
                cells = []
                for name in CHECKPOINTS:
                    point = config_data["checkpoints"][name]
                    s = point[key]
                    cell = (
                        f"{s['median'] / 1024:.0f} "
                        f"({s['min'] / 1024:.0f}-{s['max'] / 1024:.0f})"
                    )
                    cells.append(cell if point["all_steady"] else cell + " !")
                row = f"| {config} (n={config_data['repetitions']}) | "
                print(row + " | ".join(cells) + " |")
            print()
        clears = [
            f"{config}: "
            + ", ".join(
                "?" if c is None else str(c) for c in config_data["cache_clears"]
            )
            for config, config_data in data["configs"].items()
            if "cache_clears" in config_data
        ]
        if clears:
            print("basedpyright type cache clears per repetition: " + "; ".join(clears))
            print()
    print("! = at least one repetition timed out before memory was steady")
    print("? = the run's .log file is missing")


def summarize(args: argparse.Namespace) -> None:
    summary = build_summary(args.results)
    _print_summary(summary)
    if args.json is None:
        return
    environment_path = args.results / "environment.json"
    if not environment_path.exists():
        sys.exit(f"error: {environment_path} not found; it is written by `run`")
    environment = json.loads(environment_path.read_text())
    # Editable installs of the benchmarked project carry a local path; the project
    # is already identified by its commit.
    environment["freezes"] = {
        project: [line for line in lines if not line.startswith("-e ")]
        for project, lines in environment["freezes"].items()
    }
    args.json.write_text(
        json.dumps({"environment": environment} | summary, indent=2) + "\n"
    )
    print(f"\nSummary written to {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--projects-file", type=Path, default=DEFAULT_PROJECTS)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="clone projects, create venvs")
    setup_parser.add_argument("--project", nargs="+", help="default: all projects")

    run_parser = subparsers.add_parser("run", help="run the benchmark")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--project", nargs="+", help="default: all projects")
    run_parser.add_argument(
        "--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS)
    )
    run_parser.add_argument("--repetitions", type=int, default=3)
    run_parser.add_argument("--interval", type=float, default=1.0)
    run_parser.add_argument("--steady-window", type=float, default=60.0)
    run_parser.add_argument("--steady-tolerance", type=float, default=0.02)
    run_parser.add_argument("--steady-timeout", type=float, default=1800.0)
    run_parser.add_argument("--edit-delay", type=float, default=2.0)
    run_parser.add_argument("--request-timeout", type=float, default=600.0)

    summarize_parser = subparsers.add_parser("summarize", help="print a summary")
    summarize_parser.add_argument("results", type=Path)
    summarize_parser.add_argument(
        "--json",
        type=Path,
        help="also write the summary, with environment details, to this file",
    )

    args = parser.parse_args()
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        sys.exit("error: only macOS and Linux are supported")
    {"setup": setup, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
