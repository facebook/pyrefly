# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Turn the logs that `record.sh` captured into one trace file per feature.

Pylance sends foreground requests (hover, completion) on pyrefly's main
connection, which `record.sh` captures from stdin, and background-checker
requests on an extra connection, which only Pylance's own log
(`python.analysis.logTypeServerMessages`) shows; the runner copies that log to
`pylance.log` before VS Code rotates it.

Usage: extract.py <record.sh work dir> <PyTorch checkout> <output dir>
"""

import json
import os
import re
import sys
from datetime import datetime

GET_COMPUTED_TYPE = "typeServer/getComputedType"
GET_SNAPSHOT = "typeServer/getSnapshot"


def phases(work):
    """The `(start ms, name)` of each phase that the runner marked."""
    with open(f"{work}/pyrefly.log", errors="replace") as f:
        return [
            (int(m.group(2)), m.group(1))
            for m in re.finditer(r"=== PHASE (\S+) (\d+)", f.read())
        ]


def foreground(work):
    """Requests on the main connection, by phase."""
    data = open(f"{work}/main_in.log", "rb").read()
    by_phase = {}
    phase = "startup"
    pos = 0
    pattern = re.compile(rb"\n=== PHASE (\S+) \d+\n|Content-Length: (\d+)\r\n\r\n")
    while m := pattern.search(data, pos):
        if m.group(1):
            phase = m.group(1).decode()
            pos = m.end()
            continue
        end = m.end() + int(m.group(2))
        message = json.loads(data[m.end() : end])
        pos = end
        if message.get("method", "").startswith("typeServer/"):
            by_phase.setdefault(phase, []).append(
                (message["method"], message.get("params"))
            )
    return by_phase


def background(log, marks):
    """Background-checker requests from Pylance's log, by phase."""
    requests = []
    with open(log, errors="replace") as f:
        for line in f:
            m = re.match(
                r"(\S+ \S+) \[info\] \(\d+\): BG\([^)]*\): TS: \[request -->\] (\S+) (\[.*)",
                line,
            )
            if m:
                time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                # The log shows `[params, cancellation token]`, or just the token
                # for a request without params.
                params = (
                    None if m.group(2) == GET_SNAPSHOT else json.loads(m.group(3))[0]
                )
                requests.append((time.timestamp() * 1000, m.group(2), params))
    requests.sort(key=lambda request: request[0])
    by_phase = {}
    for time, method, params in requests:
        phase = max(
            ((start, name) for start, name in marks if start <= time),
            default=(0, "startup"),
        )[1]
        by_phase.setdefault(phase, []).append((method, params))
    return by_phase


def main():
    work, pytorch, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    marks = phases(work)
    fg = foreground(work)
    bg = background(f"{work}/pylance.log", marks)
    # `typeServer/connection` opens Pylance's extra connection, which the
    # benchmark's in-memory client does not need.
    traces = {
        "cold_start": [
            request
            for request in fg.get("startup", []) + bg.get("startup", [])
            if request[0] != "typeServer/connection"
        ],
        "hover": fg.get("hover_edit", []) + fg.get("hover", []),
        "completion": fg.get("completion_edit", []) + fg.get("completion", []),
        "recheck": bg.get("recheck_edit", []) + bg.get("recheck", []),
    }

    # Absolute paths become placeholders that the benchmark fills in.
    roots = {f"file://{pytorch}", f"file://{os.path.realpath(pytorch)}"}

    def portable(params):
        if params is None:
            return None
        params.pop("snapshot", None)
        text = json.dumps(params, separators=(",", ":"))
        for root in roots:
            text = text.replace(root, "${ROOT}")
        text = re.sub(
            r"file:///tmp/pyrefly_bundled_typeshed_[0-9a-f]+", "${TYPESHED}", text
        )
        text = re.sub(
            r"file:///tmp/pyrefly_bundled_third_party_[0-9a-f]+", "${THIRD_PARTY}", text
        )
        return json.loads(text)

    for name, trace in traces.items():
        # `getComputedType` of a node is most of every trace, so it is stored as
        # `[uri index, start line, start character, end line, end character]`.
        uris = []
        requests = []
        for method, params in trace:
            params = portable(params)
            arg = (params or {}).get("arg", {})
            if method == GET_COMPUTED_TYPE and set(arg) == {"uri", "range"}:
                if arg["uri"] not in uris:
                    uris.append(arg["uri"])
                start, end = arg["range"]["start"], arg["range"]["end"]
                requests.append(
                    [
                        uris.index(arg["uri"]),
                        start["line"],
                        start["character"],
                        end["line"],
                        end["character"],
                    ]
                )
            else:
                requests.append({"method": method, "params": params})
        print(f"{name}: {len(requests)} requests")
        text = json.dumps({"uris": uris, "requests": requests}, separators=(",", ":"))
        # One request per line keeps diffs of a re-recording readable.
        with open(f"{out_dir}/{name}.json", "w") as f:
            f.write(re.sub(r"([\]}]),(?=[\[{])", r"\1,\n", text) + "\n")


if __name__ == "__main__":
    main()
