# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

global build_system


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = """
        <html>
            <head>
                <style>
                    body {
                        font-family: monospace;
                    }
                    #editor {
                        width: 100%;
                        height: 300px;
                        padding: 10px;
                        font-size: 16px;
                        font-family: monospace;
                        border: 1px solid #ccc;
                        resize: vertical;
                        overflow-y: auto;
                    }
                    #output {
                        padding: 10px;
                        background-color: #f0f0f0;
                        border: 1px solid #ccc;
                    }
                </style>
            </head>
            <body>
                <div style="text-align: center;">
                    <img src="https://raw.githubusercontent.com/facebook/pyre-check/main/logo.png" alt="Pyre2 Logo">
                    <h1>Pyre2 Sandbox</h1>
                </div>
                <div id="editor"></div>
                <button onclick="checkCode()">Check</button>
                <div id="output"></div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"></script>
                <script>
                    var editor = ace.edit("editor");
                    editor.setTheme("ace/theme/monokai");
                    editor.getSession().setMode("ace/mode/python");

                    var currentCode = '';
                    // Load the code from the URL if it exists
                    var url = new URL(window.location.href);
                    var codeParam = url.searchParams.get('code');
                    if (codeParam) {
                        editor.setValue(decodeURIComponent(codeParam));
                        currentCode = codeParam;
                    }
                    editor.on('change', function() {
                        var codeValue = editor.getValue();
                        currentCode = codeValue;
                    });
                    function checkCode() {
                        var url = new URL(window.location.href);
                        url.searchParams.set('code', encodeURIComponent(currentCode));
                        window.history.pushState({}, '', url.href);
                        fetch('/check', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ code: currentCode })
                        })
                        .then(response => response.text())
                        .then(output => {
                            document.getElementById("output").innerHTML = output;
                        });
                    }
                </script>
            </body>
        </html>
        """
        self.send_response(200)
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        self.tmp_dir = tempfile.mkdtemp()
        try:
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            code = data["code"]

            with open(os.path.join(self.tmp_dir, "sandbox.py"), "w") as f:
                f.write(code)
            result = None
            if build_system == "cargo":
                # Run the cargo command
                result = subprocess.run(
                    [
                        "cargo",
                        "run",
                        "--",
                        "check",
                        os.path.join(self.tmp_dir, "sandbox.py"),
                        "-o",
                        os.path.join(self.tmp_dir, "output"),
                    ],
                    capture_output=True,
                    text=True,
                    cwd="pyre2",
                )
            elif build_system == "buck":
                result = subprocess.run(
                    [
                        "buck2",
                        "run",
                        "pyre2",
                        "--",
                        "check",
                        os.path.join(self.tmp_dir, "sandbox.py"),
                        "-o",
                        os.path.join(self.tmp_dir, "output"),
                    ],
                    capture_output=True,
                    text=True,
                )
            print(result)
            if result.returncode != 0:
                output = result.stdout + result.stderr
                html_output = f"<pre><code class='language-bash'>{output}</code></pre>"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(html_output.encode("utf-8"))
            else:
                with open(os.path.join(self.tmp_dir, "output"), "r") as f:
                    output = f.read()
                    if not output.strip():
                        html_output = "0 errors"
                    else:
                        html_output = (
                            f"<pre><code class='language-python'>{output}</code></pre>"
                        )
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(html_output.encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))
        finally:
            shutil.rmtree(self.tmp_dir)


def run_server(build_system):
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Server running on port 8000...")
    httpd.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["cargo", "buck"]:
        print("Usage: python server.py <build_system>")
        sys.exit(1)
    build_system = sys.argv[1]
    run_server(build_system)
