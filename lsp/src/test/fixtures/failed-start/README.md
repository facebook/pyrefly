A workspace whose settings point Pyrefly at a binary that is not there, so the
extension's very first `client.start()` fails. The path is relative, so it
resolves against this folder on every platform.
