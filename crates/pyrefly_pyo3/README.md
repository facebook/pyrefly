# pyrefly_api

In-process Python bindings for the pyrefly

```python
import pyrefly_api

checker = pyrefly_api.Checker(project_root="/path/to/repo")
for diag in checker.check("x: str = 1"):
    print(diag.kind, diag.message, diag.line, diag.column)
```