*Release date: September 16, 2026*

Pyrefly v1.2.1 is a patch release with a single performance improvement.

---

## ⚡ Performance

- TSP requests on a file that is not open (for example `getComputedType`) used to solve the whole module from scratch in a throw-away transaction, then discard that work once the request finished. The solve is now committed to shared state when possible, so later requests for the same file reuse the earlier solve instead of repeating it. In a captured Pylance session with 43,811 `getComputedType` requests, median latency for non-open-file requests dropped from 16.9ms to 0.07ms, and total TSP request time for that session fell from about 671s to about 6.4s.

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.2.1
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).
