*Release date: September 03, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.3.0-dev.4 bundles **198 commits** from **23 contributors**.

---

## ✨ New & Improved

### Type Checking

- New composable type system primitives have been introduced for tensor shape types. See https://github.com/facebook/pyrefly/discussions/4807 for details.
- Overload selection now infers the most general return type assignable to all candidate return types when argument types are gradual.

### Language Server

- Parameter-name inlay hints now align correctly for instance methods, skipping `self` and displaying hints for all user-visible parameters.
- Incoming call hierarchy and find-references now preserve parameter references in unopened files by escalating only the necessary modules to full analysis.
- Semantic autocomplete now works correctly on lines preceding comments by handling CRLF-aware comment offsets.
- Go-to-definition for decorated functions now navigates to the original function definition instead of the decorator's `__call__` method.
- Workspace symbol search now includes class methods from project-indexed files, not just top-level exports.

### Tensor Shape Support

- JAX, NumPy, and Torch stubs have migrated from the V1 shape DSL to type-level V2 signatures, removing decorator-based evaluation in favor of direct DSL calls.

### Configuration

- The `replace-untyped-imports-with-any` flag and config option now replace imports from third-party packages missing stubs or `py.typed` markers with `Any`, matching mypy's `follow_untyped_imports` behavior.
- Mypy's `follow_untyped_imports` setting is now auto-migrated to the new `replace-untyped-imports-with-any` option.

---

## 🐛 Bug fixes

We closed **15** bug issues this release 👏

- **#4552:** Fixed an issue where semantic autocomplete failed on lines immediately above comments due to incorrect CRLF-aware offset calculations.
- **#4484:** Stopped reporting `__tracebackhide__` as an unused variable, since pytest reads it implicitly when formatting tracebacks.
- **#4611:** Fixed parameter-name inlay hints being shifted by one position for instance-method calls, now correctly skipping `self` and aligning hints with actual parameters.
- **#4616:** Allowed plain subclasses of slotted dataclasses to assign new attributes without triggering false `missing-attribute` errors, since slot enforcement no longer applies to subclasses that don't declare slots themselves.
- **#4626:** Fixed bounded `type[T]` calls with custom metaclass returns to only substitute the constructor result with `T` when the inferred return is compatible with the bounded class, preserving unrelated metaclass `__call__` return types.
- **#3265:** Fixed `iter([0])` to infer `Iterator[int]` instead of `Iterator[Any]` by no longer using the generic `Any` bound from `iter`'s signature to infer list element types.
- **#4563:** Made unresolvable `__all__` state sticky across later mutations, so `append`, `extend`, and `remove` calls preserve `DunderAllKind::Unresolvable` and star imports continue using fallback behavior.
- **#3569:** Improved decorator error messages to detect when a decorated function is missing an injected parameter and report a clearer message instead of a confusing callable type mismatch.
- **#1493:** Fixed confusing override errors when overriding an overloaded method by improving error messages to show all overload signatures instead of suggesting conflicting parameter names.
- **#4686:** Fixed `type(protocol_value)` to produce `type[Protocol]` representing the unknown concrete runtime implementation, preventing false `invalid-argument` errors when used with `isinstance`.
- And more! #4676, #4684, #263, #3812, #3993

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.0-dev.4
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).

---

## 🖊️ Contributors this release

@stroxler, @rchen152, @samwgoldman, @grievejia, @jakevdp, @asukaminato0721, @kinto0, David Tolnay, @connernilsen, @ndmitchell, @DarkNightForge, @cakeni, @nitishagar, @shobhitmehro, @Pager-dot, @tkim602, @NathanTempest, @anishfyi, @mangeshraut712, @jorenham, @alexander-beedie, @Pyxelate, @yangdanny97

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
