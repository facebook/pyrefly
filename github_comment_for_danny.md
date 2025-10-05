# Update on PR #1185 - All Feedback Addressed, Need Guidance on Metadata Issue

Hey @yangdanny97,

I've addressed all your feedback from the PR:
- **Implemented your suggestion to store `is_abstract` flag in ClassField** (similar to your `is_deprecated` in 66ecc58)

The implementation is complete, but I'm hitting a metadata propagation issue:

The flag gets set properly during decorator processing in [`function.rs:437`](https://github.com/facebook/pyrefly/blob/abstract_final_v2/pyrefly/lib/alt/function.rs#L437) (I can see it in debug output), but by the time we check it in [`class_field.rs:1386-1391`](https://github.com/facebook/pyrefly/blob/abstract_final_v2/pyrefly/lib/alt/class/class_field.rs#L1386-L1391) after `deep_force`, it's always false.

I suspect this is why the original implementation had that complex check directly in the class field creation logic - to work around this metadata propagation issue. The type seems to lose its FuncMetadata when going through `solve_binding` → `deep_force`.

The core issue: When a method with `@abstractmethod` is processed, the `is_abstract_method` flag is set in the FuncFlags during decorator processing, but this metadata doesn't survive to the final Type that gets stored in ClassField. The type resolution pipeline (`solve_binding` → `deep_force`) seems to reconstruct the Type without preserving the FuncMetadata.

Should I:
1. **Fix the root cause** - Ensure FuncMetadata is preserved through the type resolution pipeline (might affect other metadata like `is_deprecated` too), or
2. **Work around it** - Move the abstract method detection logic directly into class field creation, checking decorators there instead of relying on FuncMetadata

Which approach aligns better with the codebase architecture? Happy to dig deeper into either path with your guidance.

Thanks!