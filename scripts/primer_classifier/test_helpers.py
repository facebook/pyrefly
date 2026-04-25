# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test helpers, scenario builders, and test data for primer classifier tests.

This module contains:
- Common ErrorEntry / ClassificationResult factory functions
- Synthetic scenario builders (variance, TYPE_CHECKING, override, etc.)
- Ground-truth scenario builders (real data from pyrefly git history)
- Synthetic and real diff strings
- Assertion helpers
"""

from __future__ import annotations

import re
from pathlib import Path

from .classifier import (
    Classification,
    ClassificationResult,
    Suggestion,
)
from .llm_client import CategoryVerdict
from .parser import ErrorEntry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "unit"


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Common factory helpers
# ---------------------------------------------------------------------------


def make_error_entry(
    kind: str = "bad-return",
    msg: str = "msg",
    file_path: str = "src/foo.py",
) -> ErrorEntry:
    """Create an ErrorEntry for use in test scenarios."""
    return ErrorEntry(
        severity="ERROR",
        file_path=file_path,
        location="10:5-20",
        message=msg,
        error_kind=kind,
        raw_line=f"ERROR {file_path}:10:5-20: {msg} [{kind}]",
    )


def build_classification_result(
    n_projects: int,
    verdict: str,
    error_kind: str,
    reason: str,
    attribution: str,
    project_prefix: str = "project",
) -> ClassificationResult:
    """Build a ClassificationResult with N identical project results."""
    classifications = []
    for i in range(n_projects):
        classifications.append(
            Classification(
                project_name=f"{project_prefix}_{i}",
                verdict=verdict,
                reason=reason,
                added_count=5 if verdict == "regression" else 0,
                removed_count=0 if verdict == "regression" else 5,
                method="llm",
                pr_attribution=attribution,
                categories=[
                    CategoryVerdict(error_kind, verdict, reason),
                ],
            )
        )
    regressions = n_projects if verdict == "regression" else 0
    improvements = n_projects if verdict == "improvement" else 0
    neutrals = n_projects if verdict == "neutral" else 0
    ambiguous = n_projects if verdict == "ambiguous" else 0
    return ClassificationResult(
        classifications=classifications,
        total_projects=n_projects,
        regressions=regressions,
        improvements=improvements,
        neutrals=neutrals,
        ambiguous=ambiguous,
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_actionable(suggestion: Suggestion) -> None:
    """Verify a suggestion names a function, lists projects, and lists error kinds."""
    assert re.search(r"\w+[\(_]", suggestion.description), (
        f"Must name a function in description, got: {suggestion.description}"
    )
    assert suggestion.affected_projects, (
        f"Must list affected projects, got: {suggestion.affected_projects}"
    )
    assert suggestion.error_kinds_fixed, (
        f"Must list error kinds, got: {suggestion.error_kinds_fixed}"
    )


# ---------------------------------------------------------------------------
# Synthetic scenario builders
# ---------------------------------------------------------------------------


def build_variance_scenario() -> ClassificationResult:
    """5 projects with variance-mismatch regressions on regular classes."""
    return build_classification_result(
        n_projects=5,
        verdict="regression",
        error_kind="variance-mismatch",
        reason="Variance check applied to regular classes, not just protocols. "
        "Per the typing spec, variance inference is only required for protocols.",
        attribution="Removed is_protocol() guard in alt/class/variance.rs",
        project_prefix="variance_proj",
    )


VARIANCE_DIFF = """\
diff --git a/pyrefly/lib/alt/class/variance.rs b/pyrefly/lib/alt/class/variance.rs
--- a/pyrefly/lib/alt/class/variance.rs
+++ b/pyrefly/lib/alt/class/variance.rs
@@ -120,8 +120,6 @@ fn check_variance(&self, cls: &ClassType) {
-        if !cls.is_protocol() {
-            return;
-        }
         for param in cls.type_params() {
             self.check_param_variance(param, cls);
         }
"""


def build_type_checking_scenario() -> ClassificationResult:
    """4 projects with TYPE_CHECKING bad-assignment regressions."""
    projects = ["urllib3", "trio", "zulip", "ibis"]
    classifications = []
    for name in projects:
        classifications.append(
            Classification(
                project_name=name,
                verdict="regression",
                reason="Cannot assign to TYPE_CHECKING because imported as final. "
                "This is a well-known pattern that mypy/pyright exempt.",
                added_count=3,
                removed_count=0,
                method="llm",
                pr_attribution="Stricter final-variable checking in binding/stmt.rs",
                categories=[
                    CategoryVerdict(
                        "bad-assignment",
                        "regression",
                        "TYPE_CHECKING reassignment is a standard pattern",
                    ),
                ],
            )
        )
    return ClassificationResult(
        classifications=classifications,
        total_projects=len(projects),
        regressions=len(projects),
    )


TYPE_CHECKING_DIFF = """\
diff --git a/pyrefly/lib/binding/stmt.rs b/pyrefly/lib/binding/stmt.rs
--- a/pyrefly/lib/binding/stmt.rs
+++ b/pyrefly/lib/binding/stmt.rs
@@ -200,6 +200,10 @@ fn check_assignment(&mut self, target: &Expr, value: &Expr) {
+        // Check for final variable reassignment
+        if self.is_final_variable(target) {
+            self.error(target.range(), "Cannot assign to final variable");
+        }
"""


def build_override_scenario() -> ClassificationResult:
    """4 projects with bad-override regressions."""
    projects = ["jax", "bokeh", "poetry", "artigraph"]
    classifications = []
    for name in projects:
        classifications.append(
            Classification(
                project_name=name,
                verdict="regression",
                reason="Override check is too strict, flagging methods that are "
                "compatible by Liskov substitution principle.",
                added_count=8,
                removed_count=0,
                method="llm",
                pr_attribution="Stricter override checking in alt/class/override.rs",
                categories=[
                    CategoryVerdict(
                        "bad-override",
                        "regression",
                        "Override check scope is too broad",
                    ),
                ],
            )
        )
    return ClassificationResult(
        classifications=classifications,
        total_projects=len(projects),
        regressions=len(projects),
    )


def build_all_improvements_scenario() -> ClassificationResult:
    """3 projects with only improvements (no regressions)."""
    return build_classification_result(
        n_projects=3,
        verdict="improvement",
        error_kind="missing-attribute",
        reason="Removed false positive missing-attribute errors. "
        "Attributes exist via inheritance from parent class.",
        attribution="Improved class hierarchy resolution in alt/class/resolve.rs",
        project_prefix="improved_proj",
    )


# ---------------------------------------------------------------------------
# Ground-truth scenario builders (real data from pyrefly git history)
# ---------------------------------------------------------------------------


def build_gt_protocol_subtyping_scenario() -> ClassificationResult:
    """Ground-truth Scenario A: PR #2428 -> #2492.

    Regression in protocol subtyping: calculate_abstract_members() in class_metadata.rs
    used class_body_fields() instead of fields(), which excluded synthesized fields
    like __dataclass_fields__. The fix (PR #2492) added a guard using get_class_member()
    to detect synthesized concrete implementations.

    Real data from fixtures/real/pr_2428.txt.
    """
    classifications = [
        Classification(
            project_name="jax",
            verdict="regression",
            reason="bad-override errors on BarrierType.type and ClusterBarrierType.type "
            "overriding ExtendedDType — these are protocol subtyping false positives "
            "caused by synthesized field exclusion in calculate_abstract_members().",
            added_count=2,
            removed_count=0,
            method="llm",
            pr_attribution="Change in calculate_abstract_members() in class_metadata.rs "
            "switched from fields() to class_body_fields(), excluding synthesized fields",
            categories=[
                CategoryVerdict("bad-override", "regression", "Protocol subtyping false positives"),
            ],
        ),
        Classification(
            project_name="bokeh",
            verdict="regression",
            reason="bad-override errors on Alias.readonly and Alias.serialized overriding "
            "Property — false positives from same protocol subtyping regression.",
            added_count=2,
            removed_count=0,
            method="llm",
            pr_attribution="Same calculate_abstract_members() change in class_metadata.rs",
            categories=[
                CategoryVerdict("bad-override", "regression", "Protocol subtyping false positives"),
            ],
        ),
        Classification(
            project_name="poetry",
            verdict="regression",
            reason="bad-override errors on plugin command overrides — false positives "
            "from protocol subtyping regression.",
            added_count=3,
            removed_count=0,
            method="llm",
            pr_attribution="calculate_abstract_members() in class_metadata.rs",
            categories=[
                CategoryVerdict("bad-override", "regression", "Protocol subtyping false positives"),
            ],
        ),
        Classification(
            project_name="artigraph",
            verdict="regression",
            reason="bad-override errors on Producer subclass overrides — false positives.",
            added_count=2,
            removed_count=0,
            method="llm",
            pr_attribution="calculate_abstract_members() in class_metadata.rs",
            categories=[
                CategoryVerdict("bad-override", "regression", "Protocol subtyping false positives"),
            ],
        ),
        Classification(
            project_name="hydra-zen",
            verdict="improvement",
            reason="reveal-type changed from @_ (unresolved) to concrete types "
            "(A, int, str, type[str], etc.) — type inference improved.",
            added_count=22,
            removed_count=30,
            method="llm",
            pr_attribution="Improved protocol handling resolved previously unresolved types",
            categories=[
                CategoryVerdict("reveal-type", "improvement", "Type inference improved from @_ to concrete"),
            ],
        ),
    ]
    return ClassificationResult(
        classifications=classifications,
        total_projects=5,
        regressions=4,
        improvements=1,
    )


GT_PROTOCOL_SUBTYPING_DIFF = """\
diff --git a/pyrefly/lib/alt/class/class_metadata.rs b/pyrefly/lib/alt/class/class_metadata.rs
--- a/pyrefly/lib/alt/class/class_metadata.rs
+++ b/pyrefly/lib/alt/class/class_metadata.rs
@@ -395,7 +395,7 @@ impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
             )
         }) {
             Some(ProtocolMetadata {
-                members: cls.fields().cloned().collect(),
+                members: cls.class_body_fields().cloned().collect(),
                 is_runtime_checkable: false,
             })
         } else {
@@ -1339,6 +1339,8 @@ impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
         let mut abstract_members = SmallSet::new();
         for field_name in fields_to_check {
+            // Use get_non_synthesized_class_member to find abstract requirements
             if let Some(field) =
                 self.get_non_synthesized_class_member_and_defining_class(cls, &field_name)
                 && (field.value.is_abstract() ||
"""


def build_gt_type_checking_scenario() -> ClassificationResult:
    """Ground-truth Scenario B: PR #2389 TYPE_CHECKING (UNRESOLVED).

    Regression from check_for_imported_final_reassignment() in bindings.rs.
    No fix exists yet, but we know the function that needs changing.
    It doesn't exempt TYPE_CHECKING from the final reassignment check.

    Real data from fixtures/real/pr_2389.txt.
    """
    classifications = [
        Classification(
            project_name="urllib3",
            verdict="regression",
            reason="Cannot assign to HAS_NEVER_CHECK_COMMON_NAME because imported as final — "
            "false positive, this is a feature-detection reassignment in a try/except block.",
            added_count=1,
            removed_count=0,
            method="llm",
            pr_attribution="check_for_imported_final_reassignment() in bindings.rs "
            "does not exempt feature-detection patterns in try/except blocks",
            categories=[
                CategoryVerdict("bad-assignment", "regression", "Final reassignment false positive"),
            ],
        ),
        Classification(
            project_name="trio",
            verdict="regression",
            reason="Cannot assign to TYPE_CHECKING because imported as final — "
            "false positive, TYPE_CHECKING = True is a standard Python pattern.",
            added_count=1,
            removed_count=0,
            method="llm",
            pr_attribution="check_for_imported_final_reassignment() in bindings.rs "
            "does not exempt TYPE_CHECKING reassignment",
            categories=[
                CategoryVerdict("bad-assignment", "regression", "TYPE_CHECKING reassignment FP"),
            ],
        ),
        Classification(
            project_name="zulip",
            verdict="regression",
            reason="Cannot assign to TYPE_CHECKING because imported as final — "
            "same TYPE_CHECKING false positive.",
            added_count=1,
            removed_count=0,
            method="llm",
            pr_attribution="check_for_imported_final_reassignment() in bindings.rs",
            categories=[
                CategoryVerdict("bad-assignment", "regression", "TYPE_CHECKING reassignment FP"),
            ],
        ),
        Classification(
            project_name="ibis",
            verdict="regression",
            reason="Cannot assign to TYPE_CHECKING because imported as final — "
            "14 false positives across ibis/expr/ files.",
            added_count=14,
            removed_count=0,
            method="llm",
            pr_attribution="check_for_imported_final_reassignment() in bindings.rs",
            categories=[
                CategoryVerdict("bad-assignment", "regression", "TYPE_CHECKING reassignment FP"),
            ],
        ),
    ]
    return ClassificationResult(
        classifications=classifications,
        total_projects=4,
        regressions=4,
    )


GT_TYPE_CHECKING_DIFF = """\
diff --git a/pyrefly/lib/binding/bindings.rs b/pyrefly/lib/binding/bindings.rs
--- a/pyrefly/lib/binding/bindings.rs
+++ b/pyrefly/lib/binding/bindings.rs
@@ -1335,6 +1335,7 @@ impl<'a> BindingsBuilder<'a> {
         style: FlowStyle,
     ) -> Option<Idx<KeyAnnotation>> {
         self.check_for_type_alias_redefinition(name, idx);
+        self.check_for_imported_final_reassignment(name, idx);
         let name = Hashed::new(name);

@@ -1374,6 +1375,20 @@ impl<'a> BindingsBuilder<'a> {
         }
     }

+    fn check_for_imported_final_reassignment(&self, name: &Name, idx: Idx<Key>) {
+        let prev_idx = self.scopes.current_flow_idx(name);
+        if let Some(prev_idx) = prev_idx
+            && let Some(Binding::Import(module, original_name, _)) = self.idx_to_binding(prev_idx)
+            && self.lookup.is_final(*module, original_name)
+        {
+            self.error(
+                self.idx_to_key(idx).range(),
+                ErrorInfo::Kind(ErrorKind::BadAssignment),
+                format!("Cannot assign to `{name}` because it is imported as final"),
+            );
+        }
+    }
"""


def build_gt_bad_override_args_scenario() -> ClassificationResult:
    """Ground-truth Scenario C: Bad-override *args/**kwargs (CONFIRMED FIX via #2322).

    Fix was has_any_args_and_kwargs() in solver/subset.rs to treat
    *args: Any, **kwargs: Any as compatible with any signature.

    Uses the same projects from pr_2428.txt (bad-override regressions) since
    the override checking tightening affected the same projects.
    """
    classifications = [
        Classification(
            project_name="jax",
            verdict="regression",
            reason="bad-override errors where child methods have *args: Any, **kwargs: Any "
            "but parent has specific parameters — should be treated as compatible.",
            added_count=2,
            removed_count=0,
            method="llm",
            pr_attribution="Stricter override checking in solver/subset.rs without "
            "*args: Any, **kwargs: Any escape hatch",
            categories=[
                CategoryVerdict("bad-override", "regression", "*args/**kwargs override FP"),
            ],
        ),
        Classification(
            project_name="bokeh",
            verdict="regression",
            reason="bad-override errors where overrides use *args: Any, **kwargs: Any pattern.",
            added_count=2,
            removed_count=0,
            method="llm",
            pr_attribution="solver/subset.rs override checking too strict",
            categories=[
                CategoryVerdict("bad-override", "regression", "*args/**kwargs override FP"),
            ],
        ),
        Classification(
            project_name="poetry",
            verdict="regression",
            reason="bad-override errors on plugin command overrides with *args/**kwargs.",
            added_count=3,
            removed_count=0,
            method="llm",
            pr_attribution="solver/subset.rs override checking",
            categories=[
                CategoryVerdict("bad-override", "regression", "*args/**kwargs override FP"),
            ],
        ),
    ]
    return ClassificationResult(
        classifications=classifications,
        total_projects=3,
        regressions=3,
    )


GT_BAD_OVERRIDE_ARGS_DIFF = """\
diff --git a/pyrefly/lib/solver/subset.rs b/pyrefly/lib/solver/subset.rs
--- a/pyrefly/lib/solver/subset.rs
+++ b/pyrefly/lib/solver/subset.rs
@@ -108,6 +108,15 @@ impl<'a, Ans: LookupAnswer> Subset<'a, Ans> {
     fn is_subset_param_list(
         &mut self,
         l_args: &[Param],
         u_args: &[Param],
     ) -> Result<(), SubsetError> {
         let mut l_args = l_args.iter();
         let mut u_args = u_args.iter();
+        // Strict parameter-by-parameter comparison without special-casing
+        // *args: Any, **kwargs: Any. This means overrides with *args: Any,
+        // **kwargs: Any are checked against the full parent signature.
"""


def build_gt_pure_improvement_scenario() -> ClassificationResult:
    """Ground-truth Scenario D: Pure improvement — no suggestion needed.

    One project (hydra-zen) with reveal-type improvements from @_ to concrete types.
    No regressions at all.
    """
    classifications = [
        Classification(
            project_name="hydra-zen",
            verdict="improvement",
            reason="22 reveal-type results changed from @_ (unresolved) to concrete types. "
            "Type inference improved across the board.",
            added_count=22,
            removed_count=30,
            method="llm",
            pr_attribution="Protocol handling improvements resolved previously unresolved types",
            categories=[
                CategoryVerdict("reveal-type", "improvement", "Type inference improved"),
            ],
        ),
    ]
    return ClassificationResult(
        classifications=classifications,
        total_projects=1,
        improvements=1,
    )


def build_gt_all_neutral_scenario() -> ClassificationResult:
    """Ground-truth Scenario E: All neutral (wording changes only)."""
    classifications = [
        Classification(
            project_name="myproject",
            verdict="neutral",
            reason="Message wording changes only, same errors at same locations.",
            added_count=5,
            removed_count=5,
            method="heuristic",
        ),
    ]
    return ClassificationResult(
        classifications=classifications,
        total_projects=1,
        neutrals=1,
    )


# ---------------------------------------------------------------------------
# Mock API responses (used by deterministic tests)
# ---------------------------------------------------------------------------


MOCK_VARIANCE_SUGGESTION_RESPONSE = {
    "summary": "Variance check needs protocol guard",
    "suggestions": [
        {
            "description": "Restore is_protocol() guard in variance checking",
            "files": ["pyrefly/lib/alt/class/variance.rs"],
            "confidence": "high",
            "reasoning": "The guard was removed, applying variance checks too broadly",
        },
    ],
}

MOCK_GT_SCENARIO_A_RESPONSE = {
    "summary": "Protocol subtyping regression from calculate_abstract_members() "
    "excluding synthesized fields like __dataclass_fields__",
    "suggestions": [
        {
            "description": "In calculate_abstract_members() in "
            "pyrefly/lib/alt/class/class_metadata.rs, add a guard using "
            "get_class_member() to check for synthesized concrete implementations "
            "before falling through to get_non_synthesized_class_member_and_defining_class()",
            "files": ["pyrefly/lib/alt/class/class_metadata.rs"],
            "confidence": "high",
            "reasoning": "The change from fields() to class_body_fields() excluded "
            "synthesized fields like __dataclass_fields__. Adding a pre-check with "
            "get_class_member() eliminates 9 bad-override errors across 4 projects.",
            "affected_projects": ["jax", "bokeh", "poetry", "artigraph"],
            "error_kinds_fixed": ["bad-override"],
        },
    ],
}

MOCK_GT_SCENARIO_B_RESPONSE = {
    "summary": "TYPE_CHECKING reassignment regression from "
    "check_for_imported_final_reassignment() not exempting TYPE_CHECKING",
    "suggestions": [
        {
            "description": "In check_for_imported_final_reassignment() in "
            "pyrefly/lib/binding/bindings.rs, add an exemption: if the name "
            "being reassigned is TYPE_CHECKING, skip the final check",
            "files": ["pyrefly/lib/binding/bindings.rs"],
            "confidence": "high",
            "reasoning": "TYPE_CHECKING = True is a standard Python pattern. "
            "This eliminates 17 bad-assignment errors across 4 projects.",
            "affected_projects": ["urllib3", "trio", "zulip", "ibis"],
            "error_kinds_fixed": ["bad-assignment"],
        },
    ],
}

MOCK_VARIANCE_SCENARIO_RESPONSE = {
    "summary": "Restore is_protocol() guard to limit variance checking to protocols",
    "suggestions": [
        {
            "description": "Restore the is_protocol() guard in check_variance()",
            "files": ["pyrefly/lib/alt/class/variance.rs"],
            "confidence": "high",
            "reasoning": "Variance inference per the typing spec is only required "
            "for protocols. The PR removed the guard, causing variance checks on "
            "all classes.",
        },
    ],
}

MOCK_TYPE_CHECKING_SCENARIO_RESPONSE = {
    "summary": "Exempt TYPE_CHECKING from final variable reassignment check",
    "suggestions": [
        {
            "description": "Exempt TYPE_CHECKING reassignment from final check",
            "files": ["pyrefly/lib/binding/stmt.rs"],
            "confidence": "high",
            "reasoning": "TYPE_CHECKING = True is a standard Python pattern. "
            "mypy and pyright explicitly allow it.",
        },
    ],
}

MOCK_OVERRIDE_SCENARIO_RESPONSE = {
    "summary": "Narrow override check scope",
    "suggestions": [
        {
            "description": "Narrow the override check to exclude compatible overrides",
            "files": ["pyrefly/lib/alt/class/override.rs"],
            "confidence": "medium",
            "reasoning": "The override check is flagging methods that are "
            "compatible by Liskov substitution.",
        },
    ],
}

MOCK_MIXED_SCENARIO_RESPONSE = {
    "summary": "Variance check too broad",
    "suggestions": [
        {
            "description": "Fix variance check scope",
            "files": ["variance.rs"],
            "confidence": "high",
            "reasoning": "Only the variance regressions need fixing",
        },
    ],
}
