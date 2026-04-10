# {{PROJECT_NAME}} {{VERSION}}
**Status : {{STATUS}}**
*Release date: {{RELEASE_DATE}}*

{{PROJECT_NAME}} {{VERSION}} bundles **{{COMMIT_COUNT}} commits** from **{{CONTRIBUTOR_COUNT}} contributors**.

---

## ✨ New & Improved

| Area | What’s new |
|------|------------|
| **Type Checking** | - enhancement for area 1. <br><br>- enhancement for area 1. <br><br>- enhancement for area 1 |
| **Language Server** | - enhancement for area 2. <br><br>- enhancement for area 2. <br><br>- enhancement for area 2 |
| **{{AREA_3}}** | - enhancement for area 3. <br><br>- enhancement for area 3. <br><br>- enhancement for area 3 |
| **{{AREA_4}}** | - enhancement for area 4. <br><br>- enhancement for area 4. <br><br>- enhancement for area 4 |
| **{{Performance}}** | - enhancement for area 4. <br><br>- enhancement for area 4. <br><br>- enhancement for area 4 |

---

## 🐛 bug fixes

We closed {{NUMBER_OF_ISSUES}} bug issues this release 👏

{{BUG_FIXES_LIST}}
- {{#GITHUB_ISSUE_NUMBER}}{{DESCRIBE_BUG_FIX_1}}
-{{#GITHUB_ISSUE_NUMBER}}{{DESCRIBE_BUG_FIX_2}}
- {{#GITHUB_ISSUE_NUMBER}}{{DESCRIBE_BUG_FIX_3}}
- And more! {{#GITHUB_ISSUE_NUMBER}}, {{#GITHUB_ISSUE_NUMBER}}, {{#GITHUB_ISSUE_NUMBER}}


Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade {{PYPI_PACKAGE_NAME}}=={{VERSION}}
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1\. `pyrefly check --suppress-errors`
2\. run your code formatter of choice
3\. `pyrefly check --remove-unused-ignores`
4\. Repeat until you achieve a clean formatting run and a clean type check.

This will add  `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the \[Pyefly documentation\](https://pyrefly.org/en/docs/error-suppressions/)

---

## 🖊️ Contributors this release
{{CONTRIBUTOR_GITHUB_HANDLES}}

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
