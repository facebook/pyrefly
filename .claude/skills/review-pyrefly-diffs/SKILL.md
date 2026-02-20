---
name: review-pyrefly-diff
description: Reviews a comma separated pyrefly diff according to the pyrefly review best practices.
---

This skill is intended to be used by interal Meta employees to review code changes in phabricator. This should not be used to review Github pull requests.

You will take in a comma separated list of Phabricator Diff numbers and review them according to the pyrefly review best practices. For each diff that is passed in, use a new subagent to perform the review in parallel.

# Code Review Guidelines for Pyrefly

You are reviewing a Phabricator Diff for Pyrefly, a fast language server and type checker for Python written in Rust.

## Review Focus Areas

### Correctness
- Logic errors and edge cases
- Null/None/Option handling
- Off-by-one errors
- Resource leaks or missing cleanup
- Incorrect error handling

### Security
- Input validation issues
- Injection risks (command injection, path traversal)
- Unsafe deserialization
- Authentication/authorization gaps

### Performance
- N+1 patterns or unnecessary iterations
- Excessive allocations or cloning
- Blocking operations in async contexts
- Missing caching opportunities for expensive operations

### Testing
- Missing test coverage for new functionality
- Edge cases not covered by tests
- Flaky test patterns

### Style and Consistency
- Naming conventions (Rust: snake_case for functions, CamelCase for types)
- Consistency with existing codebase patterns
- Unnecessary complexity or over-engineering
- Check for missed opportunities for code reuse

### Architecture
- Separation of concerns
- Tight coupling between components
- Breaking changes to public APIs

## Pyrefly-Specific Guidelines

### Rust Conventions
- Prefer `impl Trait` over `dyn Trait` where possible
- Use `?` for error propagation instead of manual matching
- Avoid `.unwrap()` in production code; prefer `.expect()` with context or proper error handling
- Check for existing helpers in the `pyrefly_types` crate before manually creating or destructuring a `Type`

### Type Checker Concerns
- Changes to type inference should consider edge cases
- Modifications to error messages should be clear and actionable
- Performance is critical for the language server

### Code Style
- Keep It Simple Stupid (KISS) - minimize concept count
- Don't Repeat Yourself (DRY) - but don't over-abstract
- Comments should explain "why", not "what"
- Minimize places where `Expr` nodes are passed around

## Output Format

Provide your review in the following structure:

### Summary
A brief (2-3 sentence) overview of what this diff does and your overall assessment.

### Issues Found
List any problems, organized by severity:
- **Critical**: Must fix before landing (correctness, security)
- **Major**: Should fix (performance, maintainability)
- **Minor**: Nice to fix (style, documentation)

For each issue, include:
- File and line reference
- Description of the problem
- Suggested fix (if applicable)

### Positive Observations
Note any particularly well-done aspects of the code.

### Questions
Any clarifying questions for the author.
