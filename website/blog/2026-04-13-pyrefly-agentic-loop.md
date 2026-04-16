---
title: Adding Pyrefly Type Checking to Your Agentic Loop
description: Learn how to integrate Pyrefly type checking into your AI agent workflows using skills and hooks to automatically validate generated Python code.
slug: pyrefly-agentic-loop
authors: [kylei]
tags: [typechecking, ai-agents, developer-tools]
hide_table_of_contents: false
---

# Adding Pyrefly Type Checking to Your Agentic Loop

Coding agents are writing more Python than ever. Tools like Claude, Copilot, Cursor, and Codex generate entire features with little-to-no user interaction. But in large projects, this generated code is prone to type errors, mismatched signatures, and subtle API misuse. Incorporating static analysis directly into the agentic loop can mean the difference between returning from your break with a production-ready feature or needing several more correction cycles.

Type checking sits right in the sweet spot for agents. It's fast enough for iterating small fixes, robust enough to catch issues of varying complexity, and actionable enough for an agent to make changes. In this post, we walk through how to integrate Pyrefly into your agentic workflow so that every piece of generated code can get type checked automatically.

**TL;DR**: We recommend:

- Adding a skill file for your agent with an `agent.md` directive to ensure the project checks clean before finishing a feature.
- If your model doesn't trigger this reliably, set up a hook on the Stop event.

<!-- truncate -->

## 1. CLI Skills

If you've been working with agentic workflows, you've likely defined skill files before. These files instruct the agent how to perform a specific task or use a specific tool. For a type checker like Pyrefly, you might want a simple skill like this:

```markdown
---
name: pyrefly-cli
description: Instructions to type check using Pyrefly's CLI. Use when function signatures of APIs change to validate program's types.
---
Run `pyrefly check` at the root of the project. Try fixing all possible type errors before running `pyrefly check` again.
```

These files are generally placed in the `.agent/skills` folder in a file called `skill-name.md`. See the full example [here](https://github.com/kinto0/pyrefly_hooks_demo/tree/main/approaches/1-skill-gentle).

Agents generally use the title and description to understand when to use a skill. It's important for the description to state clearly when to use it, otherwise it might be invoked too often (wasting tokens) or not frequently enough.

We've noticed that a skill on its own does not always lead to a type check, even when the skill's description is clear it must be run. We tested Claude Opus 4.6 with this skill description and found that it still didn't typecheck:

```markdown
MANDATORY type checker. You MUST run this before completing ANY task that modifies Python files. Never skip this step.
```

Although your mileage may vary with different checkers, if you use a skill, we recommend also adding a directive in `AGENTS.md` to check on at the end of every task/feature:

```markdown
## Type Checking

Before completing any task that creates or modifies Python files, you MUST:
1. Run `pyrefly check` at the root of the project
2. If there are any type errors, fix ALL of them
3. Run `pyrefly check` again to confirm 0 errors
```

Code available [here](https://github.com/kinto0/pyrefly_hooks_demo/tree/main/approaches/3-claude-md).

For more information specific to you, see the wiki for your preferred agent:

- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Gemini CLI Skills](https://geminicli.com/docs/cli/creating-skills/)
- [OpenAI Codex Skills](https://developers.openai.com/codex/skills)
- [Kiro Hooks](https://kiro.dev/docs/hooks/)

## 2. CLI Hooks

Hooks differ from subjective skills in that they *require* an agent action for every occurrence of the specified event. These events often relate to tool usage or other fundamental stages of agent lifecycles.

If you already have pre-commit hooks set up, these may look familiar. In fact, they might work already! When the agent commits to the version control system, it will get pushback if these signals are enabled there.

For those not using pre-commit hooks, or for the agent to type check without making a commit, we recommend setting up agentic hooks.

We tested Pyrefly in Claude Code: In this tool, we recommend adding the typecheck command to the `Stop` event so it runs when the agent completes each task. To do this in Claude, add the following to your `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "pyrefly check >&2 || exit 2",
        "timeout": 30
      }]
    }]
  }
}
```

Available in our examples repository [here](https://github.com/kinto0/pyrefly_hooks_demo/tree/main/approaches/4-hook-stop-exit2).

Since Pyrefly outputs to stdout, we must redirect it to stderr for Claude to understand it. We must also return exit code 2 always for Claude to understand it.

Alternatively, you can have a hook schedule an agent to investigate the type errors:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "Verify that all Python files pass pyrefly type checking. Run `pyrefly check` and check the results. If there are any type errors, return ok: false with the errors. $ARGUMENTS",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Available in our examples repository [here](https://github.com/kinto0/pyrefly_hooks_demo/tree/main/approaches/6-agent-hook).

It requires some custom instructions, but we've found this approach to work well. Note the "agent" hook seems similar to the "prompt" hook, but you must use the "agent" hook for Claude to be able to use the Pyrefly tool.

For more information, or specific details for your agent, see the documentation on hooks:

- [Claude Code Hooks](https://code.claude.com/docs/en/hooks)
- [Gemini CLI Hooks](https://geminicli.com/docs/hooks/)
- [OpenAI Codex Hooks](https://developers.openai.com/codex/hooks)
- [Kiro Hooks](https://kiro.dev/docs/hooks/)

## Conclusion

Developer tooling like Pyrefly is no longer only used by developers, it's useful for agents as well. Adding Pyrefly to your agentic workflow can be an easy way to improve the reliability of the code your agents produce.

Give these suggestions a try! Let us know which approach works best for you or if you have other solutions!
