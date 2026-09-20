---
title: "Talk: Type Checking in Agentic Workflows"
description: Does adding type checking to an agentic workflow really help agents?
slug: type-checking-agentic-workflows
authors: [connernilsen]
tags: [typechecking, ai-agents]
hide_table_of_contents: false
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/xNaKm4fTFtw?si=jYzPHYUWa64QT1Sw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<br/>

Does adding type checking to an agentic workflow really help agents?

We ran an experiment recently to determine whether there are improvements in the success rate for completing different kinds of tasks. In theory, having a type checker present should help the agent catch type errors earlier, validate fixes incrementally as it works, and reduce the need for slow, test-driven iterative feedback loops.

This talk was originally presented at the PyCon US 2026 Typing Conference. The slides and edited transcript are provided below for your convenience.

📎 [Slides (PDF)](https://drive.google.com/file/d/1dUQPGxaV_9kN7ulOr01ojg0AOOu-Gf9L/view)

<!-- truncate -->

## Transcript

The agenda for today: we'll start with the background to give you some context, go through some of our findings, talk about our experiment setup, and end with some of the remaining open questions. The slide deck contains an appendix with more details which you can read later.

### Background

At a high level, our experiment consisted of having a bunch of tasks and having agents attempt to work on those tasks with and without different type checkers. We mainly looked at three success metrics:

- **Success rate** — how often the agent was able to successfully complete different tasks, verified with tests.
- **Number of steps** — how many times an agent searches for information, makes edits, and does other operations.
- **Task duration** — basically wall time: how long does it take an agent to do the task?

### Findings

To answer the initial question of whether type checking helped or not: it depends.

We noticed that **type checking helped the agent only when the code base was well typed**. In those cases there was less exploration work by the agent, and the success rate increased from about 80% to 84%. We also saw a reduced number of steps, and the agent generally finished tasks faster.

When there was low type coverage, however, there was no meaningful impact: we noticed that the type errors actually distracted the agent and took it off course. In those cases, the agent would typically solve for type checker cleanliness in adjacent code rather than solving the task it was meant to do. This involved fixing import issues, missing attributes, or type signature mismatches, instead of focusing on real fixes.

We also found that *how* you deliver the feedback to the agent matters at least as much as the feedback itself. Models generally don't just use tools that you tell them about. To get around this, we ended up creating a separate lightweight agent to make sure that the model focused on the type errors it saw on every edit. More on that later.

Models also responded better to feedback delivered as a separate conversation step rather than in edit output—that is, as a separate conversation step rather than bundled in when we told the agent "Hey, your edits were successful," or other follow-up information.

Different models also had different sensitivities to feedback. For example, Claude was very sensitive to errors. It would fix whatever type errors it was shown, but that also meant noisy signals would throw it off course often. GPT Codex, however, was very goal-oriented and needed structural intervention to make sure it actually addressed the errors we provided. We expect these sensitivities to change as the models progress over time. So it's worth exploring how aggressively you want to filter the errors that are surfaced, and where in your agentic loop you want to add this feedback.

Our last finding is that feedback at higher frequencies gives the model more confidence. Consistent feedback verifies that the model is going in the right direction and prevents what we found are "search spirals", where the model goes back and re-verifies things that occur after every edit. We also noticed that empty output, like "type checked X files and found zero errors," acted as a great external reflection cue for agents and helped make sure they stayed on course.

Only running once at the end of a task is too late. In that case, the models never went back to fix anything. It needs to be very consistent that you're providing this feedback to the model.

### Experiment Setup

Now onto how we came up with some of our findings. We ran two similar experiments: one on an open-source benchmark and another on an internal benchmark that we created.

The external experiment tested Pyrefly on **SWE-bench Verified**, a benchmark for evaluating AI agents on solving real-world engineering tasks. This involves a couple of very popular libraries like Matplotlib, Django, SymPy, and others. Just due to the involvement of different legacy code bases, there's generally low type coverage in these code bases.

We also re-ran the experiment with an internal benchmark called **MetaSWEBench**, internally curated for evaluating AI agents. With this experiment, we used Pyre as our type checker, since that was what was available at the time the code in MetaSWEBench was committed. This benchmark had generally higher type coverage, just due to code quality standards internally.

#### The custom integration

I mentioned earlier that we had a lightweight custom agent. This pretty much consisted of a simple think–act–observe loop, rather than an off-the-shelf coding agent like Claude Code or Codex. We would plug the actual models directly into that to get our agent.

We did it this way because it gives us full control over how the agent interacts with Pyrefly. While it's common to write agent skills or `skills.md` files to interact with different tools, we found that this provides a similar but less rigid outcome. Models currently need a lot of structure and gating to ensure that they actually address issues provided by lints or other verifiable checks.

We also didn't want the agent quality to be the variable under test. Simple logic means that the other tools available to the agent aren't the thing we're testing. We're really only looking at the type checker itself and the model using the type checker.

#### Models

For the external experiment we used Claude Sonnet 4.5 and GPT-5.3. Those models are a little older, but we used them because the newer models are subject to rate limits at the moment, which constrained our ability to run many tests in parallel. For the internal run we used Claude Opus. GPT wasn't used internally just to constraints around the models that were available to us at the time.

Finally, we tested several different methods of interacting with the type checker to see which approaches have benefits over others. This is mostly the stuff we talked about before, such as when to check or how type errors are presented. As mentioned earlier, we found that checks on each edit, with results included as part of the conversation. were the most effective at getting the models to do what we want.

### Open Questions

There are still many open questions remaining. This is definitely a non-exhaustive list, and we're curious about testing more, but here's what's top of mind.

A lot of it is around the question: what if we re-ran the experiment with different inputs?

- **Different type checkers.** Instead of running the internal experiment with Pyre, what if we redid it with Pyrefly, ty, or another type checker? Do we see the same improvements? Could we plug in other models to see if there's a similar improvement or degradation? Is the quality of the type checker, or its conformance rating, or other aspects of how the type checker runs something that affects how well these models complete their tasks? Or is there maybe just something about the type checker we used at the time that led to that ~4.5% improvement—maybe the way we had errors selected or what we filtered?
- **A well-typed corpus.** What if we re-ran the experiment with a well-typed corpus? That would probably be the most direct way to validate that ~4.5 percentage point improvement in open-source code bases. For that, we're thinking we could curate a subset of SWE-bench from better-typed repositories, or come up with our own benchmark from natively typed projects.
- **Frontier models.** What if we use the newest models available to us? Will there be better internal reflection, or do they consume feedback differently in ways that just work better with type checking?
- **Other task types.** Probably the biggest question we have: are there other tasks that perform better with type checking available? SWE-bench is a very unfocused, generic benchmark where tasks really don't have much to do with type checking. So there's a question of whether there are more targeted problems where type checking is much more effective at improving model performance—or where even just the pervasiveness of types helps the model work better.

### Closing

If you're interested in this problem, please reach out on [Discord](https://discord.gg/Cf7mFQtW7W). This is definitely something we're interested in continuing research about, and we'd love to collaborate on it.

And huge thanks to [Jia Chen](https://github.com/grievejia) for putting the experiment together, as well as the rest of the Pyrefly team and our open-source contributors for all the work that's gone into Pyrefly. Thank you.
