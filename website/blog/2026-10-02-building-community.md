---
title: "Building an Open Source Community in the Age of AI"
description: How Pyrefly is adapting its contributing guidelines to the age of AI coding agents.
slug: building-community
authors: [rebeccachen]
tags: [news, ai-agents]
hide_table_of_contents: true
---

A question that has been on our minds lately is how to build a community around Pyrefly when AI is rapidly reshaping longstanding open source norms. I don't think anyone has fully figured this out yet, including us, but we're making some changes to Pyrefly's contributing guidelines that we think are a good start. Here's what we've seen change in open source and what we're doing about it.

<!--truncate-->

## `@deprecated("open source norms")`

Traditionally, an active and welcoming GitHub presence has been one of a project’s main levers for building community. That reality is quickly breaking down:

| Conventional wisdom | New reality |
| :---- | :---- |
| Writing a PR takes time and effort proportional to its size. Maintainers should prioritize responding to PRs promptly. | With coding agents, contributors can rapidly generate PRs that take much longer to review than they did to write. Large PRs are especially likely to be slop. |
| Users who have positive interactions with maintainers spread positive word-of-mouth. | Maintainers increasingly find themselves talking to Gemetaclaudex output. |
| Repeat contributors build context and expertise over time. | Contributors who rely heavily on AI-generated code have fewer opportunities to learn. |
| Issue and PR volume and throughput are good measures of a project’s health. | AI-generated issues and PRs obscure meaningful signals. |

Maintainers can sink more time than ever into conventional community-building activities, while the payoff is lower than ever.

## `# TODO(pyrefly)`

Pyrefly’s backlog of untriaged issues and unreviewed PRs is growing steadily longer. A sizable fraction of our Discord activity is PR review nudges. On our current trajectory, we run the risk of burning out our team, burning community goodwill, or both.

## `copy.copy(other_project)`?

How have other open source projects adapted to this new reality? A common reaction has been to introduce policies on AI usage. Looking at projects like [PyTorch](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md), [JAX](https://docs.jax.dev/en/latest/contributing.html#can-i-contribute-ai-generated-code), and [Zed](https://github.com/zed-industries/zed/blob/main/CONTRIBUTING.md#ai-policy), a typical set of policies is:

* LLMs cannot be used for communication.
* Contributions from autonomous agents are not accepted.
* You have to understand and be able to explain AI-generated code.

Some projects have gone further and banned AI usage entirely.

GitHub has also introduced settings for [throttling PRs](https://github.blog/changelog/2026-06-17-limit-open-pull-requests-for-users-without-write-access/) or even [disabling them altogether](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/disabling-pull-requests).

## `from __future__ import guidelines`

The Pyrefly devs are enthusiastic AI users, and we don't think the answer is to strictly police AI usage. Neither do we want to restrict pull requests: we strongly believe that accepting contributions is an essential part of what makes a project open source. So we're following the lead of other projects in tightening our AI policy against misuse of AI tooling, but - more importantly - we're adding explicit guidance to help contributors understand how to engage with Pyrefly, regardless of whether or how you're using AI.

### AI Policy

We've adopted a new [AI policy](https://github.com/facebook/pyrefly/blob/main/AI_POLICY.md) with one central tenet: we want to interact with you, not your AI. We no longer allow LLM-generated communications, autonomous agent contributions, or PRs that haven't been thoroughly reviewed by their human author. We also ask that AI not be used to one-shot issues labeled as "good first issue", since the purpose of these issues is to provide learning opportunities.

We still allow and encourage using AI to learn about Pyrefly, investigate issues, write code, validate changes, and any other task that can benefit from AI, as long as you understand and own your AI-assisted contributions.

### Guidance

We've expanded our contributing guidelines with sections on [contributor etiquette](https://github.com/facebook/pyrefly/blob/main/CONTRIBUTING.md#contributor-etiquette), [getting started with your first contribution](https://github.com/facebook/pyrefly/blob/main/CONTRIBUTING.md#getting-started), and [creating and managing PRs](https://github.com/facebook/pyrefly/blob/main/CONTRIBUTING.md#making-a-pull-request). We've also fleshed out our PR template to make these guidelines easier to find. We're aiming for these to be helpful, common-sense tips rather than onerous rules.

Finally, none of these changes are meant to keep people out. By spending less time on slop, we hope to spend more time building a space where contributors learn, grow, and stick around. We're still figuring out the best way to do that, and we expect to keep adjusting as the ways people use AI evolve. If you'd like to contribute, share ideas, or just say hi, you can find us on [Discord](https://discord.com/invite/Cf7mFQtW7W) or [GitHub](https://github.com/facebook/pyrefly).

*Any slop in this post is 100% human-generated.*
