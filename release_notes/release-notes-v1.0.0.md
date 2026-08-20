**Status: STABLE**
*Release date: 12 May 2026*

## Pyrefly v1.0.0 is here\!

We're thrilled to announce that Pyrefly has reached its stable 1.0.0 release\! Since our [beta release](https://github.com/facebook/pyrefly/releases/tag/0.42.0) in November 2025, we've fixed hundreds of bugs, improved performance, and added lots of new functionality. Pyrefly is already the default type checker for Instagram at Meta and has been adopted by other large production codebases like PyTorch and JAX. Today, we're making it official: Pyrefly is production ready.

This would not have been possible without our amazing open-source community. To everyone who filed GitHub issues, submitted pull requests, gave us feedback at conferences, or joined us on Discord: thank you. Your contributions shaped this release.

These release notes cover the major highlights since our beta release. For the full history, see our [past weekly release notes](https://github.com/facebook/pyrefly/releases).

---

## Performance Improvements

We've continued to push Pyrefly's performance since the [speed improvements we shared in February](https://pyrefly.org/blog/2026/02/06/performance-improvements/). Since beta:

- **2–125x faster updated diagnostics** after saving a file (no, that’s not a typo\!). Thanks to fine-grained dependency tracking and streaming diagnostics, updates now consistently arrive in milliseconds
- **20–36% faster full type checking** on large projects like PyTorch and Pandas
- **2–3x faster initial indexing** when Pyrefly first scans your project
- **40–60% less memory usage** during both indexing and incremental type checking

(Tested on an M4 Macbook Pro using open-source benchmarks from [type\_coverage\_py](https://github.com/lolpack/type_coverage_py) and [ty\_benchmark](https://github.com/astral-sh/ruff/tree/e990dfd069fceef96f797b46161ef78862608449/scripts/ty_benchmark).)

Compare the performance of Pyrefly and other Python type checkers on our regularly updated [benchmarking suite](https://python-type-checking.com/typecheck_benchmark/), which runs against 53 popular Python packages.

---

## Configuration Presets

A new `preset` configuration option provides named bundles of error severities and behavior settings.

| Preset | Description |
| :---- | :---- |
| `off` | Silences all diagnostics. Useful for IDE-only users or if you want total control of which errors are enabled. |
| `basic` | Low-noise, high-confidence diagnostics only (syntax errors, missing imports, unknown names, etc.). Ideal for unconfigured projects or IDE-first users. |
| `legacy` | For codebases migrating from mypy. Disables checks mypy doesn't have. `pyrefly init` now emits this preset automatically when migrating from a mypy config. |
| `default` | The standard Pyrefly experience. Equivalent to having no preset. |
| `strict` | Enables additional strict checks on top of the `default` preset. For users who want to avoid `Any` types in their codebase. |

See the [configuration docs](https://pyrefly.org/en/docs/configuration/#preset) for details.

---

## Onboarding Experience

We’ve made improvements to the out-of-the-box experience for projects without a `pyrefly.toml`.

- **Automatic config synthesis** — if you have a mypy or pyright config, Pyrefly automatically migrates your settings and synthesizes an appropriate in-memory Pyrefly config. (This is the same migration that `pyrefly init` would commit to disk.)
- **Basic preset for unconfigured projects** — projects with no type checker config get the lightweight “basic” preset, which surfaces only high-confidence errors.
- **VS Code status bar** — the status bar shows the active preset — e.g. Pyrefly (Basic) or Pyrefly (Legacy) — so you always know which mode is active.
- **Type error display settings** — new VS Code settings let you control which preset applies to unconfigured files and suppress all diagnostics workspace-wide.

---

## Type Checker Improvements

We've been hard at work making the type checker robust and feature-complete, with a focus on driving down false positives and improving type quality in real-world code bases. Here are some highlights:

- Across the board we've eliminated many sources of false positives in enums, dataclasses, ParamSpec, descriptors, and more.
- Support has been added for more type narrowing patterns, including preserving narrows in nested scopes and recognizing container membership checks.
- Overload resolution was substantially reworked to handle more real-world patterns.
- Pyrefly’s conformance to the [Python typing specification](https://typing.readthedocs.io/en/latest/spec/) has improved from 70% at beta to over 90% today.
- We've added experimental support for tracking tensor dimensions through PyTorch models — see "What's Next" below.

---

## LSP & IDE Improvements

- We've added new refactoring capabilities like Safe Delete (with reference checking) and bulk `source.fixAll`.
- Navigation is more precise, and hover cards surface richer information for imports, tuples, and NamedTuples.
- Workspace mode is more stable, with multiple crash fixes and improved diagnostic publishing.

---

## Framework & Notebook Support

- **Django** — Pyrefly has improved support for model relationships, fields, and views, and understands [factory\_boy](https://factoryboy.readthedocs.io/) factories.
- **Pydantic** — Pyrefly models Pydantic's runtime behavior more faithfully, with support for lax mode and range constraint validation, and handles more of the Pydantic ecosystem: `RootModel`, `pydantic-settings`, and `pydantic.dataclasses`.
- **Pytest integration** — We've added Code Lens run buttons for test functions, as well as code actions to annotate fixture return types and parameters.
- **Jupyter notebooks** — `.ipynb` IDE support has reached full parity with `.py` files, with rename, find references, code actions, and document symbols all supported.

---

## Complementary Tooling

Pyrefly ships with tools to aid with adopting type checking in an existing codebase. Two new tools since beta:

- [**`pyrefly report`](https://pyrefly.org/en/docs/report/)** outputs a JSON report with annotation completeness and type completeness metrics per function, class, and module, so you can track coverage over time.
- [**Baseline files**](https://pyrefly.org/en/docs/error-suppressions/#baseline-files-experimental) let you snapshot current errors into a JSON file so only *new* errors are reported, as an alternative to inline suppression comments.

---

## Updated Version Policy

Going forward, we’ll switch from a weekly to monthly cadence for minor (`1.x.0`) releases, with patch releases in between as-needed for critical fixes. We’ll continue providing [release notes](https://github.com/facebook/pyrefly/releases) for minor versions, so you can see what’s new in each release.

---

## What's Next

- **Tensor shape checking** — Experimental support for tracking tensor dimensions through PyTorch models and catching shape mismatches statically. [Learn more](https://pyrefly.org/en/docs/tensor-shapes/).
- **Pyrefly \+ AI agents** — Pyrefly's speed makes it a natural verification step in agentic workflows. See our guide on [adding Pyrefly to your agentic loop](https://pyrefly.org/blog/pyrefly-agentic-loop/).
- **Continued improvements** — We'll keep expanding library support, reducing false positives, and iterating on your feedback. Let us know what you need on [GitHub](https://github.com/facebook/pyrefly/issues) or [Discord](https://discord.gg/Cf7mFQtW7W).

---

## Thank you to all our contributors\!

The following people have directly contributed at least once to the development of Pyrefly since our first Alpha release.

@aahanaggarwal, @aaron-ang, @abdallhfattah, @Abel981, @abesto, @abhi-jha, @ABohra3, @Adamkaram, @Adist319, @adsi7698, @AHA705, @ahornby, @airvzxf, @ajaymiranda, @akmalsoliev, Alan Du, @Alex-Aron, Alok Priyadarshi, Alvaro Leiva Geisse, @AmalenduManoj, Anass Al-Wohoush, Anqi Wu, @ArchieBinnie, @arnav-jain1, @arosenber, @arthaud, @Arths17, @AryanBagade, @asm89, @asukaminato0721, @austin3dickey, @avikchaudhuri, Ben Carr, @bharath-2022, @bigfootjon, @bluetech, @bowiechen, @brchien, Brian Rosenfeld, @bv-saketha-rama, @capickett, Carlos Fernandez, @cbarrete, @cclauss, @charliecloudberry, @cjlongoria, Claudionor Santos, @connernilsen, @CookieComputing, @cooperlees, @cybardev, @DanielNoord, @danielocfb-test, Danny Yang, @darricklaidin, David Tolnay, David Tolnay, @davidbarsky, @ddrcoder, @dhleong, Dhruv Mongia, @diliop, @disrupted, @dluo, @Dogacel, @ducdetronquito, @ericweb2, @facebook-github-bot, @fangyi-zhou, @fannheyward, @fatelei, generatedunixname2066905484085733, generatedunixname537391475639613, generatedunixname89002005232357, generatedunixname89002005287564, generatedunixname89002005307016, @github-main-user, @grantlouisherman, @grievejia, @gvozdvmozgu, @hanzel-sc, @hashiranhar, @hugovk, @iamPulakesh, @IDrokin117, @immanuel-peter, @Imran-S-heikh, @InSyncWithFoo, @ipr-ams, Ivan Loskutov, @j-piasecki, @jack-mcivor, @jackulau, @jagill, Jaimin Brahmbhatt, @JakobDegen, @javabster, @jchanke, Jess Wass, @jorenham, Jun Hao, @jvansch1, @K1T3K1, @KaranPradhan266, @Karman-singh15, Keito Uchiyama, @kinto0, @kitagry, @knQzx, @krathul, @krikera, @Krishnachaitanyakc, @kshitijgetsac, @kv9898, Li Shen, lianne, @lolpack, @Louisvranderick, @maggiemoss, @maifeeulasad, @maldoinc, @MarcoGorelli, @markmarkmarkthebest, @martindemello, @melvinhe, Miae Kim, @michaelcortese, @michel-slm, Mick Killianey, @migeed-z, Miles Conn, @mohesham88, Morgan Bartholomew, @MountainGod2, @mrsobakin, @mstykow, @mvanhorn, @NathanTempest, @ndmitchell, @nhawkes, @nikita-ashihmin, @nitinsingh-meta, @NSPC911, @ogios, @oopscompiled, @oriori1703, Owen Valentine, @oyarsa, @pavelzw, @pawelstrzmeta, @pawlowskialex, @PhilHem, @prasannavenkateshgit, @praskr-wisdom, @Prathamesh-tech-eng, @pswitchy, @pt2302, @QEDady, @QuantumManiac, @quark-zju, @Raf-Hs, @Rayahhhmed, @rchen152, @rchiodo, @regexyl, @rexledesma, @ricardoleal20, Robert Rusch, @robertoaloi, Ron Mordechai, @rubmary, @runlevel5, @salmanmkc, @salvatorebenedetto, @SamChou19815, @samwgoldman, @sandeshbhusal, @sargun, @Sehat1137, @self-sasi, @serephus, @sgavriil01, @shayne-fletcher, @shining44, @ship-it-ship-it, @shuv-amp, @sifex, @simonhollis, @singiamtel, @ska-kialo, @slawlor, @Solumin, @stanleyshen2003, @stroxler, Takuma Iwaki, @Tamchuk, @tannguyencse19, @tejasreddyvepala, @terror, @TheRustyPickle, @thomaspolasek, Tianhan Lu, @tkaleas, @tsembp, @ukautz, @Viicos, @vinnymeller, Vladimir Matveev, Weixi Ma, @Wilfred, Will Li, @willylau, @xaskii, @xl4624, @yamgent, @yangdanny97, @yeetypete, @yslim-030622, @zachmullen, @zanieb, @zbowling, @Zeko369, @zertosh, @zhuolix, @zpao, @zriser, @zsol
