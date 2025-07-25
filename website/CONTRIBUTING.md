# Contributing to Pyrefly Website

Please see CONTRIBUTING.md in the root of the repository for more details on contributing.
This file shows details on how to develop the Website locally, as well as deployment instructions.

## Development

### Prerequisites

- Website development isn't currently supported on Windows, please use Linux or Mac or WSL (Windows Subsystem for Linux) for development.
- Ensure you have clang installed:
    - For mac: `brew install llvm` ([source](https://github.com/briansmith/ring/issues/1824))
    - For CentOS: `sudo dnf install clang`
- Install Mercurial: `sudo apt install mercurial`

Install dependencies:

```bash
yarn install-with-wasm-deps
```

This builds the wasm dependencies and installs the yarn dependencies.
If you run into any issues with "SSL peer certificate or SSH remote key was not OK (SSL certificate problem: unable to get local issuer certificate)", double check your `~/.gitconfig` that you aren't setting a proxy, as this will override the proxy override set in `setup_cargo.sh`.
If you are running into issues with compiling zstd on your mac when running `build.sh` or `start.sh`, see `pyrefly_wasm/README.md` for more details.

### Running the Website

**Running the External Site**

Please ignore the flags for `unset FB_INTERNAL` and `INTERNAL_DOCS=0` in the following commands, these are commands used to hide information meant to be shown in the internal site.

```bash
yarn start-with-wasm
```

If you don't need to rebuild the wasm, you can use the following command instead:

```bash
yarn start
```

To see the website, go to the following URI in your browser:

```
localhost:3000
```

Most changes are reflected live without having to restart the server.

**Running the Internal Docs Site (for Meta Employees)**

If you want to run the static docs website, run the following command:

```bash
yarn start-internal
```

Similar to above, if you don't need to rebuild the wasm, you can use the following command instead:

```bash
yarn start-internal-with-wasm
```

## Writing Docs

The docs are written in markdown (with JSX) and are located in the `docs` directory. Note that you should not explicitly add H1 headers to the docs, because docusaurus does not support this by default (https://github.com/facebook/docusaurus/issues/5036).

## Deployment

The website is currently deployed on a daily basis (14 UTC) to pyrefly.org. You can also choose to manually run this if needed.
It is triggered by the [deploy website workflow](https://github.com/facebook/pyrefly/actions/workflows/deploy_website.yml), which first [builds and test the website](https://github.com/facebook/pyrefly/actions/workflows/build_and_test_website.yml).

For details on how to manually trigger a github workflow, see [this link](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/manually-running-a-workflow).

## Rollback

If there's a major issue on the website and you want to quickly roll it back to a stable state, you can Run the [rollback website workflow](https://github.com/facebook/pyrefly/actions/workflows/rollback_website.yml) with the "Use workflow from" dropdown set to the branch you made the changes on. This will prompt you with the date and run ID To rollback to. You can find the run ID by going to the [Deploy Website workflow](https://github.com/facebook/pyrefly/actions/workflows/deploy_website.yml), clicking into a run and taking the number from the URL. For example, if the URL is `https://github.com/facebook/pyrefly/actions/runs/123456789`, then the run ID is `123456789`.

**Internal Docs Site (for Meta Employees)**
The internal docs page is deployed whenever diffs that touches the site lands. A manual deploy can also be triggered using the static docs hub: https://www.internalfb.com/staticdocs/hub.

## Logging

To debug logging, you can use the chrome extension [google analytics debugger](https://chromewebstore.google.com/detail/google-analytics-debugger/jnkmfdileelhofjcijamephohjechhna) to see events logged to google analytics. Note that for events to be logged, you need to not be in development mode, which can be done using `yarn build` and
`yarn serve` instead of `yarn start`.
