# Website

The website is located at https://pyrefly.org/.
It is a combination of static content (built using [Docusaurus 2](https://docusaurus.io/)) and a sandbox built with Rust/WASM.

## Development

### Prerequisites

- Website development isn't currently supported on Windows, please use Linux or Mac or WSL (Windows Subsystem for Linux) for development.
- Ensure you have clang installed:
    - For mac:
        - Install [homebrew](https://brew.sh/`)
        - `brew install llvm` ([source](https://github.com/briansmith/ring/issues/1824))
    - For CentOS: `sudo dnf install clang`
- Install [nodejs](https://nodejs.org/en/download)
- Install yarn:
    - For mac: `brew install yarn`
    - Folr CentOS `sudo dnf install yarn`

### Install dependencies:

If you just want to install dependencies for work on the static docs site (i.e. not including the Sandbox), you can use the following command:

```bash
yarn install
```

If you need to do development work on the Sandbox feature, you will need to install the wasm dependencies as well, so use this command instead:

```bash
yarn install-with-wasm-deps
```

This builds the wasm dependencies and installs the yarn dependencies.

#### Troubleshooting

- If you run into any issues with "SSL peer certificate or SSH remote key was not OK (SSL certificate problem: unable to get local issuer certificate)", double check your `~/.gitconfig` that you aren't setting a proxy, as this will override the proxy override set in `setup_cargo.sh`.
- If `yarn start` fails with `Error: not found: hg`, you are on `docusaurus-plugin-internaldocs-fb` 1.19.1 or older. That plugin version calls `which.sync('hg')` without `nothrow`, so a missing Mercurial binary crashes the public docs server. Upgrade the plugin to 1.19.2+ (this repository pins `~1.19.3`). Public website development does not require Mercurial.

### Run the Website

Please ignore the flags for `unset FB_INTERNAL` and `INTERNAL_DOCS=0` in the following commands, these are commands used to hide information meant to be shown in the internal site.

For work on the static docs site (i.e. not including the Sandbox), you can use the following command:

```bash
yarn start
```

If you need to rebuild the wasm, you can use the following command instead:

```bash
yarn start-with-wasm
```

To see the website, go to the following URI in your browser:

```
localhost:3000
```

Most changes are reflected live without having to restart the server.

## Tests

We have snapshot tests powered by Jest.

Run them with:

```bash
yarn install
scripts/build_wasm_for_test_for_sandcastle.sh
yarn test
```

When you make a UI change, you should update the snapshots with:

```bash
yarn install
scripts/build_wasm_for_test_for_sandcastle.sh
yarn test --updateSnapshot
```

#### Troubleshooting

- If you are running into errors with compiling zstd on your mac (typically when running `build.sh` or `start.sh`), check that the llvm you installed with homebrew is actually being used [source](https://github.com/briansmith/ring/issues/1824#issuecomment-2059955073):
    - run `llvm-config --version`
    - if you don't see anything, run `export PATH="/opt/homebrew/opt/llvm/bin:$PATH"`
    - restart your terminal and try again

## Writing Docs

The docs are written in markdown (with JSX) and are located in the `docs` directory. Note that you should not explicitly add H1 headers to the docs, because docusaurus does not support this by default (https://github.com/facebook/docusaurus/issues/5036).

## Writing Blogs

The Pyrefly website uses Docusaurus v2, which has a built in blog feature. The blogs are written in markdown and are located in the `website/blog` directory. See the [Docusaurus blog documentation](https://docusaurus.io/docs/blog) for more details.

### Add a new blog post

1. create a new file in the `website/blog` directory using this file format: `YYYY-MM-DD-blog-post-title.md`. This will be the file where you write your blog post.
2. If you are a new author, also add yourself to the `authors.yml` file, then use your unique author name in the `author:` metadata in the blog file itself.
3. Go to your new blog file and write your content in markdown, adding appropriate metadata. Your blog file should look something like this:

    ```markdown
    ---
    title: My Blog # the title of your blog post
    description: This is my first blog! # a short description of your blog post
    slug: my-first-blog # what you want to url to be
    authors: jmarcey # your author name from the authors.yml file, for multiple authors, use a comma separated list [author1, author2]
    tags: [hello, python] # a list of tags for your blog post
    image: https://i.imgur.com/mErPwqL.png #the image to be used for preview links
    hide_table_of_contents: false # set to true to hide the table of contents
    ---

    ![image](path/to/image.png) # your blog header image

    The first line/paragraph of your blog post

    <!-- truncate -->

    the rest of your blog post.
    ```

4. Save your file and run `yarn start` to see your blog post in the website. Check that everything renders correctly.

### Adding a canonical url

If your blog was originally posted on another site, you can add a canonical url to the metadata of your blog post. This will allow search engines to know that this is a duplicate of the original post, and will help with SEO. To do this you need to manually add a `<head>` tag into your blog post file ([source](https://github.com/facebook/docusaurus/issues/2603)). For example:

```
   <head>
    <link rel="canonical" href="<url-to-original-blog-post>" />
   </head>
```

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
