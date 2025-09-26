/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';
import { fbContent } from 'docusaurus-plugin-internaldocs-fb/internal';
import * as webpack from 'webpack';
import MonacoWebpackPlugin from 'monaco-editor-webpack-plugin';
import StylexPlugin from '@stylexjs/webpack-plugin';
import PyodidePlugin from '@pyodide/webpack-plugin';
import path from "path";
import fs from "fs";

const BasePath = 'en/docs';

const baseUrl = process.env.DOCUSAURUS_BASE_URL || '/';

function getNavBarItems() {
    return [
        {
            to: `${BasePath}/`,
            activeBasePath: BasePath,
            label: 'Docs',
            position: 'left' as const,
        },
        {
            to: `${BasePath}/typing-for-python-developers/`,
            activeBasePath: `${BasePath}/typing-for-python-developers`,
            label: 'Learn',
            position: 'left' as const,
        },

        // TODO (T222948083) Re-enable internal sandbox after we fix issues with sandbox on static docs
        process.env.INTERNAL_STATIC_DOCS === '1'
        ? {
            href: 'https://pyrefly.org/sandbox',
            label: 'Sandbox',
            position: 'left' as const,
          }
        : {
            to: 'sandbox/',
            activeBasePath: 'sandbox',
            label: 'Sandbox',
            position: 'left' as const,
        },
        {
            to: `${BasePath}/installation/`,
            activeBasePath: `${BasePath}/installation`,
            label: 'Install',
            position: 'left' as const,
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        // Please keep GitHub link to the right for consistency.
        {
            href: 'https://github.com/facebook/pyrefly',
            'aria-label': 'GitHub',
            position: 'right' as const,
            className: 'navbar__icon github__link',
        },
        {
            href: 'https://discord.gg/Cf7mFQtW7W',
            'aria-label': 'Discord',
            position: 'right' as const,
            className: 'navbar__icon discord__link',
        },
    ].filter((x): x is NonNullable<typeof x> => x != null);
}

async function generateLlmsTxt({ content, routes, outDir }, context) {
    const { allMdx } = content as { allMdx: string[] };
    // Write concatenated MDX content
    const concatenatedPath = path.join(outDir, "llms-full.txt");
    await fs.promises.writeFile(concatenatedPath, allMdx.join("\n\n---\n\n"));
    // we need to dig down several layers:
    // find PluginRouteConfig marked by plugin.name === "docusaurus-plugin-content-docs"
    const docsPluginRouteConfig = routes.filter(
        (route) => route.plugin.name === "docusaurus-plugin-content-docs"
    )[0];
    // docsPluginRouteConfig has a routes property has a record with the path "/" that contains all docs routes.
    const allDocsRouteConfig = docsPluginRouteConfig.routes?.filter(
        (route) => route.path.endsWith(`/${BasePath}/`)
    )[0];
    if (!allDocsRouteConfig?.props?.version) {
        throw new Error(`/${BasePath}/ route not found`);
    }
    // this route config has a `props` property that contains the current documentation.
    const currentVersionDocsRoutes = (
        allDocsRouteConfig.props.version as Record<string, unknown>
    ).docs as Record<string, Record<string, unknown>>;
    // for every single docs route we now parse a path (which is the key) and a title
    const docsRecords = Object.entries(currentVersionDocsRoutes).map(([path, record]) => {
        return `- [${record.title}](${path}): ${record.description}`;
    });
    const llmsTxt = `# ${context.siteConfig.title}\n\n## Docs\n\n${docsRecords.join("\n")}`;
    const llmsTxtPath = path.join(outDir, "llms.txt");
    fs.writeFileSync(llmsTxtPath, llmsTxt);
}

const config: Config = {
    title: 'Pyrefly',
    tagline: 'A fast Python type checker and language server',
    url: 'https://pyrefly.org',
    baseUrl: baseUrl,
    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'warn',
    // This is used instead of favicon to get the right icon based on light vs dark mode in google chrome. Note that this currently doesn't work in
    // firefox, which is intentional as we expect more users to be using chrome instead of firefox. If we want to have this working in
    // firefox instead of google chrome, we can add `favicon: "img/Pyrefly-Symbol-Dynamic.svg", which will override the headTags setting
    headTags: [
        {
          tagName: "link",
          attributes: {
            rel: "icon",
            href: baseUrl + "img/Pyrefly-Symbol.svg",
            type: "image/svg",
            sizes: "32x32",
            media: "(prefers-color-scheme: light)",
          },
        },
        {
          tagName: "link",
          attributes: {
            rel: "icon",
            href: baseUrl + "img/Pyrefly-Symbol-Invert.svg",
            type: "image/svg",
            sizes: "32x32",
            media: "(prefers-color-scheme: dark)",
          },
        },
        // Open Graph meta tags for social media previews
        {
          tagName: "meta",
          attributes: {
            property: "og:image",
            content: baseUrl + "img/Pyrefly-Preview-Symbol.png",
          },
        },
        {
          tagName: "meta",
          attributes: {
            property: "og:image:width",
            content: "1200",
          },
        },
        {
          tagName: "meta",
          attributes: {
            property: "og:image:height",
            content: "630",
          },
        },
        {
          tagName: "meta",
          attributes: {
            property: "og:image:type",
            content: "image/png",
          },
        },
      ],
    organizationName: 'facebook', // Usually your GitHub org/user name.
    projectName: 'Pyre', // Usually your repo name.
    trailingSlash: true,
    markdown: {
        mermaid: true,
    },
    themes: ['@docusaurus/theme-mermaid'],
    // We likely won't be able to use the faster docusaurus build for this website due to
    // the custom configuration in the plugins section. We would need another way to import these
    // plugins into the build if we want to use the experimental_faster option.
    // See https://fb.workplace.com/groups/docusaurus/posts/2314262428945865/?comment_id=2314271988944909
    // for more details.
    // future: {
    //   experimental_faster: true,
    // },
    plugins: [
        async function pluginLlmsTxt(context) {
            // This plugin generates a llms-full.txt file (containing all the content) and a llms.txt file
            // (containing a link/overview for each page) to serve as context for LLMs.
            // See https://llmstxt.org/
            // Based on jharrel's solution from https://github.com/facebook/docusaurus/issues/10899
            return {
                name: "llms-txt-plugin",
                loadContent: async () => {
                    const { siteDir } = context;
                    const contentDir = path.join(siteDir, "docs");
                    const allMdx: string[] = [];
                    const getMdxFiles = async (dir: string) => {
                        const entries = await fs.promises.readdir(dir, { withFileTypes: true });
                        for (const entry of entries) {
                            const fullPath = path.join(dir, entry.name);
                            if (entry.isDirectory()) {
                                await getMdxFiles(fullPath);
                            } else if (entry.name.endsWith(".mdx")) {
                                const content = await fs.promises.readFile(fullPath, "utf8");
                                allMdx.push(content);
                            }
                        }
                    };
                    await getMdxFiles(contentDir);
                    return { allMdx };
                },
                // The file will only get generated with production builds, since the plugin
                // operates on the output directory
                postBuild: async props => await generateLlmsTxt(props, context),
            };
        },
        function polyfillNodeBuiltinsForFlowJS(context: any, options: any) {
            return {
                name: 'polyfillNodeBuiltinsForFlowJS',
                configureWebpack() {
                    return { resolve: { fallback: { fs: false, constants: false } } };
                },
            };
        },
        function enableSomeEnvVarsAsBuildTimeConstants() {
            return {
                name: 'enableSomeEnvVarsAsBuildTimeConstants',
                configureWebpack() {
                    return {
                        plugins: [
                            new webpack.EnvironmentPlugin({
                                INTERNAL_STATIC_DOCS:
                                    process.env.INTERNAL_STATIC_DOCS === '1' || false,
                            }),
                        ],
                    };
                },
            };
        },
        function enableMonacoEditorPlugin() {
            return {
                name: 'enableMonacoEditorPlugin',
                configureWebpack() {
                    return {
                        // https://stackoverflow.com/questions/69265357/monaco-editor-worker
                        plugins: [new MonacoWebpackPlugin()],
                    };
                },
            };
        },
        function enablePyodidePlugin() {
            return {
                name: 'enablePyodidePlugin',
                configureWebpack() {
                    return {
                        plugins: [new PyodidePlugin()],
                    };
                },
            };
        },
        function enableStyleX(context: any, options: any) {
            return {
                name: 'stylex-docusaurus',
                injectHtmlTags() {
                    return {
                        headTags: [
                            {
                                tagName: 'link',
                                attributes: {
                                    rel: 'stylesheet',
                                    href: context.baseUrl + 'stylex.css',
                                },
                            },
                        ],
                    };
                },

                configureWebpack(config: any, isServer: boolean, utils: any) {
                    const dev = config.mode === 'development';

                    return {
                        plugins: [
                            new StylexPlugin({
                                dev,
                                genConditionalClasses: true,
                                treeshakeCompensation: true,
                                unstable_moduleResolution: {
                                    type: 'commonJS',
                                    rootDir: context.siteDir,
                                },
                                filename: 'stylex.css',
                            }),
                        ],
                    };
                },
            };
        },
        function enableRedirectsPlugin() {
            return {
              name: 'enableRedirectsPlugin',
              async contentLoaded({ actions }) {
                const { addRoute } = actions;
                // Example: Add a redirect from /old-path to /new-path
                addRoute({
                  path: '/pyohio2025',
                  component: '@site/src/pages/landingPage.tsx',
                  exact: true,
                });
              },
            };
        },
    ],
    themeConfig: {
        prism: {
            theme: prismThemes.jettwaveLight,
            darkTheme: prismThemes.dracula,
            additionalLanguages: [
                'toml'
            ],
        },
        colorMode: {
            defaultMode: 'dark',
            disableSwitch: false,
            respectPrefersColorScheme: false,
        },
        navbar: {
            title: 'Pyrefly',
            logo: {
                alt: 'Pyrefly Logo',
                src: 'img/Pyrefly-Symbol.svg',
                srcDark: 'img/Pyrefly-Symbol-Invert.svg',
                height: 24, // Set a smaller height for the header logo
                width: 24,  // Maintain aspect ratio
            },
            items: getNavBarItems(),
        },
        footer: {
            links: [
                {
                    title: 'Community & Support',
                    items: [
                        {
                            label: 'Join Discord',
                            href: 'https://discord.gg/Cf7mFQtW7W',
                        },
                        {
                            label: 'Open a Github issue',
                            href: 'https://github.com/facebook/pyrefly',
                        },
                        {
                            label: 'Attend Office Hours',
                            href: 'https://discord.gg/MuFSdD7uCr?event=1389635637090713672',
                        },
                    ],
                },
                {
                    title: 'Legal',
                    // Please do not remove the privacy and terms, it's a legal requirement.
                    items: [
                        {
                            label: 'Privacy',
                            href: 'https://opensource.facebook.com/legal/privacy/',
                        },
                        {
                            label: 'Terms',
                            href: 'https://opensource.facebook.com/legal/terms/',
                        },
                        {
                            label: 'Data Policy',
                            href: 'https://opensource.facebook.com/legal/data-policy/',
                        },
                        {
                            label: 'Cookie Policy',
                            href: 'https://opensource.facebook.com/legal/cookie-policy/',
                        },
                    ],
                },
            ],
            logo: {
                alt: 'Meta Open Source Logo',
                src: 'img/meta_open_source_logo.svg',
                href: 'https://opensource.fb.com/',
            },
            // Please do not remove the credits, help to publicize Docusaurus :)
            copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.`,
        },
    },
    customFields: {
        fbRepoName: 'fbsource',
    },
    presets: [
        [
            require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
            {
                docs: {
                    routeBasePath: BasePath,
                    sidebarPath: require.resolve('./sidebars.ts'),
                    editUrl: fbContent({
                        internal:
                            'https://www.internalfb.com/code/fbsource/fbcode/pyrefly/website/',
                        external: 'https://github.com/facebook/pyrefly/edit/main/website/',
                    }),
                },
                staticDocsProject: 'Pyrefly',
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
                enableEditor: true,
                gtag:
                    process.env.INTERNAL_STATIC_DOCS === '1'
                        ? undefined
                        : { trackingID: 'G-GSX14JC495', anonymizeIP: true },
                blog: {
                    blogSidebarTitle: 'All posts',
                    blogSidebarCount: 'ALL',
                },
            },
        ],
    ],
    scripts: [
        'https://buttons.github.io/buttons.js',
        'https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js',
        '/js/code-block-buttons.js',
    ],
    stylesheets: ['/css/code-block-buttons.css']
};

export default config;
