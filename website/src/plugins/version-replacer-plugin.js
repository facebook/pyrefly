/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const fs = require('fs');
const path = require('path');

// Plugin to replace "rev: main" in installation.mdx with the current version from version.bzl
module.exports = function versionReplacerPlugin() {
    return {
        name: 'version-replacer-plugin',
        async loadContent() {
            // Read the version from version.bzl
            const versionFilePath = path.resolve(
                process.cwd(),
                '../version.bzl'
            );
            const versionFileContent = fs.readFileSync(versionFilePath, 'utf8');

            // Extract the version using regex
            const versionMatch = versionFileContent.match(
                /VERSION\s*=\s*"([^"]+)"/
            );
            if (!versionMatch) {
                throw new Error('Could not find VERSION in version.bzl');
            }

            const version = versionMatch[1];
            return { version };
        },

        async contentLoaded({ content, actions }) {
            const { version } = content;

            // Path to the installation.mdx file
            const installationPath = path.resolve(
                process.cwd(),
                'docs/installation.mdx'
            );

            // Read the installation.mdx file
            let installationContent = fs.readFileSync(installationPath, 'utf8');

            // Replace "rev: main" with the actual version
            installationContent = installationContent.replace(
                /rev: main/g,
                `rev: ${version}`
            );

            // Write the updated content back to the file
            fs.writeFileSync(installationPath, installationContent);

            console.log(
                `Replaced "rev: main" with "rev: ${version}" in installation.mdx`
            );
        },
    };
};
