/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

module.exports = {
  presets: [
    require.resolve('@docusaurus/core/lib/babel/preset'),
    require.resolve('@babel/preset-flow'),
  ],
  plugins: ['babel-plugin-syntax-hermes-parser'],
  parserOpts: {flow: 'all'},
};
