/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// Mock for @docusaurus/theme-common
export const useColorMode = jest.fn().mockReturnValue({
    colorMode: 'light',
    setColorMode: jest.fn(),
});

export const useThemeConfig = jest.fn().mockReturnValue({});

export const usePrismTheme = jest.fn().mockReturnValue({
    plain: {},
    styles: [],
});
