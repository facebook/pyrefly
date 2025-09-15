/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import * as stylex from '@stylexjs/stylex';

const SUPPORTED_VERSIONS = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13', '3.14'] as const;

export interface PythonVersionSelectorProps {
    selectedVersion: string;
    onVersionChange: (version: string) => void;
    loading: boolean;
}

export default function PythonVersionSelector({
    selectedVersion,
    onVersionChange,
    loading,
}: PythonVersionSelectorProps): React.ReactElement {
    return (
        <div {...stylex.props(styles.container)}>
            <select
                {...stylex.props(styles.versionSelector)}
                value={selectedVersion}
                onChange={(e) => onVersionChange(e.target.value)}
                disabled={loading}
                aria-label="Select Python version"
            >
                {SUPPORTED_VERSIONS.map((version) => (
                    <option key={version} value={version} {...stylex.props(styles.option)}>
                        Python {version}
                    </option>
                ))}
            </select>
        </div>
    );
}

const styles = stylex.create({
    container: {
        padding: '8px 12px',
        border: '1px solid var(--color-background-secondary)',
        borderRadius: '24px',
        backgroundColor: 'var(--color-background)',
        color: 'inherit',
        fontFamily: 'inherit',
        cursor: 'pointer',
        outline: 'none',
        minWidth: '120px',
        boxShadow: '0 2px 8px var(--color-shadow)',
       '@media (min-width: 769px)': {
            ':hover': {
                background: 'var(--color-background)',
                transform: 'translateY(-1px)',
                boxShadow: '0 4px 12px var(--color-shadow-hovered)',
            },
        },
        ':disabled': {
            opacity: 0.6,
            cursor: 'not-allowed',
        },
    },
    versionSelector: {
        border: 'none',
        outline: 'none',
        width: '100%',
        height: '100%',
        backgroundColor: 'transparent',
        paddingRight: '8px',
        boxSizing: 'border-box',
        display: 'block'
    },
    option: {
        color: 'var(--color-text)',
        backgroundColor: 'var(--color-background)',
    },
});
