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
        <select
            {...stylex.props(styles.versionSelector)}
            value={selectedVersion}
            onChange={(e) => onVersionChange(e.target.value)}
            disabled={loading}
            aria-label="Select Python version"
        >
            {SUPPORTED_VERSIONS.map((version) => (
                <option key={version} value={version}>
                    Python {version}
                </option>
            ))}
        </select>
    );
}

const styles = stylex.create({
    versionSelector: {
        padding: '8px 12px',
        border: '1px solid var(--color-background-secondary)',
        borderRadius: '4px',
        backgroundColor: 'var(--color-background)',
        color: 'var(--color-text)',
        fontSize: '14px',
        fontFamily: 'inherit',
        cursor: 'pointer',
        outline: 'none',
        minWidth: '120px',
        ':hover': {
            borderColor: 'var(--color-primary)',
        },
        ':focus': {
            borderColor: 'var(--color-primary)',
            boxShadow: '0 0 0 2px var(--color-primary-alpha)',
        },
        ':disabled': {
            opacity: 0.6,
            cursor: 'not-allowed',
        },
    },
});