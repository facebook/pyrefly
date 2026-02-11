/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pyrefly_python::ignore::Tool;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::de::MapAccess;
use serde::de::Visitor;
use starlark_map::small_set::SmallSet;

use crate::error_kind::ErrorKind;
use crate::error_kind::Severity;

/// Represents overrides for errors to emit when collecting/printing errors.
/// Not all error kinds are required to be defined in this map. Any that are missing
/// will use the default severity associated with that error kind.
#[derive(Debug, PartialEq, Eq, Serialize, Clone, Default)]
pub struct ErrorDisplayConfig(HashMap<ErrorKind, Severity>);

impl ErrorDisplayConfig {
    pub fn new(config: HashMap<ErrorKind, Severity>) -> Self {
        Self(config)
    }

    /// Gets the severity for the given `ErrorKind`. If the value isn't
    /// found, then assume it should be the default for that error kind.
    pub fn severity(&self, kind: ErrorKind) -> Severity {
        self.0
            .get(&kind)
            .copied()
            .unwrap_or_else(|| kind.default_severity())
    }

    pub fn set_error_severity(&mut self, kind: ErrorKind, severity: Severity) {
        self.0.insert(kind, severity);
    }
}

impl<'de> Deserialize<'de> for ErrorDisplayConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ErrorDisplayConfigVisitor;

        impl<'de> Visitor<'de> for ErrorDisplayConfigVisitor {
            type Value = ErrorDisplayConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a map of error kinds to severity level")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut config = HashMap::new();

                while let Some(key) = map.next_key::<ErrorKind>()? {
                    let severity = match map.next_value::<serde_json::Value>()? {
                        serde_json::Value::Bool(false) => Severity::Ignore,
                        serde_json::Value::Bool(true) => {
                            let default_severity = key.default_severity();
                            if default_severity > Severity::Ignore {
                                default_severity
                            } else {
                                Severity::Error
                            }
                        }
                        serde_json::Value::String(s) => {
                            serde_json::from_str::<Severity>(&format!("\"{s}\""))
                                .map_err(serde::de::Error::custom)?
                        }
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "expected string or boolean, found {other}"
                            )));
                        }
                    };
                    config.insert(key, severity);
                }

                Ok(ErrorDisplayConfig::new(config))
            }
        }

        deserializer.deserialize_map(ErrorDisplayConfigVisitor)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ErrorConfig<'a> {
    pub display_config: &'a ErrorDisplayConfig,
    pub ignore_errors_in_generated_code: bool,
    pub enabled_ignores: SmallSet<Tool>,
}

impl<'a> ErrorConfig<'a> {
    pub fn new(
        display_config: &'a ErrorDisplayConfig,
        ignore_errors_in_generated_code: bool,
        enabled_ignores: SmallSet<Tool>,
    ) -> Self {
        Self {
            display_config,
            ignore_errors_in_generated_code,
            enabled_ignores,
        }
    }
}
