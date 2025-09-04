/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Things we disagree with
#![allow(clippy::new_without_default)]

use std::collections::HashMap;
use pyrefly::playground::Playground;
use pyrefly::playground::Position;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct State(Playground);

#[wasm_bindgen]
impl State {
    #[wasm_bindgen(constructor)]
    pub fn new(version: String) -> Self {
        let playground =
            Playground::new(Some(&version)).unwrap_or_else(|e| wasm_bindgen::throw_str(&e));
        Self(playground)
    }

    #[wasm_bindgen(js_name=updateSource)]
    pub fn update_source(&mut self, source: String) {
        self.0.update_source(source)
    }

    #[wasm_bindgen(js_name=updateSandboxFiles)]
    pub fn update_sandbox_files(&mut self, files: JsValue) -> Result<(), JsValue> {
        let files_map: HashMap<String, String> = serde_wasm_bindgen::from_value(files)
            .map_err(|e| JsValue::from_str(&format!("Failed to deserialize files: {}", e)))?;
        self.0.update_sandbox_files(files_map);
        Ok(())
    }

    #[wasm_bindgen(js_name=updateSingleFile)]
    pub fn update_single_file(&mut self, filename: String, content: String) {
        self.0.update_single_file(filename, content);
    }

    #[wasm_bindgen(js_name=setActiveFile)]
    pub fn set_active_file(&mut self, filename: String) {
        self.0.set_active_file(&filename);
    }

    #[wasm_bindgen(js_name=getErrors)]
    pub fn get_errors(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.0.get_errors()).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen(js_name=queryType)]
    pub fn query_type(&mut self, line: i32, column: i32) -> JsValue {
        self.0
            .query_type(Position::new(line, column))
            .map(|result| serde_wasm_bindgen::to_value(&result).unwrap())
            .unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen(js_name=gotoDefinition)]
    pub fn goto_definition(&mut self, line: i32, column: i32) -> JsValue {
        self.0
            .goto_definition(Position::new(line, column))
            .map(|result| serde_wasm_bindgen::to_value(&result).unwrap())
            .unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen(js_name=autoComplete)]
    pub fn autocomplete(&mut self, line: i32, column: i32) -> JsValue {
        serde_wasm_bindgen::to_value(&self.0.autocomplete(Position::new(line, column))).unwrap()
    }

    #[wasm_bindgen(js_name=inlayHint)]
    pub fn inlay_hint(&mut self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.0.inlay_hint()).unwrap()
    }
}
