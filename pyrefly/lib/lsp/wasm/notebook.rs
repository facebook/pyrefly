/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unused)]

//! LSP Notebook Document Synchronization support
//! Based on LSP 3.17.0 specification

use lsp_types::TextDocumentIdentifier;
use lsp_types::TextDocumentItem;
use lsp_types::Url;
use lsp_types::VersionedTextDocumentIdentifier;
use lsp_types::notification::Notification;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

// ===== Core Notebook Types =====

/// A notebook document.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocument {
    /// The notebook document's URI.
    pub uri: Url,

    /// The type of the notebook.
    pub notebook_type: String,

    /// The version number of this document (it will increase after each
    /// change, including undo/redo).
    pub version: i32,

    /// Additional metadata stored with the notebook document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// The cells of a notebook.
    pub cells: Vec<NotebookCell>,
}

/// A notebook cell.
///
/// A cell's document URI must be unique across ALL notebook
/// cells and can therefore be used to uniquely identify a
/// notebook cell or the cell's text document.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCell {
    /// The cell's kind
    pub kind: NotebookCellKind,

    /// The URI of the cell's text document content.
    pub document: Url,

    /// Additional metadata stored with the cell.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// Additional execution summary information if supported by the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_summary: Option<ExecutionSummary>,
}

/// A notebook cell kind.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Copy, Deserialize, Serialize)]
pub enum NotebookCellKind {
    /// A markup-cell is formatted source that is used for display.
    Markup = 1,

    /// A code-cell is source code.
    Code = 2,
}

/// Execution summary for a notebook cell.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionSummary {
    /// A strict monotonically increasing value
    /// indicating the execution order of a cell
    /// inside a notebook.
    pub execution_order: u32,

    /// Whether the execution was successful or
    /// not if known by the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
}

// ===== Notebook Filters =====

/// A notebook cell text document filter denotes a cell text
/// document by different properties.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellTextDocumentFilter {
    /// A filter that matches against the notebook
    /// containing the notebook cell. If a string
    /// value is provided it matches against the
    /// notebook type. '*' matches every notebook.
    pub notebook: NotebookDocumentFilterOrString,

    /// A language id like `python`.
    ///
    /// Will be matched against the language id of the
    /// notebook cell document. '*' matches every language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NotebookDocumentFilterOrString {
    String(String),
    Filter(NotebookDocumentFilter),
}

/// A notebook document filter denotes a notebook document by
/// different properties.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentFilter {
    /// The type of the enclosing notebook.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notebook_type: Option<String>,

    /// A Uri scheme, like `file` or `untitled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,

    /// A glob pattern.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

// ===== Notebook Identifiers =====

/// A literal to identify a notebook document in the client.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentIdentifier {
    /// The notebook document's URI.
    pub uri: Url,
}

/// A versioned notebook document identifier.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VersionedNotebookDocumentIdentifier {
    /// The version number of this notebook document.
    pub version: i32,

    /// The notebook document's URI.
    pub uri: Url,
}

// ===== Notebook Change Events =====

/// A change describing how to move a `NotebookCell`
/// array from state S to S'.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellArrayChange {
    /// The start offset of the cell that changed.
    pub start: u32,

    /// The deleted cells
    pub delete_count: u32,

    /// The new cells, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cells: Option<Vec<NotebookCell>>,
}

/// Changes to notebook cells structure.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellStructureChange {
    /// The change to the cell array.
    pub array: NotebookCellArrayChange,

    /// Additional opened cell text documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub did_open: Option<Vec<TextDocumentItem>>,

    /// Additional closed cell text documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub did_close: Option<Vec<TextDocumentIdentifier>>,
}

/// Changes to the text content of notebook cells.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellTextContentChange {
    pub document: VersionedTextDocumentIdentifier,
    pub changes: Vec<Value>,
}

/// Changes to notebook cells.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellChanges {
    /// Changes to the cell structure to add or remove cells.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure: Option<NotebookCellStructureChange>,

    /// Changes to notebook cells properties like its
    /// kind, execution summary or metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<NotebookCell>>,

    /// Changes to the text content of notebook cells.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_content: Option<Vec<NotebookCellTextContentChange>>,
}

/// A change event for a notebook document.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentChangeEvent {
    /// The changed meta data if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    /// Changes to cells
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cells: Option<NotebookCellChanges>,
}

// ===== Capabilities =====

/// Notebook specific client capabilities.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentSyncClientCapabilities {
    /// Whether implementation supports dynamic registration. If this is
    /// set to `true` the client supports the new
    /// `(NotebookDocumentSyncRegistrationOptions & NotebookDocumentSyncOptions)`
    /// return value for the corresponding server capability as well.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_registration: Option<bool>,

    /// The client supports sending execution summary data per cell.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_summary_support: Option<bool>,
}

/// Selector for notebook cells.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookCellSelector {
    pub language: String,
}

/// Selector for notebook documents.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentSelector {
    /// The notebook to be synced. If a string
    /// value is provided it matches against the
    /// notebook type. '*' matches every notebook.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notebook: Option<NotebookDocumentFilterOrString>,

    /// The cells of the matching notebook to be synced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cells: Option<Vec<NotebookCellSelector>>,
}

/// Enum for notebook selector variants.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NotebookSelector {
    ByNotebook {
        notebook: Notebook,
        #[serde(skip_serializing_if = "Option::is_none")]
        cells: Option<Vec<NotebookCellSelector>>,
    },
    ByCells {
        #[serde(skip_serializing_if = "Option::is_none")]
        notebook: Option<Notebook>,
        cells: Vec<NotebookCellSelector>,
    },
}

/// Enum for notebook type identification.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Notebook {
    String(String),
    Filter(NotebookDocumentFilter),
}

/// Options specific to a notebook plus its cells
/// to be synced to the server.
///
/// If a selector provides a notebook document
/// filter but no cell selector all cells of a
/// matching notebook document will be synced.
///
/// If a selector provides no notebook document
/// filter but only a cell selector all notebook
/// documents that contain at least one matching
/// cell will be synced.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentSyncOptions {
    /// The notebooks to be synced
    pub notebook_selector: Vec<NotebookDocumentSelector>,

    /// Whether save notification should be forwarded to
    /// the server. Will only be honored if mode === `notebook`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save: Option<bool>,
}

/// Registration options specific to a notebook.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotebookDocumentSyncRegistrationOptions {
    #[serde(flatten)]
    pub sync_options: NotebookDocumentSyncOptions,

    /// The id used to register the request. The id can be used to deregister
    /// the request again. See also Registration#id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

// ===== Notification Params =====

/// The params sent in an open notebook document notification.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenNotebookDocumentParams {
    /// The notebook document that got opened.
    pub notebook_document: NotebookDocument,

    /// The text documents that represent the content
    /// of a notebook cell.
    pub cell_text_documents: Vec<TextDocumentItem>,
}

/// The params sent in a change notebook document notification.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidChangeNotebookDocumentParams {
    /// The notebook document that did change. The version number points
    /// to the version after all provided changes have been applied.
    pub notebook_document: VersionedNotebookDocumentIdentifier,

    /// The actual changes to the notebook document.
    ///
    /// The change describes single state change to the notebook document.
    /// So it moves a notebook document, its cells and its cell text document
    /// contents from state S to S'.
    ///
    /// To mirror the content of a notebook using change events use the
    /// following approach:
    /// - start with the same initial content
    /// - apply the 'notebookDocument/didChange' notifications in the order
    ///   you receive them.
    pub change: NotebookDocumentChangeEvent,
}

/// The params sent in a save notebook document notification.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidSaveNotebookDocumentParams {
    /// The notebook document that got saved.
    pub notebook_document: NotebookDocumentIdentifier,
}

/// The params sent in a close notebook document notification.
///
/// @since 3.17.0
#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidCloseNotebookDocumentParams {
    /// The notebook document that got closed.
    pub notebook_document: NotebookDocumentIdentifier,

    /// The text documents that represent the content
    /// of a notebook cell that got closed.
    pub cell_text_documents: Vec<TextDocumentIdentifier>,
}

// ===== Notification Definitions =====

/// The `notebookDocument/didOpen` notification.
///
/// @since 3.17.0
#[derive(Debug)]
pub enum DidOpenNotebookDocument {}

impl Notification for DidOpenNotebookDocument {
    type Params = DidOpenNotebookDocumentParams;
    const METHOD: &'static str = "notebookDocument/didOpen";
}

/// The `notebookDocument/didChange` notification.
///
/// @since 3.17.0
#[derive(Debug)]
pub enum DidChangeNotebookDocument {}

impl Notification for DidChangeNotebookDocument {
    type Params = DidChangeNotebookDocumentParams;
    const METHOD: &'static str = "notebookDocument/didChange";
}

/// The `notebookDocument/didSave` notification.
///
/// @since 3.17.0
#[derive(Debug)]
pub enum DidSaveNotebookDocument {}

impl Notification for DidSaveNotebookDocument {
    type Params = DidSaveNotebookDocumentParams;
    const METHOD: &'static str = "notebookDocument/didSave";
}

/// The `notebookDocument/didClose` notification.
///
/// @since 3.17.0
#[derive(Debug)]
pub enum DidCloseNotebookDocument {}

impl Notification for DidCloseNotebookDocument {
    type Params = DidCloseNotebookDocumentParams;
    const METHOD: &'static str = "notebookDocument/didClose";
}
