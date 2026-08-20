/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use proc_macro2::TokenStream;
use quote::quote;
use syn::Attribute;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;
use syn::Lit;
use syn::LitStr;
use syn::Token;
use syn::parse_macro_input;

pub(crate) fn derive_config_keys(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_config_keys_impl(&input) {
        Ok(x) => x.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// How a field is treated by serde when serializing, as far as the config-key
/// set is concerned.
enum FieldKind {
    /// Serialized under a single key (the resolved `rename`/`rename_all` name),
    /// plus any deprecated `#[serde(alias = "...")]` spellings serde also accepts.
    Key { key: String, aliases: Vec<String> },
    /// A `#[serde(flatten)]` sub-struct whose keys and aliases are spliced in via
    /// its own `ConfigKeys` impl.
    Flatten,
    /// Not serialized (`#[serde(skip)]`, `#[serde(skip_serializing)]`, or
    /// `#[config_keys(skip)]` for the flatten catch-all).
    Skip,
}

fn derive_config_keys_impl(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "ConfigKeys does not support generic types",
        ));
    }
    let rename_all = serde_rename_all(&input.attrs)?;
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    input,
                    "ConfigKeys requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "ConfigKeys can only be derived for structs",
            ));
        }
    };

    // Emit one statement per serialized field, in declaration order, so the
    // resulting key order matches serde's serialized order (which the "did you
    // mean" ranking uses as a tiebreaker).
    let mut key_stmts = Vec::new();
    let mut alias_stmts = Vec::new();
    for field in fields {
        match field_kind(field, &rename_all)? {
            FieldKind::Skip => {}
            FieldKind::Key { key, aliases } => {
                key_stmts.push(quote! { keys.push(#key); });
                for alias in aliases {
                    alias_stmts.push(quote! { aliases.push((#alias, #key)); });
                }
            }
            FieldKind::Flatten => {
                let ty = &field.ty;
                key_stmts.push(quote! {
                    keys.extend(<#ty as pyrefly_util::config_keys::ConfigKeys>::config_keys());
                });
                alias_stmts.push(quote! {
                    aliases.extend(<#ty as pyrefly_util::config_keys::ConfigKeys>::config_key_aliases());
                });
            }
        }
    }

    Ok(quote! {
        impl pyrefly_util::config_keys::ConfigKeys for #name {
            fn config_keys() -> Vec<&'static str> {
                let mut keys: Vec<&'static str> = Vec::new();
                #(#key_stmts)*
                keys
            }

            fn config_key_aliases() -> Vec<(&'static str, &'static str)> {
                let mut aliases: Vec<(&'static str, &'static str)> = Vec::new();
                #(#alias_stmts)*
                aliases
            }
        }
    })
}

fn field_kind(field: &syn::Field, rename_all: &Option<String>) -> syn::Result<FieldKind> {
    if config_keys_skip(&field.attrs)? {
        return Ok(FieldKind::Skip);
    }
    let mut rename = None;
    let mut aliases = Vec::new();
    let mut skip = false;
    let mut flatten = false;
    for attr in &field.attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                rename = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("alias") {
                aliases.push(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("skip") || meta.path.is_ident("skip_serializing") {
                skip = true;
            } else if meta.path.is_ident("flatten") {
                flatten = true;
            } else if meta.input.peek(Token![=]) {
                // Consume and ignore values of serde options we don't care about
                // (default, skip_serializing_if, ...) to keep the parser in sync.
                let _: Lit = meta.value()?.parse()?;
            }
            Ok(())
        })?;
    }
    // A skipped field is never serialized, so serde ignores its aliases too;
    // dropping them here keeps the alias set in sync with serde's behavior.
    if skip {
        return Ok(FieldKind::Skip);
    }
    if flatten {
        return Ok(FieldKind::Flatten);
    }
    let key = match rename {
        Some(rename) => rename,
        None => {
            let name = field.ident.as_ref().unwrap().to_string();
            apply_rename_all(rename_all, &name, field)?
        }
    };
    Ok(FieldKind::Key { key, aliases })
}

/// Read the struct-level `#[serde(rename_all = "...")]`, if present.
fn serde_rename_all(attrs: &[Attribute]) -> syn::Result<Option<String>> {
    let mut rule = None;
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename_all") {
                rule = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.input.peek(Token![=]) {
                let _: Lit = meta.value()?.parse()?;
            }
            Ok(())
        })?;
    }
    Ok(rule)
}

/// Apply a serde `rename_all` rule to a snake_case field name. Only the rules
/// actually used by pyrefly's config structs are supported; anything else is a
/// compile error rather than a silent wrong guess.
fn apply_rename_all(rule: &Option<String>, name: &str, field: &syn::Field) -> syn::Result<String> {
    match rule.as_deref() {
        None => Ok(name.to_owned()),
        // serde's kebab-case rule on a snake_case identifier is exactly this.
        Some("kebab-case") => Ok(name.replace('_', "-")),
        Some(other) => Err(syn::Error::new_spanned(
            field,
            format!("ConfigKeys does not support rename_all = \"{other}\""),
        )),
    }
}

/// Whether a field carries `#[config_keys(skip)]` (the flatten catch-all).
fn config_keys_skip(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut skip = false;
    for attr in attrs {
        if !attr.path().is_ident("config_keys") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(meta.error("unsupported `config_keys` attribute"))
            }
        })?;
    }
    Ok(skip)
}
