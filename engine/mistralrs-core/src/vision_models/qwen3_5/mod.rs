//! Qwen3.5 text tower (kitty-infer salvage).
//!
//! Upstream `qwen3_5/mod.rs` implemented the full multimodal model (vision
//! tower, encoder cache, `MultimodalModel`). kitty-infer is text-only: this
//! shim keeps the text tower (`Qwen3_5TextModel`), its config, and the MTP /
//! packed-GDN helpers it needs. Anything visual was cut (see F0-RECON.md).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod config;
pub(crate) mod mtp;
pub(crate) mod packed_gdn;
mod text;

pub(crate) use config::TextConfig;
pub(crate) use text::Qwen3_5TextModel;
