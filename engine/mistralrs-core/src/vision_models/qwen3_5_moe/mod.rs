//! Qwen3.5-MoE text tower (kitty-infer salvage).
//!
//! Same rule as `super::qwen3_5`: text tower only. The multimodal model
//! struct, vision tower, and processor re-exports were cut.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) mod config;
mod text;

pub(crate) use config::TextConfig;
pub(crate) use text::Qwen3_5MoeTextModel;
