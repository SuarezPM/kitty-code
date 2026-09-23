//! Vision-models directory (kitty-infer salvage).
//!
//! Upstream keeps one directory per multimodal family here. kitty-infer is a
//! text-only engine: the ONLY residents are text towers needed by the normal
//! (text) loaders — Qwen3.5 dense + MoE — plus the processor-config readers
//! used by the text chat-template path (`pipeline::paths` / `processing`).
//! Everything visual, auditory, or diffusion-shaped was cut (see F0-RECON.md).

pub(crate) mod preprocessor_config;
pub(crate) mod processor_config;
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_5_moe;
