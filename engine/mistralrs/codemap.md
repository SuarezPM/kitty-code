# engine/mistralrs/ (facade crate)

## Responsibility
Public SDK facade over `mistralrs-core`: ergonomic builders, request/message types, and curated re-exports. Thin mapping layer — no model code, no kernels, no serving logic lives here.

## Design
- Builders: `GgufModelBuilder` (`gguf.rs`: `new(dir, files)` + `with_device` / `with_device_mapping` / `with_dtype` / `with_max_model_len` / `with_paged_attn` / `with_tokenizer_json` / `with_logging` → `build()`), `TextModelBuilder`/`UqffTextModelBuilder` (`text_model.rs`, incl. `PagedAttentionMetaBuilder`), auto-detect `ModelBuilder` (`auto_model.rs`), multi-model `AnyModelBuilder`/`MultiModelBuilder` (`model_builder_trait.rs`). Unset mapping defaults to `DeviceMapSetting::Auto`; unset device resolves via `best_device` (`model.rs`: `cuda_if_available(0)` unless `force_cpu`).
- `Model` (`model.rs`): handle returned by builders; `send_chat_request` and related request paths.
- Re-exports (`lib.rs`): request/response/sampling/tool/config/speculative/device-map/topology/loader/token-source types from core, `candle_core` (`DType`, `Device`, `Tensor`), `best_device`, `MISTRALRS_VERSION`; `pub use mistralrs_core::serving` (line 301) exposes the kitty serving surface; `pub mod core` documents low-level internals for advanced use.
- Features (`Cargo.toml [features]`): `cuda` (+`cutile`, `cudnn`), `metal`, `flash-attn`, `accelerate`, `mkl`, `nccl`, `ring`, `code-execution`, … — each forwards to `mistralrs-core/<same>`; CPU-only is the default build.

## Flow
1. Caller picks a builder → sets device/mapping/dtype/paged-attn → `build().await` → `build_*_pipeline` in `model_builder_trait.rs` (device resolution → `DeviceMapSetting` → loader `load_model_from_*` → scheduler + engine config → `Model`).
2. `kitty-infer` embedded path: `GgufModelBuilder::new(parent, vec![file])` with paged-attn `ContextSize(ctx)` and optional explicit spill mapping.

## Integration
- Consumed by: `crates/kitty-runtime` (`KittyInferClient`), examples, downstream SDK users.
- Depends on: `mistralrs-core` (engine, pipelines, serving, device_map), `mistralrs-quant`, `candle-core`, `tokenizers`, `clap` (CLI surface in `mistralrs-cli`, excluded from default workspace).
