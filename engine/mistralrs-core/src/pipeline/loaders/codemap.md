# engine/mistralrs-core/src/pipeline/loaders/

## Responsibility
Model-loading layer: loader-type registry (`NormalLoaderType`, incl. `K2Horizon`), GGUF native path, ISQ/UQFF quantization flows, checkpoint inventory sizing, and automatic device mapping with GPU-first audible placement.

## Design
- Loader dispatch (`mod.rs`): `Loader` trait (`load_model_from_hf` / `load_model_from_path`); `DeviceMappedModelLoader` trait (activation/weight sizing, `num_layers`, `model_config`, `get_device_layers` defaulting to `auto_device_map`); `AutoDeviceMapQuantization` (ISQ vs weight-source pack factors).
- Type registry (`normal_loaders.rs`): `NormalLoaderType` enum with serde renames (`k2_horizon`), causal-LM name ↔ config-file detection (`K2HorizonForCausalLM`), per-type `K2HorizonLoader` (runtime_config caps `max_position_embeddings`, `build_k2_horizon` from `rope.dimension_count`, honest refusals for projector/MoE markers).
- Auto map (`auto_device_map.rs`, GPU-first audible): `get_device_layers` fits whole model on first device when possible, else fills per-device via pure `count_fitting_layers` (overflow-safe). `GpuPreference` (`Offered`/`Probed`/`Absent`): no offered accelerator → system probe (`cuda_if_available(0)`, metal where compiled) prepended GPU-first; none found → loud `WARNING … mapping all layers to CPU` (never a silent default). Every decision emits `[mistralrs-auto-map]` to stderr+tracing with `gpu_layers`/`cpu_layers`/`reason` (`fits-on-gpu` | `no-accelerator-cpu-only` | `gpu-fits-zero-spill-to-cpu` | `gpu-overflow-spill-to-cpu`) + measured free MiB. Paged-attention KV budgets (`MemoryGpuConfig::MbAmount`/`Utilization`/`ContextSize`) feed the estimate; CPU-spill with paged-attn falls back to a non-paged re-map.
- GGUF path (`../gguf.rs` → `load_native_normal`): text-only native load; GGUF VRAM precheck lives in `crate::engine::gguf_loader` (`vram_precheck`, `file_bytes`, `resolve_ctx_len`) and fails fast with byte counts before touching the engine.
- Sizing (`checkpoint_inventory.rs`): per-layer byte inventory from checkpoint tensor names for mapping estimates.

## Flow
1. `GgufModelBuilder` (no explicit mapping) → `DeviceMapSetting::Auto` → `load_native_normal` → `get_all_similar_devices(&device)` → `get_device_layers` (GPU preference + `[mistralrs-auto-map]` log) → `DeviceMapSetting::Map` → `into_mapper` → `LayerDeviceMapper` (`Layers a-b: <device>` lines).
2. Explicit `KITTY_INFER_GPU_LAYERS=n` bypasses auto via `DeviceMapMetadata::from_num_device_layers` (spill verbatim).
3. Mixed GPU+CPU mapping disables paged-attention with a warning (no CPU paged support).

## Integration
- Consumed by: `pipeline/normal.rs` + `pipeline/gguf.rs` loaders, `tuning.rs` auto-tune, `GgufModelBuilder`/`TextModelBuilder` (facade crate).
- Depends on: `device_map` (`DeviceMapMetadata`, mappers, peer access), `paged_attention` (`calculate_cache_config`, `device_memory_cap`), `MemoryUsage` queries, `ModelConfigLike`.
