# engine/mistralrs-core/src/kv_cache/

## Responsibility
KV-cache storage backends for autoregressive decoding: eager per-layer caches plus the paged-attention block pool, scheduler, and cache engine that multiplex many sequences over fixed GPU/CPU blocks.

## Design
- `KvCache` enum (`mod.rs`): `Normal { k: SingleCache, v: SingleCache }` | `Rotating { k, v }` (sliding-window models) | `Shared { owner }` (GQA/KV-sharing layers reuse the owner's cache); `KvCacheSnapshot` mirrors for save/restore. `CacheManager` trait (`clone_in_cache`/`clone_out_cache`/`set_none_cache`).
- `SingleCache` (`single_cache.rs`): lazy `all_data: Option<Tensor>` (allocated on first append when batch size is known; safe to clone after reset). One dtype per cache: later appends convert to the first writer's dtype (F5 fix for the F32-compute/F16-cache `slice-set` mismatch; no-op when dtypes match). Tracks `current_seq_len`/`capacity_seq_len`/`max_seq_len`.
- `RotatingCache` (`rotating_cache.rs`): ring buffer for sliding-window attention.
- `HybridCache` (`hybrid_cache.rs`): recurrent/linear halves of hybrid models (Mamba-style state pools, GDN deferred/pending transition pools, CUDA-gated snapshots).
- `paged_attention/` (sibling, primary consumer): `CacheConfig`/`CacheEngine` allocate `num_gpu_blocks` + `num_cpu_blocks` of `block_size` (default 32) tokens; `PagedAttentionScheduler` + `PagedAttentionSchedulerConfig` assign block tables per sequence (`reserved prompt chunk must fit in the KV cache` is the CTX floor guard); `MemoryGpuConfig` (`MbAmount` | `Utilization` | `BestEffortMbAmount` | `ContextSize`) sizes the pool; `block_pool` LRU + `block_hash` prefix-cache reuse; `kv_cache_manager` high-level multiplexing.

## Flow
1. Load: `calculate_cache_config` sizes GPU/CPU block counts from the memory budget → `CacheEngine` allocates block pools.
2. Prefill: scheduler reserves prompt blocks per sequence → `append` writes K/V (`SingleCache` dtype-unified).
3. Decode: block tables map logical → physical blocks; prefix hits reuse hashed blocks; eviction returns blocks to the LRU pool.
4. KV exhaustion surfaces as `KittyError::KvExhausted` with the CTX hint.

## Integration
- Consumed by: `pipeline/` loaders and `engine/` step loop, `cuda_graph` capture (graph pools), `serving::infer_policy` (CTX floors: MiniCPM5 8192, Qwen3-4B 8192/2048).
- Depends on: `candle_core` tensors/devices, `paged_attention::config` (`ModelConfigLike`: kv heads, head dims, per-layer cache topology).
