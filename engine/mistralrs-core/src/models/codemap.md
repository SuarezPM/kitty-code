# engine/mistralrs-core/src/models/ — alcance: `k2_horizon.rs` + `mod.rs` + registro

## Responsibility
Ported K2-Horizon dense text tower (`general.architecture = k2-horizon`, IFM/K2-Horizon-3.7B): config deserialization, weight loading, and forward pass behind the `NormalModel` interface. Registered in `mod.rs` (`pub(crate) mod k2_horizon`) and in the loader registry as `NormalLoaderType::K2Horizon` (`K2HorizonForCausalLM`, serde `k2_horizon`).

## Design
- Port of `K2HorizonForCausalLM`, derived from the llama tower but NOT reusable as llama: explicit `head_dim = 128` (not `hidden/heads` = 80), pre-norm decoder, GQA 32 Q / 8 KV, full-head NeoX-style RoPE with θ=1e7, SwiGLU MLP (10240), final RMSNorm, untied lm_head.
- `GroupedRmsNorm`: variance computed PER GROUP (`n_groups = 2`), upcast float32, `broadcast_mul` by rsqrt — faithful to reference `K2HorizonRMSNorm`, not equivalent to full-width RMSNorm (a double-inversion bug here once produced uniform logits; fixed).
- Layout: 100% llama-style tensors (`blk.N.*`, `token_embd/output/output_norm`), norms F32, rest BF16; 327 tensors (36×9+3). No projector, no biases/qk_norm/sliding-window, no MoE (`num_experts = 0`).
- Honest refusals at config synthesis: projector tensor markers and MoE/MoVA (`expert_count > 0`) bail instead of running degraded. Native window 524288 OOMs the KV batch on 8GB → `runtime_config` caps `max_position_embeddings` (harness CTX=4096).
- Remaining towers in this directory (`llama.rs`, `qwen3.rs`, `mistral.rs`, `mixtral.rs`, `phi*.rs`, `gemma*.rs`, `deepseek*.rs`, `glm4*.rs`, `gpt_oss.rs`, `qwen2/3_moe`, `smollm3`, `lfm2*`, `starcoder2`, `granite`, `hunyuan*`, `quantized_llama`, …) are inherited upstream towers, untouched by the port.

## Flow
1. GGUF header → arch `k2-horizon` → `NormalLoaderType::from_causal_lm_name` → `K2Horizon` → `K2HorizonLoader`.
2. `K2HorizonLoader::load` → `build_k2_horizon` (head_dim from `rope.dimension_count`, `runtime_config` CTX cap) → device-mapped `ShardedVarBuilder` weights → `K2HorizonForCausalLM`.
3. Forward: `input_layernorm → GQA → +residual → post_attention_layernorm → SwiGLU → +residual` ×36 → final norm → lm_head; KV via `EitherCache`/paged layers.

## Integration
- Consumed by: `pipeline/loaders/normal_loaders.rs` (`K2HorizonLoader`, `runtime_config`), `gguf/normal_registry` (RopePairing Adjacent), `serving::infer_policy` (sidecar `K2-Horizon-3.7B/tokenizer.json`, default F16 dtype).
- Depends on: `layers` (embedding, `CausalMasker`, `RotaryEmbedding`, `Sdpa`, `Mlp`, `Activation`), `device_map` (`DeviceMapper`, sharded builders), `paged_attention` (`PagedAttention`, `ModelConfigMetadata`), `mistralrs_quant` parallel layers.
