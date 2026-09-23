//! K2-Horizon dense text tower (`general.architecture = k2-horizon`).
//!
//! Port of `K2HorizonForCausalLM` (IFM/K2-Horizon-3.7B, dense 36-layer
//! checkpoint: `mlp_only_layers` covers every layer, experts = 0):
//! pre-norm decoder, GQA (32 Q / 8 KV, explicit `head_dim = 128`), full
//! rotary (`rope_theta = 1e7`), SwiGLU MLP, untied lm_head.
//!
//! Deltas vs the llama tower it derives from:
//! - [`GroupedRmsNorm`]: variance is computed PER GROUP (`n_groups = 2`,
//!   `hidden % groups == 0`), in float32, exactly like the reference
//!   `K2HorizonRMSNorm`. This is NOT equivalent to full-width RMSNorm.
//! - `head_dim` is explicit (128), not `hidden / heads` (which would be 80).
//! - No Q/K norms, no biases, no sliding window, no MoE.
//!
//! Scope: text only. A checkpoint carrying projector/MoVA tensors is
//! refused at config synthesis (unsupported tensors), never silently run.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::layers_masker::CausalMaskConfig;
use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{
        embedding_with_legacy_tied_uqff, Activation, CausalMasker, Mlp, RotaryEmbedding, Sdpa,
    },
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, ModelForwardContext, NormalCache, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, false);
serde_default_fn!(usize, norm_groups_default, 2);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit per-head dimension (128 for K2-Horizon, NOT hidden/heads).
    pub head_dim: usize,
    /// Grouped-RMSNorm group count (2 for K2-Horizon).
    #[serde(default = "norm_groups_default")]
    pub norm_num_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

/// Grouped RMSNorm: per-group variance in float32, like `K2HorizonRMSNorm`.
#[derive(Debug, Clone)]
pub struct GroupedRmsNorm {
    eps: f64,
    n_groups: usize,
    weight: Tensor,
}

impl GroupedRmsNorm {
    pub fn new(size: usize, n_groups: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        if n_groups == 0 || size % n_groups != 0 {
            candle_core::bail!(
                "grouped RMSNorm needs hidden_size % n_groups == 0, got size={size} groups={n_groups}"
            );
        }
        let w = vb.get(size, "weight")?;
        Ok(Self {
            eps,
            n_groups,
            weight: w,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let dims = x.dims().to_vec();
        let group_size = dims[dims.len() - 1] / self.n_groups;
        let mut grouped_shape = dims.clone();
        grouped_shape.pop();
        grouped_shape.push(self.n_groups);
        grouped_shape.push(group_size);
        let h = x.to_dtype(DType::F32)?.reshape(grouped_shape)?;
        let var = h.sqr()?.mean_keepdim(D::Minus1)?;
        // NOTE: multiply by rsqrt — dividing by recip(sqrt(var)) would
        // double-invert and collapse the signal (F5 lesson, verified with a
        // standalone candle repro: normalized RMS came out 0.0014 = var).
        let h = h.broadcast_mul(&(var + self.eps)?.sqrt()?.recip()?)?;
        let h = h.reshape(dims)?.to_dtype(in_dtype)?;
        h.broadcast_mul(&self.weight)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let (q, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let rope_positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
        let (q, k) = self.rotary_emb.forward(&q, &k, &rope_positions)?;
        let metadata = ctx.paged_layer(layer_idx);

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(ctx.flash_params()),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(!matches!(attention_mask, AttentionMask::None));
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(ctx.flash_params()),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(ctx.flash_params()),
                    &self.sdpa_params,
                )?
            }
        };

        y = if !matches!(attention_mask, AttentionMask::None) {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        let res = self.o_proj.forward(&y)?;
        Ok(res)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        rope: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = cfg.num_attention_heads * cfg.head_dim;
        let size_kv = cfg.num_key_value_heads * cfg.head_dim;
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.head_dim,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim: cfg.head_dim,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }
}

struct Block {
    rms_1: GroupedRmsNorm,
    attn: CausalSelfAttention,
    rms_2: GroupedRmsNorm,
    mlp: Box<dyn MlpLayer>,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, attention_mask, kv_cache, ctx, layer_idx)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            rope,
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let rms_1 = GroupedRmsNorm::new(
            cfg.hidden_size,
            cfg.norm_num_groups,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = GroupedRmsNorm::new(
            cfg.hidden_size,
            cfg.norm_num_groups,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp: Box::new(mlp),
        })
    }
}

pub struct K2Horizon {
    wte: Arc<dyn QuantMethod>,
    blocks: Vec<Block>,
    ln_f: GroupedRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl K2Horizon {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        // Resolved by `is_gptx_for` from the GGUF rope pairing (Adjacent →
        // false, like Llama): selects the rotary convention matching the
        // stored Q/K layout. Hardcoding this destroyed attention (flat
        // logits) — never assume it per-arch.
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if cfg.norm_num_groups == 0 || cfg.hidden_size % cfg.norm_num_groups != 0 {
            candle_core::bail!(
                "K2-Horizon needs hidden_size % norm_num_groups == 0, got hidden_size={} groups={}",
                cfg.hidden_size,
                cfg.norm_num_groups
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let wte = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb_lm_head.clone(), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            wte.clone()
        };
        let ln_f = GroupedRmsNorm::new(
            cfg.hidden_size,
            cfg.norm_num_groups,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let location = device.location();
            if ropes.contains_key(&location) {
                continue;
            }
            ropes.insert(
                location,
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }
        let blocks: Vec<_> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(cfg.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(i)?;
            Block::load(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
            )
        })?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            dtype,
            kv_cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.wte.embedding_forward(input_ids, self.dtype)?,
            ctx,
        )
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let cache = &mut self.kv_cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            x.dtype(),
            &CausalMaskConfig::default(),
        )?;
        let mask = if ctx.is_first_prompt_chunk() {
            mask
        } else {
            AttentionMask::None
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            let mask_for_layer = &mask.get(x.device());
            x = block.forward(&x, mask_for_layer, &mut cache[block_idx], ctx, block_idx)?;
        }
        let x = x.to_device(&self.device)?;
        let x = self.ln_f.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for K2Horizon {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl crate::speculative::SpeculativeTargetMixin for K2Horizon {}

// Dense text tower: AnyMoE/LoRA flows are not supported (honest default:
// the mixin bails instead of pretending).
impl AnyMoeBaseModelMixin for K2Horizon {}

impl NormalModel for K2Horizon {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn supports_packed_prefill(&self) -> bool {
        true
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}
