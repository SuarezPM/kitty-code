use super::attention_backend::{AttentionBackend, AttentionBackendKind, AttentionLayerSpec};
use crate::flashinfer::FlashInferAttentionBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    Standard,
    StandardNoFlashInfer,
    FlashInferHnd,
    Mla {
        kv_lora_rank: usize,
        kpe_head_dim: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheTopology {
    layer_to_owner: Vec<usize>,
}

impl KvCacheTopology {
    pub fn all_own(num_layers: usize) -> Self {
        Self {
            layer_to_owner: (0..num_layers).collect(),
        }
    }

    pub fn from_layer_owners(layer_to_owner: Vec<usize>) -> Self {
        debug_assert!(layer_to_owner
            .iter()
            .all(|owner| *owner < layer_to_owner.len()));
        Self { layer_to_owner }
    }

    pub fn owner_for_layer(&self, layer_idx: usize) -> usize {
        self.layer_to_owner
            .get(layer_idx)
            .copied()
            .unwrap_or(layer_idx)
    }

    pub fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        self.owner_for_layer(layer_idx) == layer_idx
    }

    pub fn has_shared_layers(&self) -> bool {
        self.layer_to_owner
            .iter()
            .enumerate()
            .any(|(layer_idx, owner)| layer_idx != *owner)
    }

    pub fn group_ids(&self) -> Vec<u32> {
        if !self.has_shared_layers() {
            return vec![0];
        }

        let mut group_ids = self
            .layer_to_owner
            .iter()
            .map(|owner| u32::try_from(*owner).expect("KV cache owner index exceeds u32"))
            .collect::<Vec<_>>();
        group_ids.sort_unstable();
        group_ids.dedup();
        if group_ids.is_empty() {
            vec![0]
        } else {
            group_ids
        }
    }
}

fn flashinfer_supported_for_model<M: ModelConfigLike + ?Sized>(config: &M) -> bool {
    let backend = FlashInferAttentionBackend;
    (0..config.num_layers())
        .all(|layer_idx| backend.supports_layer(config.attention_layer_spec(layer_idx)))
}

fn select_attention_backend<M: ModelConfigLike + ?Sized>(config: &M) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if flashinfer_supported_for_model(config) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_attention_backend_for_layer<M: ModelConfigLike + ?Sized>(
    config: &M,
    layer_idx: usize,
) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if backend.supports_layer(config.attention_layer_spec(layer_idx)) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_kv_cache_layout<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind() {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
            }
        }
    }
}

fn select_kv_cache_layout_for_layer<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
    layer_idx: usize,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind_for_layer(layer_idx) {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
            }
        }
    }
}

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn num_attn_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_attn_heads()
    }
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads()
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim()
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim()
    }
    fn has_kv_cache_sharing(&self) -> bool {
        false
    }
    fn kv_cache_topology(&self) -> KvCacheTopology {
        KvCacheTopology::all_own(self.num_layers())
    }
    fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        !self.has_kv_cache_sharing()
            || self
                .kv_cache_topology()
                .uses_own_kv_cache_for_layer(layer_idx)
    }
    /// False for layers with no paged KV cache at all, e.g. the linear/recurrent half of a hybrid model.
    fn layer_has_paged_kv_cache(&self, _layer_idx: usize) -> bool {
        true
    }
    /// Paged KV elements per token for one layer, or `None` when the layer keeps no paged cache.
    fn layer_kv_cache_elements_per_token(&self, layer_idx: usize) -> Option<usize> {
        self.layer_has_paged_kv_cache(layer_idx)
            .then(|| self.kv_cache_elements_per_token())
    }
    /// Summed over every layer holding a paged cache. Size against this rather than multiplying a
    /// per-layer average by a layer count: the two diverge as soon as layers differ from each other.
    fn total_kv_cache_elements_per_token(&self) -> usize {
        (0..self.num_layers())
            .filter_map(|idx| self.layer_kv_cache_elements_per_token(idx))
            .sum()
    }
    fn kv_cache_group_ids(&self) -> Vec<u32> {
        if self.has_kv_cache_sharing() {
            self.kv_cache_topology().group_ids()
        } else {
            vec![0]
        }
    }
    fn attention_layer_spec(&self, layer_idx: usize) -> AttentionLayerSpec {
        AttentionLayerSpec {
            q_heads: self.num_attn_heads_for_layer(layer_idx),
            kv_heads: self.num_kv_heads_for_layer(layer_idx),
            k_head_dim: self.k_head_dim_for_layer(layer_idx),
            v_head_dim: self.v_head_dim_for_layer(layer_idx),
        }
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        select_attention_backend(self)
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        select_attention_backend_for_layer(self, layer_idx)
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(KvCacheLayout::Standard, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(KvCacheLayout::Standard, self, layer_idx)
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        2 * self.num_kv_heads() * self.k_head_dim().max(self.v_head_dim())
    }

    fn prefix_prefill_attention_features(
        &self,
        _layer_idx: usize,
    ) -> Option<PrefixPrefillAttentionFeatures> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixPrefillAttentionFeatures {
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_softcap: bool,
    pub has_sliding_window: bool,
}

/// Per-layer architecture tag → whether the layer keeps a paged KV cache.
///
/// Covers every `layer_types` vocabulary the GGUF synthesizer emits:
/// granite (`attention`/`mamba`), LFM2 (`full_attention`/`conv`), Qwen
/// (`full_attention`/`linear_attention`) and sliding-window dense models
/// (`sliding_attention` — windowed layers still page their KV, only the
/// linear/recurrent half of a hybrid skips the cache entirely).
/// Unknown tags → `None`: callers keep the dense assumption instead of
/// guessing (over-allocating is safe, under-allocating panics the cache).
pub fn layer_tag_has_paged_kv_cache(tag: &str) -> Option<bool> {
    match tag {
        "attention" | "full_attention" | "sliding_attention" => Some(true),
        "mamba" | "linear_attention" | "conv" => Some(false),
        _ => None,
    }
}

/// Paged-KV mask for GDN hybrids (Qwen3-Next, Qwen3.5): every Nth layer
/// (1-based) is full attention with a paged cache, the rest are linear.
/// `None` when the interval is unknown (`0`) — the caller keeps dense.
pub fn full_attention_interval_paged_mask(
    num_layers: usize,
    full_attention_interval: usize,
) -> Option<Vec<bool>> {
    if num_layers == 0 || full_attention_interval == 0 {
        return None;
    }
    Some(
        (0..num_layers)
            .map(|i| (i + 1) % full_attention_interval == 0)
            .collect(),
    )
}

/// Derive the per-layer paged-KV mask from a GGUF-synthesized model config.
///
/// `layer_types` (explicit per-layer kinds) wins when present and fully
/// known; otherwise `full_attention_interval` (Qwen3-Next/Qwen3.5 GDN rule)
/// applies. Returns `None` when the arch is unrecognized — absent signals,
/// unknown tags, or a length mismatch against `num_layers` — so the caller
/// keeps the dense all-paged assumption and `Utilization` stays the only
/// fallback sizing for such archs.
pub fn hybrid_paged_mask_from_config_json(
    num_layers: usize,
    config_json: &str,
) -> Option<Vec<bool>> {
    let value: serde_json::Value = serde_json::from_str(config_json).ok()?;
    if let Some(names) = value.get("layer_types").and_then(|v| v.as_array()) {
        if names.len() != num_layers {
            return None;
        }
        let mut mask = Vec::with_capacity(num_layers);
        for name in names {
            mask.push(layer_tag_has_paged_kv_cache(name.as_str()?)?);
        }
        return Some(mask);
    }
    if let Some(interval) = value
        .get("full_attention_interval")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
    {
        return full_attention_interval_paged_mask(num_layers, interval);
    }
    None
}

/// Wrap `base` with the hybrid paged-KV mask when `config_json` names a
/// genuinely hybrid layout (F1a: Ornith/qwen35 8/32, granite `mamba`,
/// LFM2 `conv`), else return `base` untouched.
///
/// Same rule as the auto-map estimate path: length matches, every tag
/// known, at least one paged and one pageless layer. Anything else
/// (dense, sliding-window-only, unrecognized) keeps the dense assumption
/// so windowing/layout metadata can never be lost in a rebuild.
pub fn maybe_wrap_hybrid(
    base: ModelConfigMetadata,
    config_json: &str,
) -> Box<dyn ModelConfigLike> {
    let num_layers = base.num_layers();
    let Some(mask) = hybrid_paged_mask_from_config_json(num_layers, config_json) else {
        return Box::new(base);
    };
    if mask.len() != num_layers || !mask.contains(&true) || !mask.contains(&false) {
        return Box::new(base);
    }
    Box::new(HybridPagedKvCacheConfig::new(base, mask))
}

#[derive(Clone)]
pub struct ModelConfigMetadata {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_kv_heads: usize,
    pub num_attn_heads: usize,
    pub sliding_window: Option<usize>,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_cache_layout: KvCacheLayout,
}

impl ModelConfigLike for ModelConfigMetadata {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.k_head_dim
    }
    fn v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend(self)
            }
        }
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend_for_layer(self, layer_idx)
            }
        }
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(self.kv_cache_layout, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(self.kv_cache_layout, self, layer_idx)
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        match self.kv_cache_layout() {
            KvCacheLayout::Standard
            | KvCacheLayout::StandardNoFlashInfer
            | KvCacheLayout::FlashInferHnd => {
                2 * self.num_kv_heads * self.k_head_dim.max(self.v_head_dim)
            }
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
        }
    }
}

/// Base config plus a per-layer mask of which layers keep a paged KV cache.
///
/// Hybrid models interleave attention layers with linear/recurrent ones that hold no paged cache.
/// Sizing the cache off `num_layers` there over-allocates by the ratio of attention to total layers.
#[derive(Clone)]
pub struct HybridPagedKvCacheConfig {
    base: ModelConfigMetadata,
    paged_layers: Vec<bool>,
    prefix_prefill_attention_features: Vec<Option<PrefixPrefillAttentionFeatures>>,
}

impl HybridPagedKvCacheConfig {
    pub fn new(base: ModelConfigMetadata, paged_layers: Vec<bool>) -> Self {
        let num_layers = base.num_layers();
        Self {
            base,
            paged_layers,
            prefix_prefill_attention_features: vec![None; num_layers],
        }
    }

    /// Count of layers keeping a paged cache (the KV sizing layer count).
    pub fn paged_layer_count(&self) -> usize {
        self.paged_layers.iter().filter(|b| **b).count()
    }

    pub fn with_uniform_prefix_prefill_attention_features(
        mut self,
        features: PrefixPrefillAttentionFeatures,
    ) -> Self {
        for (features_slot, has_paged_cache) in self
            .prefix_prefill_attention_features
            .iter_mut()
            .zip(&self.paged_layers)
        {
            if *has_paged_cache {
                *features_slot = Some(features);
            }
        }
        self
    }
}

impl ModelConfigLike for HybridPagedKvCacheConfig {
    fn max_seq_len(&self) -> usize {
        self.base.max_seq_len()
    }
    fn num_layers(&self) -> usize {
        self.base.num_layers()
    }
    fn hidden_size(&self) -> usize {
        self.base.hidden_size()
    }
    fn num_kv_heads(&self) -> usize {
        self.base.num_kv_heads()
    }
    fn num_attn_heads(&self) -> usize {
        self.base.num_attn_heads()
    }
    fn k_head_dim(&self) -> usize {
        self.base.k_head_dim()
    }
    fn v_head_dim(&self) -> usize {
        self.base.v_head_dim()
    }
    fn num_attn_heads_for_layer(&self, layer_idx: usize) -> usize {
        self.base.num_attn_heads_for_layer(layer_idx)
    }
    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        self.base.num_kv_heads_for_layer(layer_idx)
    }
    fn k_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.base.k_head_dim_for_layer(layer_idx)
    }
    fn v_head_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.base.v_head_dim_for_layer(layer_idx)
    }
    fn has_kv_cache_sharing(&self) -> bool {
        self.base.has_kv_cache_sharing()
    }
    fn kv_cache_topology(&self) -> KvCacheTopology {
        self.base.kv_cache_topology()
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        self.base.attention_backend_kind()
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        self.base.attention_backend_kind_for_layer(layer_idx)
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.base.kv_cache_layout()
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        self.base.kv_cache_layout_for_layer(layer_idx)
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        self.base.kv_cache_elements_per_token()
    }
    fn layer_has_paged_kv_cache(&self, layer_idx: usize) -> bool {
        self.paged_layers.get(layer_idx).copied().unwrap_or(true)
    }
    fn prefix_prefill_attention_features(
        &self,
        layer_idx: usize,
    ) -> Option<PrefixPrefillAttentionFeatures> {
        if !self.layer_has_paged_kv_cache(layer_idx) {
            return None;
        }
        self.prefix_prefill_attention_features
            .get(layer_idx)
            .copied()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionBackendKind, KvCacheLayout, KvCacheTopology, ModelConfigLike, ModelConfigMetadata,
    };

    #[test]
    fn all_own_topology_uses_single_prefix_cache_group() {
        let topology = KvCacheTopology::all_own(4);
        assert_eq!(topology.group_ids(), vec![0]);
        assert!(!topology.has_shared_layers());
        assert!(topology.uses_own_kv_cache_for_layer(3));
    }

    #[test]
    fn shared_topology_groups_by_owner_layers() {
        let topology = KvCacheTopology::from_layer_owners(vec![0, 1, 0, 1, 4]);
        assert_eq!(topology.group_ids(), vec![0, 1, 4]);
        assert!(topology.has_shared_layers());
        assert!(topology.uses_own_kv_cache_for_layer(1));
        assert!(!topology.uses_own_kv_cache_for_layer(2));
        assert_eq!(topology.owner_for_layer(3), 1);
    }

    #[test]
    fn hybrid_totals_skip_layers_without_a_paged_cache() {
        let base = ModelConfigMetadata {
            max_seq_len: 32_768,
            num_layers: 4,
            hidden_size: 896,
            num_kv_heads: 2,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        let per_layer = base.kv_cache_elements_per_token();
        assert_eq!(base.total_kv_cache_elements_per_token(), 4 * per_layer);

        let hybrid = super::HybridPagedKvCacheConfig::new(base, vec![true, false, false, true]);
        assert_eq!(hybrid.total_kv_cache_elements_per_token(), 2 * per_layer);
        assert_eq!(hybrid.layer_kv_cache_elements_per_token(1), None);
    }

    #[test]
    fn hybrid_kv_bytes_are_quarter_of_dense_for_ornith() {
        use super::{full_attention_interval_paged_mask, maybe_wrap_hybrid};
        use candle_core::DType;

        // Ornith-1.5-9B shape: 32 GDN-hybrid layers, every 4th paged (8/32).
        let base = ModelConfigMetadata {
            max_seq_len: 131_072,
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 256,
            v_head_dim: 256,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        let dtype_size = DType::F16.size_in_bytes();
        let dense_bytes = base.total_kv_cache_elements_per_token() * dtype_size;

        let mask = full_attention_interval_paged_mask(32, 4).expect("known hybrid interval");
        let hybrid = super::HybridPagedKvCacheConfig::new(base.clone(), mask);
        let hybrid_bytes = hybrid.total_kv_cache_elements_per_token() * dtype_size;

        // BYTES, not blocks: the hybrid pays a quarter of the dense KV.
        assert_eq!(hybrid.paged_layer_count(), 8);
        assert_eq!(dense_bytes, 4 * hybrid_bytes);

        // Loader path: the same JSON wraps to hybrid, dense JSON stays dense.
        let ornith_json =
            serde_json::json!({"num_hidden_layers": 32, "full_attention_interval": 4}).to_string();
        let wrapped = maybe_wrap_hybrid(base.clone(), &ornith_json);
        assert_eq!(
            wrapped.total_kv_cache_elements_per_token() * dtype_size,
            hybrid_bytes
        );
        let dense_json = serde_json::json!({"num_hidden_layers": 32}).to_string();
        let kept = maybe_wrap_hybrid(base, &dense_json);
        assert_eq!(
            kept.total_kv_cache_elements_per_token() * dtype_size,
            dense_bytes
        );
    }

    #[test]
    fn unsupported_flashinfer_group_uses_standard_paged_attention() {
        let config = ModelConfigMetadata {
            max_seq_len: 32_768,
            num_layers: 24,
            hidden_size: 1152,
            num_kv_heads: 2,
            num_attn_heads: 18,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: KvCacheLayout::Standard,
        };

        assert_eq!(
            config.attention_backend_kind(),
            AttentionBackendKind::Standard
        );
        assert_eq!(config.kv_cache_layout(), KvCacheLayout::Standard);
    }

    #[test]
    fn hybrid_interval_mask_matches_ornith_eight_of_thirty_two() {
        use super::{full_attention_interval_paged_mask, layer_tag_has_paged_kv_cache};
        use super::{HybridPagedKvCacheConfig, KvCacheLayout, ModelConfigMetadata};

        // Tag vocabulary: every synthesizer spelling maps, unknowns refuse.
        assert_eq!(layer_tag_has_paged_kv_cache("attention"), Some(true));
        assert_eq!(layer_tag_has_paged_kv_cache("full_attention"), Some(true));
        assert_eq!(
            layer_tag_has_paged_kv_cache("sliding_attention"),
            Some(true)
        );
        assert_eq!(layer_tag_has_paged_kv_cache("mamba"), Some(false));
        assert_eq!(
            layer_tag_has_paged_kv_cache("linear_attention"),
            Some(false)
        );
        assert_eq!(layer_tag_has_paged_kv_cache("conv"), Some(false));
        assert_eq!(layer_tag_has_paged_kv_cache("mystery"), None);

        // Ornith-1.5-9B shape: qwen35 GDN hybrid, 32 layers, every 4th paged.
        let mask = full_attention_interval_paged_mask(32, 4).expect("known interval");
        assert_eq!(mask.len(), 32);
        let paged: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.then_some(i))
            .collect();
        assert_eq!(paged, vec![3, 7, 11, 15, 19, 23, 27, 31]);

        // Totals scale with the 8 paged layers, not the 32 physical ones.
        let base = ModelConfigMetadata {
            max_seq_len: 131_072,
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 256,
            v_head_dim: 256,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        let per_layer = base.kv_cache_elements_per_token();
        assert_eq!(base.total_kv_cache_elements_per_token(), 32 * per_layer);
        let hybrid = HybridPagedKvCacheConfig::new(base, mask);
        assert_eq!(hybrid.paged_layer_count(), 8);
        assert_eq!(hybrid.total_kv_cache_elements_per_token(), 8 * per_layer);

        // Degenerate intervals refuse instead of building an empty cache.
        assert!(full_attention_interval_paged_mask(32, 0).is_none());
        assert!(full_attention_interval_paged_mask(0, 4).is_none());
    }

    #[test]
    fn hybrid_mask_from_config_json_covers_synthesized_kinds() {
        use super::hybrid_paged_mask_from_config_json;

        // Granite-style explicit layer types (mamba layers hold no paged KV).
        let granite = serde_json::json!({
            "architectures": ["GraniteMoeHybridForCausalLM"],
            "num_hidden_layers": 4,
            "layer_types": ["attention", "mamba", "mamba", "attention"],
        });
        assert_eq!(
            hybrid_paged_mask_from_config_json(4, &granite.to_string()),
            Some(vec![true, false, false, true])
        );

        // Ornith-style: no layer_types, full_attention_interval drives the mask.
        let ornith = serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
        });
        let mask =
            hybrid_paged_mask_from_config_json(32, &ornith.to_string()).expect("known hybrid");
        assert_eq!(mask.iter().filter(|b| **b).count(), 8);

        // Explicit layer_types win over the interval when both are present.
        let both = serde_json::json!({
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"],
            "full_attention_interval": 1,
        });
        assert_eq!(
            hybrid_paged_mask_from_config_json(2, &both.to_string()),
            Some(vec![true, false])
        );

        // Unrecognized arch: absent signals, unknown tags, or length
        // mismatch all refuse — the caller keeps the dense assumption.
        assert_eq!(hybrid_paged_mask_from_config_json(32, "{}"), None);
        assert_eq!(
            hybrid_paged_mask_from_config_json(32, "not json at all"),
            None
        );
        let unknown_tag = serde_json::json!({
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "flux_capacitor"],
        });
        assert_eq!(
            hybrid_paged_mask_from_config_json(2, &unknown_tag.to_string()),
            None
        );
        let bad_len = serde_json::json!({
            "num_hidden_layers": 4,
            "layer_types": ["attention", "mamba"],
        });
        assert_eq!(
            hybrid_paged_mask_from_config_json(4, &bad_len.to_string()),
            None
        );
    }
}
