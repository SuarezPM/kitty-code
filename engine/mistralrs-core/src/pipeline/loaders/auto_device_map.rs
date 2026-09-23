use std::fmt::{self, Display};

use crate::paged_attention::{
    calculate_cache_config, device_memory_cap, CacheMemoryReservations, MemoryGpuConfig,
    ModelConfigLike, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

use super::DeviceMappedModelLoader;

fn saturating_memory_sum<const N: usize>(parts: [usize; N]) -> usize {
    parts
        .into_iter()
        .fold(0usize, |total, part| total.saturating_add(part))
}

fn checked_memory_sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
}

fn post_load_memory_config(
    requested: MemoryGpuConfig,
    pre_load_budget: MemoryGpuConfig,
    resolve_utilization_after_load: bool,
) -> MemoryGpuConfig {
    match requested {
        MemoryGpuConfig::Utilization(_) if !resolve_utilization_after_load => pre_load_budget,
        _ => requested,
    }
}

#[derive(Clone, Debug)]
pub(crate) enum NonMappedSubModel {
    Vision,
    Audio,
}

impl Display for NonMappedSubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonMappedSubModel::Vision => write!(f, "vision"),
            NonMappedSubModel::Audio => write!(f, "audio"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AutoDeviceMapParams {
    Text {
        max_seq_len: usize,
        max_batch_size: usize,
    },
    Multimodal {
        max_seq_len: usize,
        max_batch_size: usize,
        max_image_shape: (usize, usize),
        max_num_images: usize,
    },
}

impl AutoDeviceMapParams {
    pub fn maybe_promote_to_multimodal(&self) -> Self {
        match *self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape: (
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                ),
                max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            },
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            },
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Text { max_seq_len, .. } | Self::Multimodal { max_seq_len, .. } => *max_seq_len,
        }
    }

    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Text { max_batch_size, .. } | Self::Multimodal { max_batch_size, .. } => {
                *max_batch_size
            }
        }
    }
}

impl Display for AutoDeviceMapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => write!(
                f,
                "text[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}]"
            ),
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => write!(
                f,
                "multimodal[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}, max_image_shape: {max_image_shape:?}, max_num_images: {max_num_images}]"
            ),
        }
    }
}

impl AutoDeviceMapParams {
    // Default max sequence length for memory estimation when not specified
    pub const DEFAULT_MAX_SEQ_LEN: usize = 4 * 1024;
    pub const DEFAULT_MAX_BATCH_SIZE: usize = 1;
    pub const DEFAULT_MAX_NUM_IMAGES: usize = 1;
    pub const DEFAULT_MAX_IMAGE_LENGTH: usize = 1024;

    pub fn default_text() -> Self {
        Self::Text {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
        }
    }

    pub fn default_multimodal() -> Self {
        Self::Multimodal {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            max_image_shape: (
                Self::DEFAULT_MAX_IMAGE_LENGTH,
                Self::DEFAULT_MAX_IMAGE_LENGTH,
            ),
        }
    }
}

fn calculate_key_block_shape(
    model_config: &dyn ModelConfigLike,
    dtype: DType,
    block_size: usize,
) -> (usize, usize, usize, usize) {
    let element_size = dtype.size_in_bytes();
    let x = 16 / element_size;
    (
        model_config.num_kv_heads(),
        model_config.k_head_dim() / x,
        block_size,
        x,
    )
}

fn calculate_value_block_shape(
    model_config: &dyn ModelConfigLike,
    block_size: usize,
) -> (usize, usize, usize) {
    (
        model_config.num_kv_heads(),
        model_config.v_head_dim(),
        block_size,
    )
}

macro_rules! b_to_mb {
    ($x:expr) => {
        $x / (1024 * 1024)
    };
}

/// GPU preference for the auto decision (D1 forense: auto elegía CPU en
/// silencio aun con GPU libre). `Offered` = el caller ya traía acelerador;
/// `Probed` = no traía ninguno y el probe del sistema encontró uno (se
/// antepone: GPU-first); `Absent` = no hay acelerador en ningún lado y el
/// fallback a CPU se avisa a gritos (nunca por defecto silencioso).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuPreference {
    Offered,
    Probed,
    Absent,
}

/// Truth table of the preference. Pure and unit-tested.
fn gpu_preference(offered_has_accelerator: bool, probed_has_accelerator: bool) -> GpuPreference {
    if offered_has_accelerator {
        GpuPreference::Offered
    } else if probed_has_accelerator {
        GpuPreference::Probed
    } else {
        GpuPreference::Absent
    }
}

/// System accelerator probe for the auto path. `cuda_if_available` compiles
/// on every backend (CPU-only builds just report CPU); metal is probed only
/// where it is compiled in.
fn probe_system_accelerator() -> Option<Device> {
    if let Ok(cuda) = Device::cuda_if_available(0) {
        if !cuda.is_cpu() {
            return Some(cuda);
        }
    }
    #[cfg(feature = "metal")]
    if let Ok(metal) = Device::new_metal(0) {
        return Some(metal);
    }
    None
}

/// Loud notice for the CPU-only fallback. Pure and unit-tested: the text
/// must name CPU explicitly (the bug was the silence, not the fallback).
fn absent_accelerator_notice(offered: &str) -> String {
    format!(
        "[mistralrs-auto-map] WARNING: no GPU accelerator offered ({offered}) and none found on this system — mapping all layers to CPU. Explicit fallback, never a silent default: pass a CUDA device or check CUDA_VISIBLE_DEVICES/driver if a GPU exists."
    )
}

/// Pure inner fill of the auto-map loop: how many next-first `(weight, kv)`
/// pairs fit under `cap` with `used` bytes already spent. Overflow-safe;
/// 0 when not even one layer fits. The productive loop below uses it (single
/// implementation); the tests cover it directly.
fn count_fitting_layers(sizes: &[usize], kv: &[usize], mut used: usize, cap: usize) -> usize {
    let mut count = 0;
    for (sz, kv) in sizes.iter().zip(kv.iter()) {
        let Some(delta) = sz.checked_add(*kv) else {
            break;
        };
        let Some(next_used) = used.checked_add(delta) else {
            break;
        };
        if next_used > cap {
            break;
        }
        used = next_used;
        count += 1;
    }
    count
}

/// Audible reason for the final split. Pure and unit-tested.
fn decision_reason(
    gpu_layers: usize,
    cpu_layers: usize,
    preference: GpuPreference,
) -> &'static str {
    if cpu_layers == 0 {
        "fits-on-gpu"
    } else if gpu_layers == 0 {
        if preference == GpuPreference::Absent {
            "no-accelerator-cpu-only"
        } else {
            "gpu-fits-zero-spill-to-cpu"
        }
    } else {
        "gpu-overflow-spill-to-cpu"
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
/// Core logic for automatic device mapping
pub fn get_device_layers(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    mut layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &AutoDeviceMapParams,
    paged_attn_config: Option<&mut PagedAttentionConfig>,
) -> Result<DeviceMapMetadata> {
    let mapped_max = loader.mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();
    let non_mapped_max =
        loader.non_mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();

    // GPU preference: auto must never silently default to CPU (D1 forense).
    // If the caller offered no accelerator, probe the system and prepend
    // every similar device GPU-first, so every layer that fits the pre-check
    // lands on GPU and CPU only takes real overflow. Every branch logs loudly
    // (stderr + tracing), never tracing-only.
    let offered_gpu = devices.iter().any(|d| !d.is_cpu());
    let probed: Vec<Device> = if offered_gpu {
        Vec::new()
    } else {
        match probe_system_accelerator() {
            Some(gpu) => {
                crate::device_map::get_all_similar_devices(&gpu).unwrap_or_else(|_| vec![gpu])
            }
            None => Vec::new(),
        }
    };
    let preference = gpu_preference(offered_gpu, !probed.is_empty());
    if preference == GpuPreference::Absent {
        let notice = absent_accelerator_notice(&format!(
            "[{}]",
            devices.iter().map(|d| d.device_pretty_repr()).join(", ")
        ));
        eprintln!("{notice}");
        warn!("{notice}");
    } else if preference == GpuPreference::Probed {
        let msg = format!(
            "[mistralrs-auto-map] no accelerator in offered devices; system probe found {} — prepending GPU-first",
            probed.iter().map(|d| d.device_pretty_repr()).join(", ")
        );
        eprintln!("{msg}");
        info!("{msg}");
    }
    let owned_devices: Vec<Device>;
    let devices: &[Device] = if probed.is_empty() {
        devices
    } else {
        owned_devices = probed.into_iter().chain(devices.iter().cloned()).collect();
        &owned_devices
    };

    let mut layer_sizes_backup = if paged_attn_config.is_some() {
        Some(layer_sizes_in_bytes.clone())
    } else {
        None
    };

    let mut remaining = total_model_size_in_bytes;
    let max_seq_len = match params {
        AutoDeviceMapParams::Text { max_seq_len, .. }
        | AutoDeviceMapParams::Multimodal { max_seq_len, .. } => *max_seq_len,
    };
    let max_batch_size = match params {
        AutoDeviceMapParams::Text { max_batch_size, .. }
        | AutoDeviceMapParams::Multimodal { max_batch_size, .. } => *max_batch_size,
    };

    let model_cfg = loader.model_config(config)?;
    let has_paged_attn = paged_attn_config.is_some();
    let base_device_memory_reservation_bytes = paged_attn_config
        .as_ref()
        .map_or(0, |config| config.base_device_memory_reservation_bytes);
    let kv_cache_elems = match paged_attn_config {
        Some(cfg) => {
            // The mapping estimate is bounded independently from the post-load memory mode.
            let requested_mem_gpu = cfg.mem_gpu;
            let effective_mem_gpu = match requested_mem_gpu {
                MemoryGpuConfig::MbAmount(user_mb) => {
                    // Clamp user's KV budget to available memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb.min(user_mb))
                }
                MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::BestEffortMbAmount {
                        target_mb: budget_mb.min(target_mb),
                        min_mb,
                    }
                }
                MemoryGpuConfig::Utilization(f) => {
                    // Prevent overallocation when total_memory > available_memory
                    // (e.g., unified memory systems, other GPU processes using VRAM).
                    // Cap the KV budget so model + activations + KV fits within
                    // the device capacity derived from *available* memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let occupied = saturating_memory_sum([
                        remaining,
                        act_overhead,
                        base_device_memory_reservation_bytes,
                    ]);
                    let budget_mb =
                        ((cap as f64 * f as f64) as usize).saturating_sub(occupied) / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb)
                }
                // ContextSize passes through to calculate_cache_config.
                other => other,
            };
            info!(
                "Reserving {} MB on the primary device and {} MB on mapped devices for activations (predicted).",
                b_to_mb!(non_mapped_max.max(mapped_max)),
                b_to_mb!(mapped_max),
            );
            cfg.reserve_activation_memory(non_mapped_max.max(mapped_max), mapped_max);
            if base_device_memory_reservation_bytes > 0 {
                info!(
                    "Reserving {} MB on the primary device for post-load model components.",
                    base_device_memory_reservation_bytes.div_ceil(1024 * 1024)
                );
            }
            // Re-resolve utilization after recurrent serving state is allocated.
            cfg.mem_gpu = post_load_memory_config(
                requested_mem_gpu,
                effective_mem_gpu,
                cfg.resolve_memory_utilization_after_load,
            );

            let cache = calculate_cache_config(
                effective_mem_gpu,
                CacheMemoryReservations::default(),
                Some(cfg.block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE)),
                dtype,
                cfg.cache_type,
                &*model_cfg,
                &devices[0],
                &devices.iter().map(|d| Some(d.clone())).collect::<Vec<_>>(),
                true,
                Some(total_model_size_in_bytes),
                Some(max_seq_len * max_batch_size),
            )?;
            let key_shape = calculate_key_block_shape(&*model_cfg, dtype, cache.block_size);
            let key_sz =
                cache.num_gpu_blocks * key_shape.0 * key_shape.1 * key_shape.2 * key_shape.3;
            let val_shape = calculate_value_block_shape(&*model_cfg, cache.block_size);
            let val_sz = cache.num_gpu_blocks * val_shape.0 * val_shape.1 * val_shape.2;
            key_sz + val_sz
        }
        None => {
            let key_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.k_head_dim(),
            ];
            let val_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.v_head_dim(),
            ];
            key_shape.iter().product::<usize>() + val_shape.iter().product::<usize>()
        }
    };
    // Per paged layer; hybrid models leave the recurrent/linear layers out of the cache entirely.
    let kv_cache_bytes = kv_cache_elems * dtype.size_in_bytes();
    let kv_bytes_for_layer = |idx: usize| {
        if model_cfg.layer_has_paged_kv_cache(idx) {
            kv_cache_bytes
        } else {
            0
        }
    };
    // Paged layers past the mapped stack (an MTP head) are charged to the non-mapped device.
    let extra_kv_bytes = (num_layers..model_cfg.num_layers())
        .map(kv_bytes_for_layer)
        .sum::<usize>();

    // prepare available memory per device, CPU fallback last (unless unified memory)
    let has_unified_memory = devices.iter().any(crate::utils::normal::is_integrated_gpu);

    let mut avail = Vec::new();
    for dev in devices {
        let a = MemoryUsage.query(dev)?.available();
        avail.push((a, dev.clone()));
    }
    // On unified memory systems (iGPUs), GPU and CPU share the same physical RAM.
    // Don't add CPU as a fallback device since it would double-count memory.
    if !has_unified_memory {
        let a = MemoryUsage.query(&Device::Cpu)?.available();
        avail.push((a, Device::Cpu));
    }

    avail.reverse();
    layer_sizes_in_bytes.reverse();

    let mut mappings = Vec::new();
    info!("Using automatic device mapping parameters: {params}.");
    if let Some(subs) = loader.non_mapped_sub_models_for_config(config)? {
        let (_, last) = avail.last().unwrap();
        info!(
            "The following sub-models will not be device mapped and will be loaded on {}: {}",
            last.device_pretty_repr(),
            subs.iter().map(|x| x.to_string()).join(", ")
        );
    }

    let mut ordinal = 0;
    let mut layer = 0;
    let avail_copy = avail.clone();
    let mut includes_cpu = false;
    while remaining > 0 && !avail.is_empty() {
        let (avail_bytes, dev) = avail
            .pop()
            .context("No more devices to map to. The model does not fit on this system.")?;

        // For GPU/accelerators: keep a small dynamic safety reserve to avoid OOMs
        let cap = device_memory_cap(avail_bytes, &dev);
        if ordinal == 0
            && checked_memory_sum([
                non_mapped_max.max(mapped_max),
                non_mapped_size_in_bytes,
                base_device_memory_reservation_bytes,
            ])
            .is_none_or(|required| required > cap)
        {
            anyhow::bail!(
                "Primary device {} cannot fit its fixed model components, activations, and post-load reservation within {} MB of usable capacity.",
                dev.device_pretty_repr(),
                b_to_mb!(cap),
            );
        }

        // Algorithm is to check the following:
        // 1) (no mapping) if *everything* fits on the first dev (non mapped and mapped)
        // 2) if the mapped activations plus remaining fits on the nth device
        // 3) common case, iteratively find the optimal amount of layers to put on the nth device
        //   - if this is the first dev: must hold the non-mapped act and non-mapped model
        //   - otherwise, must hold the mapped act
        let remaining_kv_bytes = (layer..num_layers).map(kv_bytes_for_layer).sum::<usize>();
        let required_whole_capacity = if ordinal == 0 {
            checked_memory_sum([
                remaining,
                non_mapped_max.max(mapped_max),
                remaining_kv_bytes,
                extra_kv_bytes,
                base_device_memory_reservation_bytes,
            ])
        } else {
            checked_memory_sum([remaining, mapped_max, remaining_kv_bytes])
        };

        let layers_on_dev = if required_whole_capacity.is_some_and(|required| cap >= required) {
            remaining = 0;
            num_layers - layer
        } else {
            let mut used = mapped_max;
            let mut used_weight_bytes = 0usize;
            if ordinal == 0 {
                used = checked_memory_sum([
                    used.max(non_mapped_max),
                    non_mapped_size_in_bytes,
                    extra_kv_bytes,
                    base_device_memory_reservation_bytes,
                ])
                .unwrap_or(usize::MAX);
                used_weight_bytes = used_weight_bytes.saturating_add(non_mapped_size_in_bytes);
            }
            // Next-first views over the unplaced tail (`layer_sizes_in_bytes`
            // is stored reversed, so `.last()` is the next layer to place).
            // Same arithmetic as the historical fill loop, via the single
            // `count_fitting_layers` implementation (unit-tested below).
            let tail = layer_sizes_in_bytes.len();
            let next_sizes: Vec<usize> =
                layer_sizes_in_bytes.iter().rev().copied().collect();
            let next_kv: Vec<usize> =
                (0..tail).map(|i| kv_bytes_for_layer(layer + i)).collect();
            let count = count_fitting_layers(&next_sizes, &next_kv, used, cap);
            if count > 0 {
                for _ in 0..count {
                    let sz = layer_sizes_in_bytes
                        .pop()
                        .expect("counted from the live tail");
                    used_weight_bytes = used_weight_bytes.saturating_add(sz);
                }
                remaining = remaining.saturating_sub(used_weight_bytes);
            } else {
                let msg = format!(
                    "[mistralrs-auto-map] device {} fits 0 layers (usable {} MB): consider reducing auto map params from current: {params} (ex. reducing max seq len or max num images)",
                    dev.device_pretty_repr(),
                    b_to_mb!(cap),
                );
                eprintln!("{msg}");
                warn!("{msg}");
                ordinal += 1;
                continue;
            }
            count
        };
        if !dev.is_cpu() {
            mappings.push(DeviceLayerMapMetadata {
                ordinal,
                layers: layers_on_dev,
            });
            ordinal += 1;
        } else {
            includes_cpu = true;
        }
        layer += layers_on_dev;
    }
    if remaining > 0 {
        let over = b_to_mb!(remaining);
        anyhow::bail!(
            "This model does not fit on the devices {:?}, and exceeds total capacity by {}MB. Auto device mapping params: {params}",
            avail_copy.iter().rev().map(|(a, d)| format!("{} (avail: {}MB)", d.device_pretty_repr(), b_to_mb!(a))).collect::<Vec<_>>(),
            over
        );
    }
    if has_paged_attn && includes_cpu {
        let original_layers = layer_sizes_backup
            .take()
            .expect("layer sizes backup missing for paged attention fallback");
        // The original vector was in forward order, but `get_device_layers` handles
        // reversing internally, so we can pass it along unchanged.
        return get_device_layers(
            loader,
            config,
            num_layers,
            original_layers,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            devices,
            dtype,
            params,
            None,
        );
    }
    // Always-visible decision log (stderr + tracing, never tracing-only):
    // GPU/CPU layer totals, measured free VRAM per device, and the reason.
    // Bench verification: grep `mistralrs-auto-map` on stderr.
    let gpu_layers: usize = mappings.iter().map(|m| m.layers).sum();
    let cpu_layers = num_layers.saturating_sub(gpu_layers);
    let reason = decision_reason(gpu_layers, cpu_layers, preference);
    let free_detail = avail_copy
        .iter()
        .rev()
        .map(|(a, d)| format!("{}:free={}MiB", d.device_pretty_repr(), b_to_mb!(*a)))
        .join(", ");
    let summary = format!(
        "[mistralrs-auto-map] decision: gpu_layers={gpu_layers} cpu_layers={cpu_layers} total={num_layers} reason={reason} preference={preference:?} [{free_detail}]"
    );
    eprintln!("{summary}");
    info!("{summary}");
    Ok(DeviceMapMetadata::from_num_device_layers(mappings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_params_promote_to_multimodal_defaults_after_detection() {
        let params = AutoDeviceMapParams::Text {
            max_seq_len: 4096,
            max_batch_size: 7,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(max_batch_size, 7);
                assert_eq!(
                    max_image_shape,
                    (
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                    )
                );
                assert_eq!(max_num_images, AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn multimodal_params_preserve_explicit_limits_after_detection() {
        let params = AutoDeviceMapParams::Multimodal {
            max_seq_len: 8192,
            max_batch_size: 3,
            max_image_shape: (1536, 1024),
            max_num_images: 5,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1536, 1024));
                assert_eq!(max_num_images, 5);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn memory_sum_saturates_capacity_accounting_overflow() {
        assert_eq!(saturating_memory_sum([usize::MAX - 2, 1, 1]), usize::MAX);
        assert_eq!(saturating_memory_sum([usize::MAX, 1]), usize::MAX);
    }

    #[test]
    fn checked_memory_sum_distinguishes_max_from_overflow() {
        assert_eq!(checked_memory_sum([usize::MAX - 1, 1]), Some(usize::MAX));
        assert_eq!(checked_memory_sum([usize::MAX, 1]), None);
    }

    #[test]
    fn post_load_utilization_is_not_frozen_to_the_inventory_estimate() {        let requested = MemoryGpuConfig::Utilization(0.85);
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), true),
            MemoryGpuConfig::Utilization(value) if value == 0.85
        ));
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), false,),
            MemoryGpuConfig::MbAmount(45_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::MbAmount(60_000),
                MemoryGpuConfig::MbAmount(45_000),
                true,
            ),
            MemoryGpuConfig::MbAmount(60_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 60_000,
                    min_mb: Some(40_000),
                },
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 45_000,
                    min_mb: Some(40_000),
                },
                true,
            ),
            MemoryGpuConfig::BestEffortMbAmount {
                target_mb: 60_000,
                min_mb: Some(40_000),
            }
        ));
    }

    // ---- GPU-preference auto-map (D1 forense) ----

    #[test]
    fn gpu_preference_truth_table() {
        use super::{gpu_preference, GpuPreference};
        assert_eq!(gpu_preference(true, false), GpuPreference::Offered);
        assert_eq!(gpu_preference(true, true), GpuPreference::Offered);
        assert_eq!(gpu_preference(false, true), GpuPreference::Probed);
        assert_eq!(gpu_preference(false, false), GpuPreference::Absent);
    }

    #[test]
    fn fitting_count_all_when_ample() {
        use super::count_fitting_layers;
        // Ample simulated GPU budget: the whole stack lands on cuda.
        assert_eq!(
            count_fitting_layers(&[100, 100, 100, 100], &[10, 10, 10, 10], 50, 10_000),
            4
        );
    }

    #[test]
    fn fitting_count_partial_is_deterministic() {
        use super::count_fitting_layers;
        // 50 + 110 + 110 = 270 <= 280; the third would reach 380 > 280.
        assert_eq!(
            count_fitting_layers(&[100, 100, 100], &[10, 10, 10], 50, 280),
            2
        );
        // Exact fit counts (the boundary is `> cap`, not `>=`).
        assert_eq!(count_fitting_layers(&[100, 100], &[10, 10], 50, 270), 2);
        // Scarce VRAM spill is prefix-deterministic: always the FIRST n.
        assert_eq!(count_fitting_layers(&[60, 60, 60, 60], &[5, 5, 5, 5], 0, 130), 2);
    }

    #[test]
    fn fitting_count_zero_and_overflow_safe() {
        use super::count_fitting_layers;
        assert_eq!(count_fitting_layers(&[100], &[10], 500, 100), 0);
        assert_eq!(count_fitting_layers(&[], &[], 0, 100), 0);
        // usize::MAX-scale sums must break, never wrap.
        assert_eq!(
            count_fitting_layers(&[usize::MAX, 1], &[1, 1], 0, usize::MAX),
            0
        );
        assert_eq!(count_fitting_layers(&[8], &[8], usize::MAX, usize::MAX), 0);
    }

    #[test]
    fn absent_notice_names_cpu_explicitly() {
        use super::absent_accelerator_notice;
        let msg = absent_accelerator_notice("[cpu]");
        assert!(msg.contains("CPU"), "{msg}");
        assert!(msg.contains("WARNING"), "{msg}");
        assert!(msg.contains("silent"), "{msg}");
    }

    #[test]
    fn decision_reasons_cover_split_cases() {
        use super::{decision_reason, GpuPreference};
        assert_eq!(
            decision_reason(42, 0, GpuPreference::Offered),
            "fits-on-gpu"
        );
        assert_eq!(
            decision_reason(0, 42, GpuPreference::Absent),
            "no-accelerator-cpu-only"
        );
        assert_eq!(
            decision_reason(0, 42, GpuPreference::Offered),
            "gpu-fits-zero-spill-to-cpu"
        );
        assert_eq!(
            decision_reason(12, 30, GpuPreference::Offered),
            "gpu-overflow-spill-to-cpu"
        );
        assert_eq!(
            decision_reason(12, 30, GpuPreference::Probed),
            "gpu-overflow-spill-to-cpu"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn system_probe_finds_no_accelerator_without_cuda() {
        use super::probe_system_accelerator;
        assert!(probe_system_accelerator().is_none());
    }

    #[test]
    fn auto_prefers_gpu_when_present_else_cpu_only() {
        use super::super::{AutoDeviceMapQuantization, DeviceMappedModelLoader};
        use crate::matformer::MatformerSliceConfig;

        struct StubCfg;
        impl ModelConfigLike for StubCfg {
            fn max_seq_len(&self) -> usize {
                64
            }
            fn num_layers(&self) -> usize {
                4
            }
            fn hidden_size(&self) -> usize {
                32
            }
            fn num_kv_heads(&self) -> usize {
                2
            }
            fn num_attn_heads(&self) -> usize {
                4
            }
            fn k_head_dim(&self) -> usize {
                8
            }
            fn v_head_dim(&self) -> usize {
                8
            }
        }

        struct StubLoader;
        impl DeviceMappedModelLoader for StubLoader {
            fn non_mapped_max_act_size_elems(
                &self,
                _config: &str,
                _params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                Ok(1024)
            }
            fn mapped_max_act_size_elems(
                &self,
                _config: &str,
                _params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                Ok(1024)
            }
            fn non_mapped_size_in_bytes(
                &self,
                _config: &str,
                _dtype: DType,
                _weight_pack_factor: usize,
                _quantization: Option<&AutoDeviceMapQuantization<'_>>,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<usize> {
                Ok(1 << 20)
            }
            fn layer_sizes_in_bytes(
                &self,
                _config: &str,
                _dtype: DType,
                _weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                Ok(vec![64 << 20; 4])
            }
            fn num_layers(&self, _config: &str) -> Result<usize> {
                Ok(4)
            }
            fn model_config(&self, _config: &str) -> Result<Box<dyn ModelConfigLike>> {
                Ok(Box::new(StubCfg))
            }
        }

        // 4 x 64MiB layers + 1MiB non-mapped: fits any real device budget,
        // so placement is decided purely by the preference rule.
        let loader = StubLoader;
        let params = AutoDeviceMapParams::Text {
            max_seq_len: 64,
            max_batch_size: 1,
        };
        let total = 4 * (64 << 20) + (1 << 20);
        let meta = get_device_layers(
            &loader,
            "{}",
            4,
            vec![64 << 20; 4],
            1 << 20,
            total,
            &[Device::Cpu],
            DType::F16,
            &params,
            None,
        )
        .unwrap();
        let gpu: usize = meta
            .device_layers()
            .map(|ds| ds.iter().map(|m| m.layers).sum())
            .unwrap_or(0);
        // No system accelerator: the whole stack is CPU (explicit fallback).
        // With one (cuda build + GPU host): the ample budget puts it all on
        // GPU. Either way the preference holds — never a silent mix.
        if super::probe_system_accelerator().is_none() {
            assert_eq!(gpu, 0);
        } else {
            assert_eq!(gpu, 4);
        }
    }
}
