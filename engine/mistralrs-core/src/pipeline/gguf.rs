use super::llg::build_llg_factory;
use super::{
    get_model_paths, text_models_inputs_processor::ModelInputs, AdapterKind, CacheManager,
    GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, QuantizationKind, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::WorkerTransferData;
use crate::gguf::{
    base_model::infer_hf_base_model_id,
    convert_gguf_metadata_to_hf_tokenizer,
    get_gguf_chat_template, get_gguf_chat_template_from_metadata,
    normal_bindings::build_normal_bindings,
    normal_config::{
        normal_loader_hint_from_external_config, normalize_external_normal_config,
        synthesize_normal_config, validate_normal_config_tensor_inventory,
    },
    normal_registry::{resolve_native_adapter, GgufDescriptor, RopePairing},
    validate_external_gguf_tokenizer, GgufTokenizerConversion,
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::kv_cache::FullCacheManager;
use crate::lora::Ordering;
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::hf::{build_api, get_file, list_repo_files};
use crate::pipeline::loaders::{stamp_qk_rope_layout, DeviceMappedModelLoader};
use crate::pipeline::normal::{NormalLoaderBuilder, NormalSpecificConfig, PreparedNormalSource};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::{
    distributed, get_mut_arcmutex, get_paths_gguf, DeviceMapSetting, LocalModelPaths,
    LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig, Pipeline,
    Topology, TryIntoDType, UqffWriteConfig, GLOBAL_HF_CACHE,
};
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use either::Either;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// Architectures that only exist as multimodal checkpoints (Sept 2026 generation:
// Gemma 3n/4, Llama 4, Muse Glimmer, Qwen3-VL/MoE). Pre-2025 families (Qwen2-VL
// and older) were purged: their GGUFs fall through to the generic unsupported
// path below, which is equally honest. Text-only build: every entry here is
// refused with cause, never loaded.
const PROJECTOR_REQUIRED_ARCHITECTURES: &[&str] = &[
    "gemma3n",
    "gemma4",
    "llama4",
    "muse-glimmer",
    "qwen3vl",
    "qwen3vlmoe",
];

fn requires_multimodal_projector(architecture: &str) -> bool {
    PROJECTOR_REQUIRED_ARCHITECTURES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(architecture))
}

fn validate_native_dynamic_lora(
    dynamic_lora: Option<&DynamicLoraConfig>,
    rope_pairing: RopePairing,
    architecture: &str,
) -> Result<()> {
    if dynamic_lora.is_some() && rope_pairing == RopePairing::Adjacent {
        bail!(
            "dynamic LoRA is not supported for native GGUF architecture `{architecture}` because its Q/K tensors use converter-permuted adjacent RoPE order; load the original safetensors model or omit the LoRA adapter"
        );
    }
    Ok(())
}



fn preferred_hf_config(files: &[String]) -> Option<&'static str> {
    if files.iter().any(|file| file == "config.json") {
        Some("config.json")
    } else if files.iter().any(|file| file == "params.json") {
        Some("params.json")
    } else {
        None
    }
}






/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: crate::pipeline::IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

struct NativeNormalLoadArgs<'a> {
    paths: &'a dyn ModelPaths,
    dtype: &'a dyn TryIntoDType,
    device: &'a Device,
    silent: bool,
    mapper: DeviceMapSetting,
    in_situ_quant: Option<IsqType>,
    paged_attn_config: Option<PagedAttentionConfig>,
}

#[derive(Clone)]
struct DynamicLoraConfig {
    adapters: Vec<LoraAdapterSpec>,
    runtime: LoraRuntimeConfig,
}

struct ResolvedGgufTokenizer {
    conversion: GgufTokenizerConversion,
    generation_config_compatible: bool,
}

#[derive(Clone, Copy)]
enum TokenizerFallback {
    Strict,
    Automatic,
    ModelAssets,
}

fn resolve_tokenizer_candidate(
    path: &Path,
    fallback: TokenizerFallback,
    external: Result<GgufTokenizerConversion>,
    embedded: impl FnOnce() -> Result<GgufTokenizerConversion>,
) -> Result<ResolvedGgufTokenizer> {
    match external {
        Ok(conversion) => Ok(ResolvedGgufTokenizer {
            conversion,
            generation_config_compatible: true,
        }),
        Err(error) if !matches!(fallback, TokenizerFallback::Strict) => {
            match fallback {
                TokenizerFallback::Automatic => warn!(
                    "Ignoring automatically discovered tokenizer `{}` because it does not match \
                     the GGUF vocabulary: {error}",
                    path.display()
                ),
                TokenizerFallback::ModelAssets => warn!(
                    "Tokenizer from model assets `{}` does not match the GGUF vocabulary; using \
                     the tokenizer embedded in the GGUF instead: {error}",
                    path.display()
                ),
                TokenizerFallback::Strict => unreachable!(),
            }
            Ok(ResolvedGgufTokenizer {
                conversion: embedded().with_context(|| {
                    format!(
                        "Tokenizer `{}` was rejected ({error}); GGUF tokenizer conversion also \
                         failed",
                        path.display()
                    )
                })?,
                generation_config_compatible: false,
            })
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "Tokenizer `{}` is incompatible with the GGUF model vocabulary",
                path.display()
            )
        }),
    }
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    encoder_cache_memory_bytes: Option<usize>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` optionally overrides embedded
    /// configuration and tokenizer assets. If `chat_template` is specified, it is treated as a path
    /// and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    pub fn with_mmproj_files(mut self, mmproj_filenames: Vec<String>) -> Self {
        self.mmproj_filenames = Some(mmproj_filenames);
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: String) -> Self {
        self.tokenizer_json = Some(tokenizer_json);
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
    }

    pub fn with_dynamic_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime: LoraRuntimeConfig,
    ) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();
        self.dynamic_lora = Some(DynamicLoraConfig { adapters, runtime });
        self.xlora_model_id = None;
        self.xlora_order = None;
        self.tgt_non_granular_index = None;
        self
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            dynamic_lora: self.dynamic_lora,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            mmproj_filenames: self.mmproj_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            mmproj_filenames: None,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            tokenizer_json: None,
            dynamic_lora: None,
            kind,
            tgt_non_granular_index,
            config,
            jinja_explicit,
            encoder_cache_memory_bytes: None,
        }
    }

    fn dynamic_lora_adapters(&self) -> Option<&[LoraAdapterSpec]> {
        self.dynamic_lora
            .as_ref()
            .map(|config| config.adapters.as_slice())
    }

    fn resolve_tokenizer(
        &self,
        paths: &dyn ModelPaths,
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Result<ResolvedGgufTokenizer> {
        let (path, fallback) = if let Some(tokenizer_json) = self.tokenizer_json.as_ref() {
            (PathBuf::from(tokenizer_json), TokenizerFallback::Strict)
        } else if paths.get_tokenizer_filename().as_os_str().is_empty() {
            return Ok(ResolvedGgufTokenizer {
                conversion: convert_gguf_metadata_to_hf_tokenizer(metadata)?,
                generation_config_compatible: true,
            });
        } else {
            (
                paths.get_tokenizer_filename().clone(),
                if self.model_id.is_none() {
                    TokenizerFallback::Automatic
                } else {
                    TokenizerFallback::ModelAssets
                },
            )
        };

        let external = match get_tokenizer(&path, None) {
            Ok(tokenizer) => validate_external_gguf_tokenizer(tokenizer, metadata),
            Err(error) if matches!(fallback, TokenizerFallback::ModelAssets) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to load tokenizer from model assets `{}`",
                        path.display()
                    )
                })
            }
            Err(error) => Err(error),
        };
        resolve_tokenizer_candidate(&path, fallback, external, || {
            convert_gguf_metadata_to_hf_tokenizer(metadata)
        })
    }

    fn resolve_generation_config(
        &self,
        paths: &dyn ModelPaths,
        tokenizer: &ResolvedGgufTokenizer,
    ) -> Option<GenerationConfig> {
        let filename = paths.get_gen_conf_filename()?;
        if !tokenizer.generation_config_compatible {
            warn!(
                "Ignoring generation config `{}` because the external tokenizer was incompatible \
                 with the GGUF model",
                filename.display()
            );
            return None;
        }
        let config = match fs::read_to_string(filename)
            .with_context(|| format!("Failed to read `{}`", filename.display()))
            .and_then(|raw| {
                serde_json::from_str::<GenerationConfig>(&raw)
                    .context("Failed to parse generation_config.json")
            }) {
            Ok(config) => config,
            Err(error) => {
                warn!("Ignoring generation config: {error:#}");
                return None;
            }
        };
        if let Err(error) =
            config.validate_token_ids(tokenizer.conversion.tokenizer.get_vocab_size(true))
        {
            warn!("Ignoring generation config: {error}");
            return None;
        }
        Some(config)
    }

    fn load_native_normal(
        &self,
        args: NativeNormalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeNormalLoadArgs {
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        } = args;
        let archive = Arc::new(mistralrs_quant::GgufArchive::open(
            paths.get_weight_filenames(),
        )?);
        let architecture = match archive.metadata_value("general.architecture") {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => value.as_str(),
            Some(value) => {
                bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => bail!("GGUF metadata is missing `general.architecture`"),
        };
        let metadata_keys = archive
            .metadata()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let tensor_names = archive
            .tensors()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let metadata_string = |key| match archive.metadata_value(key) {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => Some(value.as_str()),
            _ => None,
        };
        if requires_multimodal_projector(architecture) {
            bail!(
                "GGUF architecture `{architecture}` is multimodal-only and not supported by this \
                 text-only build (see F0-RECON.md amputation map)"
            );
        }
        let descriptor = GgufDescriptor::new(architecture, &metadata_keys, &tensor_names)?
            .with_model_identity(
                metadata_string("general.name"),
                metadata_string("general.basename"),
            );

        let external_config = if paths.get_config_filename().as_os_str().is_empty() {
            None
        } else {
            Some(fs::read_to_string(paths.get_config_filename())?)
        };
        let explicit_loader = external_config
            .as_deref()
            .map(normal_loader_hint_from_external_config)
            .transpose()?;
        let explicit_loader = explicit_loader.flatten();
        let resolved = resolve_native_adapter(&descriptor, explicit_loader)?;
        let rope_pairing =
            crate::gguf::normal_registry::schema_for(descriptor.architecture).rope_pairing;
        validate_native_dynamic_lora(
            self.dynamic_lora.as_ref(),
            rope_pairing,
            descriptor.architecture.as_str(),
        )?;
        debug!(
            "Loading GGUF architecture `{}` through native {:?} ({:?}, layouts {:?})",
            descriptor.architecture,
            resolved.adapter.loader,
            resolved.reason,
            resolved.adapter.layouts
        );
        let loader_type = resolved.adapter.loader.clone();
        let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
        let config = match external_config {
            Some(config) => {
                let config = normalize_external_normal_config(
                    &loader_type,
                    descriptor.architecture,
                    &config,
                )?;
                validate_normal_config_tensor_inventory(&config, &tensor_names)?;
                config
            }
            None => synthesize_normal_config(&loader_type, archive.metadata(), &tensor_names)?,
        };
        let config = stamp_qk_rope_layout(&config, rope_pairing)?;
        let bindings = build_normal_bindings(&archive, &loader_type, descriptor.architecture)?;
        let internal_dtype = dtype.try_into_dtype(&[device])?;
        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
            archive.clone(),
            &bindings,
            internal_dtype,
        )?);
        let weights = source.sharded_var_builder(Device::Cpu);

        let tokenizer = self.resolve_tokenizer(paths, archive.metadata())?;
        let generation_config = self.resolve_generation_config(paths, &tokenizer);
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template_from_metadata(archive.metadata())?
            } else {
                None
            };
        let source = PreparedNormalSource {
            config,
            weights,
            tokenizer: tokenizer.conversion.tokenizer,
            generation_config,
            chat_template: gguf_chat_template,
            bos_token: tokenizer.conversion.bos,
            eos_token: tokenizer.conversion.eos,
            unk_token: tokenizer.conversion.unk,
            source_weight_files: paths.get_weight_filenames().to_vec(),
            rope_pairing,
        };
        let mut loader = NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: self.config.topology.clone(),
                organization: self.config.organization,
                write_uqff: self.config.write_uqff.clone(),
                imatrix: self.config.imatrix.clone(),
                calibration_file: self.config.calibration_file.clone(),
                hf_cache_path: self.config.hf_cache_path.clone(),
                matformer_config_path: self.config.matformer_config_path.clone(),
                matformer_slice_name: self.config.matformer_slice_name.clone(),
                ..Default::default()
            },
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.no_kv_cache,
            self.jinja_explicit.clone(),
        );
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader = loader.build_with_source(loader_type, source, self.kind.clone())?;
        loader.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }






}

impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let cache = self
            .config
            .hf_cache_path
            .clone()
            .map(hf_hub::Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            Some(revision.clone()),
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent,
            true
        );
        let paths = paths?;
        if self.mmproj_filenames.is_some() {
            // kitty-infer is a text-only engine: projector checkpoints need a
            // vision-capable loader, which was amputated (see F0-RECON.md).
            bail!(
                "mmproj (multimodal projector) checkpoints are not supported by this text-only build"
            );
        }

        self.load_model_from_path(
            paths.as_ref(),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        // Text-only build: the native GGUF path is the only path (X-LoRA
        // adapter loading was amputated; dynamic LoRA rides the same path).
        self.load_native_normal(NativeNormalLoadArgs {
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        })
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }
    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preferred_hf_config, requires_multimodal_projector,
        resolve_tokenizer_candidate,
        validate_native_dynamic_lora, DynamicLoraConfig, GGUFSpecificConfig,
        GgufTokenizerConversion, TokenizerFallback,
    };
    use crate::{
        gguf::{normal_registry::RopePairing, GGUFArchitecture},
    };
    use std::path::{Path, PathBuf};
    use tokenizers::{models::bpe::BPE, Tokenizer};

    fn tokenizer_conversion(marker: &str) -> GgufTokenizerConversion {
        GgufTokenizerConversion {
            tokenizer: Tokenizer::new(BPE::default()),
            bos: None,
            eos: Some(marker.to_string()),
            unk: None,
        }
    }


    #[test]
    fn native_dynamic_lora_rejects_adjacent_rope_gguf() {
        let dynamic_lora = DynamicLoraConfig {
            adapters: Vec::new(),
            runtime: Default::default(),
        };
        let error =
            validate_native_dynamic_lora(Some(&dynamic_lora), RopePairing::Adjacent, "llama")
                .unwrap_err();
        let error = error.to_string();
        assert!(error.contains("converter-permuted adjacent RoPE order"));
        assert!(error.contains("original safetensors model"));
        assert!(validate_native_dynamic_lora(
            Some(&dynamic_lora),
            RopePairing::HalfSplit,
            "qwen35",
        )
        .is_ok());
        assert!(validate_native_dynamic_lora(None, RopePairing::Adjacent, "llama").is_ok());
    }



    #[test]
    fn automatic_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("automatic-tokenizer.json"),
            TokenizerFallback::Automatic,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn model_asset_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn compatible_model_asset_tokenizer_is_preserved() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Ok(tokenizer_conversion("external")),
            || Ok(tokenizer_conversion("unused")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("external"));
        assert!(resolved.generation_config_compatible);
    }

    #[test]
    fn explicit_tokenizer_file_mismatch_is_an_error() {
        let error = resolve_tokenizer_candidate(
            Path::new("explicit-tokenizer.json"),
            TokenizerFallback::Strict,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("unused")),
        )
        .err()
        .expect("explicit mismatch must fail");

        assert!(error.to_string().contains("incompatible"));
    }

    #[test]
    fn prefers_hugging_face_config_over_native_params() {
        let files = vec!["params.json".to_string(), "config.json".to_string()];
        assert_eq!(preferred_hf_config(&files), Some("config.json"));
        assert_eq!(
            preferred_hf_config(&["params.json".to_string()]),
            Some("params.json")
        );
        assert_eq!(preferred_hf_config(&[]), None);
    }

}
