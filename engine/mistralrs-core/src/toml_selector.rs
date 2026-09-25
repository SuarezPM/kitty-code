use std::{fs::File, path::PathBuf, str::FromStr};

use serde::{de, Deserialize, Deserializer};

use crate::{
    pipeline::IsqOrganization,
    AutoDeviceMapParams,
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig, Loader,
    LoraAdapterSpec, LoraRuntimeConfig, ModelDType,
    NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig,
    Topology, UqffWriteConfig, GGUF_MULTI_FILE_DELIMITER, UQFF_MULTI_FILE_DELIMITER,
};

fn default_one() -> usize {
    1
}

fn default_dtype() -> ModelDType {
    ModelDType::Auto
}

fn default_empty_vec_usize() -> Vec<usize> {
    Vec::new()
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

fn default_max_num_images() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES
}

fn default_max_image_length() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum TomlModelSelected {
    /// Select a plain model, without quantization or adapters
    #[serde(rename = "plain")]
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: https://arxiv.org/abs/2310.02410).
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--imatrix/-i`
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,
    },

    /// Select a GGUF model.
    #[allow(clippy::upper_case_acronyms)]
    #[serde(rename = "gguf")]
    GGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use semicolons to separate multiple files.
        quantized_filename: String,

        /// Multimodal projector filename(s), separated by semicolons.
        mmproj_filename: Option<String>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization.
        organization: Option<IsqOrganization>,

        /// UQFF output configuration.
        write_uqff: Option<UqffWriteConfig>,

        /// Imatrix file used while requantizing.
        imatrix: Option<PathBuf>,

        /// Calibration file used to generate an imatrix while requantizing.
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        hf_cache_path: Option<PathBuf>,

        /// Path to a local Matryoshka Transformer configuration CSV file.
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use.
        matformer_slice_name: Option<String>,
    },

    /// Select a GGML model.
    #[allow(clippy::upper_case_acronyms)]
    #[serde(rename = "ggml")]
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LegacyModelFamily {
    XLora,
    Lora,
    LegacyLora,
    Multimodal,
    Embedding,
}

fn deserialize_model_selected<'de, D>(deserializer: D) -> Result<TomlModelSelected, D::Error>
where
    D: Deserializer<'de>,
{
    let mut value = toml::Value::deserialize(deserializer)?;
    let table = value
        .as_table_mut()
        .ok_or_else(|| de::Error::custom("model selector must be a TOML table"))?;

    if table.contains_key("adapter_model_ids") {
        return Err(de::Error::custom(
            "legacy `adapter_model_ids` is no longer supported; use `kind = \"lora\"` with `adapters`",
        ));
    }

    if !table.contains_key("kind") {
        let kind = infer_legacy_model_kind(table).map_err(de::Error::custom)?;
        table.insert("kind".to_string(), toml::Value::String(kind.to_string()));
    }

    value.try_into().map_err(de::Error::custom)
}

fn infer_legacy_model_kind(table: &toml::Table) -> Result<&'static str, String> {
    let quantized = [
        "tok_model_id",
        "quantized_model_id",
        "quantized_filename",
        "gqa",
    ]
    .iter()
    .any(|key| table.contains_key(*key));
    let has_gqa = table.contains_key("gqa");

    let mut families = Vec::new();
    if table.contains_key("xlora_model_id") || table.contains_key("tgt_non_granular_index") {
        families.push(LegacyModelFamily::XLora);
    }
    if table.contains_key("adapters") || table.contains_key("runtime_config") {
        families.push(LegacyModelFamily::Lora);
    }
    if table.contains_key("adapters_model_id") {
        families.push(LegacyModelFamily::LegacyLora);
    }
    if ["max_edge", "max_num_images", "max_image_length"]
        .iter()
        .any(|key| table.contains_key(*key))
    {
        families.push(LegacyModelFamily::Multimodal);
    }
    if let Some(family) = infer_legacy_arch_family(table)? {
        if !families.contains(&family) {
            families.push(family);
        }
    }

    if families.len() > 1 {
        return Err(
            "model selector without `kind` has conflicting fields; set `kind` explicitly"
                .to_string(),
        );
    }

    let family = families.first().copied();
    match (family, quantized, has_gqa) {
        // kitty-infer is text-only: legacy adapter/multimodal/embedding
        // shapes are refused honestly instead of mapping to cut kinds.
        (Some(_), _, _) => Err(
            "legacy xlora/lora/multimodal/embedding selector fields are not supported by this \
             text-only build; set `kind` explicitly to plain/gguf/ggml"
                .to_string(),
        ),
        (None, true, false) => Ok("gguf"),
        (None, true, true) => Ok("ggml"),
        (None, false, _) => Ok("plain"),
    }
}

fn infer_legacy_arch_family(table: &toml::Table) -> Result<Option<LegacyModelFamily>, String> {
    let Some(arch) = table.get("arch") else {
        return Ok(None);
    };
    if NormalLoaderType::deserialize(arch.clone()).is_ok() {
        return Ok(None);
    }

    // Multimodal/embedding architectures are not supported by this text-only
    // build; only NormalLoaderType arches resolve here.
    let _ = arch;
    Ok(None)
}

#[derive(Deserialize)]
pub struct TomlSelector {
    /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
    tokenizer_json: Option<String>,

    /// Selected model
    #[serde(deserialize_with = "deserialize_model_selected")]
    model: TomlModelSelected,

    /// Legacy target/draft speculative decoding was removed. Keep this field
    /// only to reject old configs explicitly instead of silently ignoring them.
    #[serde(default)]
    speculative: Option<serde::de::IgnoredAny>,

}

#[derive(Clone)]
struct TomlLoaderInnerParams {
    chat_template: Option<String>,
    no_kv_cache: bool,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    encoder_cache_memory_bytes: Option<usize>,
    max_model_len: Option<usize>,
    hf_config_overrides: Option<crate::HfConfigOverrides>,
}

impl TryInto<Box<dyn Loader>> for (TomlSelector, TomlLoaderArgs) {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<Box<dyn Loader>, Self::Error> {
        let (selector, args) = self;
        let args = TomlLoaderInnerParams {
            chat_template: args.chat_template,
            no_kv_cache: args.no_kv_cache,
            tokenizer_json: selector.tokenizer_json,
            jinja_explicit: args.jinja_explicit,
            encoder_cache_memory_bytes: args.encoder_cache_memory_bytes,
            max_model_len: args.max_model_len,
            hf_config_overrides: args.hf_config_overrides,
        };
        if selector.speculative.is_some() {
            anyhow::bail!(
                "legacy target/draft speculative decoding in TOML configs was removed; use MTP through --mtp-model or the MTP API instead"
            );
        }
        loader_from_selected(args, selector.model)
    }
}

pub struct TomlLoaderArgs {
    pub chat_template: Option<String>,
    pub no_kv_cache: bool,
    pub jinja_explicit: Option<String>,
    pub encoder_cache_memory_bytes: Option<usize>,
    pub max_model_len: Option<usize>,
    pub hf_config_overrides: Option<crate::HfConfigOverrides>,
}

pub fn get_toml_selected_model_dtype(model: &TomlSelector) -> ModelDType {
    match model.model {
        TomlModelSelected::Plain { dtype, .. }
        | TomlModelSelected::GGUF { dtype, .. }
        | TomlModelSelected::GGML { dtype, .. } => dtype,
    }
}

pub fn get_toml_selected_model_device_map_params(
    model: &TomlSelector,
) -> anyhow::Result<AutoDeviceMapParams> {
    match model.model {
        TomlModelSelected::Plain {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::GGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::GGUF {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        }),
    }
}

fn loader_from_selected(
    args: TomlLoaderInnerParams,
    model: TomlModelSelected,
) -> anyhow::Result<Box<dyn Loader>> {
    let supports_hf_config_overrides = matches!(
        &model,
        TomlModelSelected::Plain { .. }
    );
    if args.hf_config_overrides.is_some() && !supports_hf_config_overrides {
        anyhow::bail!("HF config overrides are supported only for text models in this build");
    }
    let supports_max_model_len = supports_hf_config_overrides
        || matches!(
            &model,
            TomlModelSelected::GGUF { .. }
        );
    if args.max_model_len.is_some() && !supports_max_model_len {
        anyhow::bail!("max_model_len is not supported by this model format");
    }

    let loader: Box<dyn Loader> = match model {
        TomlModelSelected::Plain {
            model_id,
            arch,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix,
                calibration_file,
                hf_cache_path,
                hf_config_overrides: args.hf_config_overrides.clone(),
                max_model_len: args.max_model_len,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(arch)?,
        TomlModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            mmproj_filename,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let mut builder = GGUFLoaderBuilder::new(
                args.chat_template,
                tok_model_id,
                quantized_model_id,
                quantized_filename
                    .split(GGUF_MULTI_FILE_DELIMITER)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>(),
                GGUFSpecificConfig {
                    topology: Topology::from_option_path(topology)?,
                    organization: organization.unwrap_or_default(),
                    write_uqff,
                    imatrix,
                    calibration_file,
                    max_edge,
                    max_model_len: args.max_model_len,
                    hf_cache_path,
                    matformer_config_path,
                    matformer_slice_name,
                },
                args.no_kv_cache,
                args.jinja_explicit,
            )
            .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes);
            if let Some(mmproj_filename) = mmproj_filename {
                builder = builder.with_mmproj_files(
                    mmproj_filename
                        .split(GGUF_MULTI_FILE_DELIMITER)
                        .map(ToOwned::to_owned)
                        .collect(),
                );
            }
            if let Some(tokenizer_json) = args.tokenizer_json {
                builder = builder.with_tokenizer_json(tokenizer_json);
            }
            builder.build()
        }
        TomlModelSelected::GGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
    };
    Ok(loader)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn legacy_plain_without_kind_is_supported() {
        let selector = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "mistralai/Mistral-7B-Instruct-v0.1"
            "#,
        )
        .unwrap();

        assert!(matches!(selector.model, TomlModelSelected::Plain { .. }));
    }

    #[test]
    fn legacy_quantized_models_without_kind_are_inferred() {
        let gguf = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.gguf"
            "#,
        )
        .unwrap();
        assert!(matches!(gguf.model, TomlModelSelected::GGUF { .. }));

        let ggml = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.bin"
                gqa = 8
            "#,
        )
        .unwrap();
        assert!(matches!(ggml.model, TomlModelSelected::GGML { .. }));
    }





    #[test]
    fn legacy_specialized_models_unknown_arch_is_rejected() {
        // Real live behavior is Err, NOT a Plain fallback: `kind` infers to
        // "plain", but the arch variants don't exist (no `serde(other)`).
        let multimodal = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "vision-model"
                arch = "qwen2vl"
            "#,
        )
        .err()
        .expect("qwen2vl arch must not parse as plain");
        assert!(multimodal.to_string().contains("qwen2vl"));

        let embedding = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "embedding-model"
                arch = "qwen3embedding"
                tokenizer_json = "tokenizer.json"
            "#,
        )
        .err()
        .expect("qwen3embedding arch must not parse as plain");
        assert!(embedding.to_string().contains("qwen3embedding"));
    }

    #[test]
    fn legacy_nested_tokenizer_json_keeps_plain_default() {
        let selector = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "text-model"
                tokenizer_json = "tokenizer.json"
            "#,
        )
        .unwrap();

        assert!(matches!(selector.model, TomlModelSelected::Plain { .. }));
    }

    #[test]
    fn explicit_unknown_kind_is_rejected() {
        // Only plain/gguf/ggml survive as `kind` tags; anything else fails
        // deserialization (no `MultimodalPlain` variant to be authoritative).
        let error = toml::from_str::<TomlSelector>(
            r#"
                [model]
                kind = "multimodal"
                model_id = "vision-model"
            "#,
        )
        .err()
        .expect("unknown kind must not parse");

        assert!(error.to_string().contains("multimodal"));
    }

    #[test]
    fn legacy_selector_with_conflicting_discriminators_is_rejected() {
        let error = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "ambiguous-model"
                max_edge = 1024
                arch = "qwen3embedding"
            "#,
        )
        .err()
        .expect("conflicting model fields should not parse");

        // `infer_legacy_arch_family` always returns Ok(None), so families is
        // [Multimodal] (len 1, no conflict branch); the real refusal is the
        // text-only message (exact string in `infer_legacy_model_kind`).
        assert!(error
            .to_string()
            .contains("not supported by this text-only build"));
    }
}
