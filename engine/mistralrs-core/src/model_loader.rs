use std::{
    fs::{self, File},
    path::PathBuf,
    str::FromStr,
};

use crate::{
    get_toml_selected_model_dtype,
    pipeline::{
        AutoLoaderBuilder, GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder,
        GGUFSpecificConfig, HfConfigOverrides, NormalLoaderBuilder, NormalSpecificConfig,
    },
    toml_selector::get_toml_selected_model_device_map_params,
    AutoDeviceMapParams, Loader, ModelDType, ModelSelected, TomlLoaderArgs, TomlSelector,
    Topology, GGUF_MULTI_FILE_DELIMITER,
    UQFF_MULTI_FILE_DELIMITER,
};

/// A builder for a loader using the selected model.
pub struct LoaderBuilder {
    model: ModelSelected,
    no_kv_cache: bool,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
    max_model_len: Option<usize>,
    hf_config_overrides: Option<HfConfigOverrides>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

impl LoaderBuilder {
    pub fn new(model: ModelSelected) -> Self {
        Self {
            model,
            no_kv_cache: false,
            chat_template: None,
            jinja_explicit: None,
            max_model_len: None,
            hf_config_overrides: None,
            mtp: false,
            encoder_cache_memory_bytes: None,
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
    }

    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }
    pub fn with_chat_template(mut self, chat_template: Option<String>) -> Self {
        self.chat_template = chat_template;
        self
    }
    pub fn with_jinja_explicit(mut self, jinja_explicit: Option<String>) -> Self {
        self.jinja_explicit = jinja_explicit;
        self
    }
    pub fn with_max_model_len(mut self, max_model_len: Option<usize>) -> Self {
        self.max_model_len = max_model_len;
        self
    }
    pub fn with_hf_config_overrides(
        mut self,
        hf_config_overrides: Option<HfConfigOverrides>,
    ) -> Self {
        self.hf_config_overrides = hf_config_overrides;
        self
    }

    pub fn build(self) -> anyhow::Result<Box<dyn Loader>> {
        loader_from_model_selected(self)
    }
}

pub fn get_tgt_non_granular_index(_model: &ModelSelected) -> Option<usize> {
    // kitty-infer carries no X-LoRA adapters: no variant has a granular index.
    None
}

pub fn get_model_dtype(model: &ModelSelected) -> anyhow::Result<ModelDType> {
    match model {
        ModelSelected::Plain { dtype, .. }
        | ModelSelected::GGML { dtype, .. }
        | ModelSelected::GGUF { dtype, .. }
        | ModelSelected::Run { dtype, .. } => Ok(*dtype),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            Ok(get_toml_selected_model_dtype(&selector))
        }
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    }
}

pub fn get_auto_device_map_params(model: &ModelSelected) -> anyhow::Result<AutoDeviceMapParams> {
    match model {
        ModelSelected::Plain {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::GGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::GGUF {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
        }),
        ModelSelected::Run {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
        }),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            get_toml_selected_model_device_map_params(&selector)
        }
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    }
}

fn loader_from_model_selected(args: LoaderBuilder) -> anyhow::Result<Box<dyn Loader>> {
    if args.max_model_len == Some(0) {
        anyhow::bail!("max_model_len must be greater than zero");
    }
    let supports_hf_config_overrides = matches!(
        &args.model,
        ModelSelected::Plain { .. }
            | ModelSelected::Run { .. }
            | ModelSelected::Toml { .. }
    );
    if args.hf_config_overrides.is_some() && !supports_hf_config_overrides {
        anyhow::bail!("HF config overrides are supported only for text and multimodal models");
    }
    let supports_max_model_len = supports_hf_config_overrides
        || matches!(
            &args.model,
            ModelSelected::GGUF { .. }
        );
    if args.max_model_len.is_some() && !supports_max_model_len {
        anyhow::bail!("max_model_len is not supported by this model format");
    }

    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            let args = TomlLoaderArgs {
                chat_template: args.chat_template,
                no_kv_cache: args.no_kv_cache,
                jinja_explicit: args.jinja_explicit,
                encoder_cache_memory_bytes: args.encoder_cache_memory_bytes,
                max_model_len: args.max_model_len,
                hf_config_overrides: args.hf_config_overrides,
            };
            (selector, args).try_into()?
        }
        ModelSelected::Plain {
            model_id,
            tokenizer_json,
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
            matformer_config_path,
            matformer_slice_name,
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
                matformer_config_path,
                matformer_slice_name,
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_mtp(args.mtp)
        .build(arch)?,
        ModelSelected::Run {
            model_id,
            tokenizer_json,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let builder = AutoLoaderBuilder::new(
                NormalSpecificConfig {
                    topology: Topology::from_option_path(topology.clone())?,
                    organization: organization.unwrap_or_default(),
                    write_uqff: write_uqff.clone(),
                    from_uqff: from_uqff.clone().map(|x| {
                        x.split(UQFF_MULTI_FILE_DELIMITER)
                            .map(PathBuf::from_str)
                            .map(|x| x.unwrap())
                            .collect::<Vec<_>>()
                    }),
                    imatrix: imatrix.clone(),
                    calibration_file: calibration_file.clone(),
                    hf_cache_path: hf_cache_path.clone(),
                    hf_config_overrides: args.hf_config_overrides.clone(),
                    max_model_len: args.max_model_len,
                    matformer_config_path: matformer_config_path.clone(),
                    matformer_slice_name: matformer_slice_name.clone(),
                },
                args.chat_template,
                tokenizer_json,
                model_id,
                args.no_kv_cache,
                args.jinja_explicit,
            );
            let builder = if let Some(ref path) = hf_cache_path {
                builder.hf_cache_path(path.clone())
            } else {
                builder
            };
            builder
                .with_mtp(args.mtp)
                .with_encoder_cache_memory_bytes(args.encoder_cache_memory_bytes)
                .build()
        }
        ModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            tokenizer_json,
            mmproj_filename,
            lora_adapters,
            lora_runtime_config,
            topology,
            organization,
            write_uqff,
            imatrix,
            calibration_file,
            max_edge,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
            ..
        } => {
            if lora_runtime_config.is_none() && !lora_adapters.is_empty() {
                anyhow::bail!("GGUF LoRA adapters require a dynamic LoRA runtime configuration");
            }
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
            if let Some(tokenizer_json) = tokenizer_json {
                builder = builder.with_tokenizer_json(tokenizer_json);
            }
            if let Some(runtime_config) = lora_runtime_config {
                builder = builder.with_dynamic_lora(lora_adapters, runtime_config);
            }
            builder.build()
        }
        ModelSelected::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    };
    Ok(loader)
}
