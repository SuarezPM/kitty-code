use std::path::PathBuf;

use clap::Subcommand;

use crate::{
    pipeline::{AutoDeviceMapParams, IsqOrganization, NormalLoaderType, UqffWriteConfig},
    LoraAdapterSpec, LoraRuntimeConfig, ModelDType,
};

// Default value functions for serde deserialization
fn default_model_dtype() -> ModelDType {
    ModelDType::Auto
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

fn parse_arch(x: &str) -> Result<NormalLoaderType, String> {
    x.parse()
}

fn parse_model_dtype(x: &str) -> Result<ModelDType, String> {
    x.parse()
}

#[derive(Debug, Clone, Subcommand, serde::Deserialize, serde::Serialize)]
pub enum ModelSelected {
    /// Select the model from a toml file
    Toml {
        /// .toml file containing the selector configuration.
        #[arg(short, long)]
        file: String,
    },

    /// Select a model for running via auto loader
    Run {
        /// Model ID to load from. May be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If specified, it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        #[arg(short, long)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[arg(short, long)]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        #[arg(short, long)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        #[arg(short, long)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[arg(short, long)]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// Only supported on specific multimodal models.
        #[arg(short = 'e', long)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        #[arg(long)]
        max_num_images: Option<usize>,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific multimodal models.
        #[arg(long)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[arg(long)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[arg(long)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[arg(long)]
        matformer_slice_name: Option<String>,
    },

    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_arch)]
        #[serde(default)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        #[serde(default = "default_model_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        #[serde(default)]
        topology: Option<String>,

        #[allow(rustdoc::bare_urls)]
        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: <https://arxiv.org/abs/2310.02410>).
        #[arg(short, long)]
        #[serde(default)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[arg(short, long)]
        #[serde(default)]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[arg(short, long)]
        #[serde(default)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--calibration-file/-c`
        #[arg(short, long)]
        #[serde(default)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        #[arg(short, long)]
        #[serde(default)]
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[arg(long)]
        #[serde(default)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[arg(long)]
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },



    /// Select a GGUF model.
    GGUF {
        /// `tok_model_id` optionally overrides embedded configuration and tokenizer assets.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use semicolons to separate multiple files.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Path to a tokenizer JSON file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Multimodal projector filename(s), separated by semicolons.
        #[arg(long)]
        mmproj_filename: Option<String>,

        /// Dynamic LoRA adapters to preload.
        #[arg(skip)]
        #[serde(default)]
        lora_adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime limits. `None` disables dynamic LoRA.
        #[arg(skip)]
        #[serde(default)]
        lora_runtime_config: Option<LoraRuntimeConfig>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        #[arg(long)]
        organization: Option<IsqOrganization>,

        /// UQFF path and quantization types to write.
        #[arg(long)]
        write_uqff: Option<UqffWriteConfig>,

        /// Imatrix file to use while requantizing the GGUF weights.
        #[arg(long)]
        imatrix: Option<PathBuf>,

        /// Calibration file used to generate an imatrix while requantizing the GGUF weights.
        #[arg(long)]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length.
        #[arg(short = 'e', long)]
        #[serde(default)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for automatic device mapping.
        #[arg(long)]
        #[serde(default)]
        max_num_images: Option<usize>,

        /// Maximum expected image edge length for automatic device mapping.
        #[arg(long)]
        #[serde(default)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[arg(long)]
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,

        /// Path to a local Matryoshka Transformer configuration CSV file.
        #[arg(long)]
        #[serde(default)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use.
        #[arg(long)]
        #[serde(default)]
        matformer_slice_name: Option<String>,
    },



    /// Select a GGML model.
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },




    /// Select a diffusion model, without quantization or adapters
    #[command(name = "diffusion")]


    /// Select multi-model mode with configuration file
    #[command(name = "multi-model")]
    MultiModel {
        /// Multi-model configuration file path (JSON format)
        #[arg(short, long)]
        config: String,

        /// Default model ID to use when no model is specified in requests
        #[arg(short, long)]
        default_model_id: Option<String>,
    },

}
