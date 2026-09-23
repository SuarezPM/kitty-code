use super::hf::{
    build_api_with_cache, hf_access_error, remote_issue_from_api_error, RemoteAccessIssue,
};
use super::{
    Loader, ModelKind, ModelPaths, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, TokenSource,
};
use crate::utils::progress::ProgressScopeGuard;
use crate::Ordering;
use crate::{
    AutoDeviceMapParams, DeviceMapSetting, IsqType, PagedAttentionConfig, Pipeline, TryIntoDType,
};
use anyhow::Result;
use candle_core::Device;
use hf_hub::{
    api::sync::{ApiError, ApiRepo},
    Cache, Repo, RepoType,
};
use serde::Deserialize;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Automatically selects the appropriate loader based on repository/config metadata.
pub struct AutoLoader {
    model_id: String,
    normal_builder: Mutex<Option<NormalLoaderBuilder>>,
    loader: Mutex<Option<Box<dyn Loader>>>,
    hf_cache_path: Option<PathBuf>,
    hf_config_overrides: Option<super::HfConfigOverrides>,
    max_model_len: Option<usize>,
}

pub struct AutoLoaderBuilder {
    normal_cfg: NormalSpecificConfig,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    model_id: String,
    jinja_explicit: Option<String>,
    no_kv_cache: bool,
    hf_cache_path: Option<PathBuf>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

impl AutoLoaderBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normal_cfg: NormalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: String,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        Self {
            normal_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            hf_cache_path: None,
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

    pub fn hf_cache_path(mut self, path: PathBuf) -> Self {
        self.hf_cache_path = Some(path);
        self
    }

    pub fn build(self) -> Box<dyn Loader> {
        let Self {
            normal_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            hf_cache_path,
            mtp,
            encoder_cache_memory_bytes,
        } = self;

        let hf_config_overrides = normal_cfg.hf_config_overrides.clone();
        let max_model_len = normal_cfg.max_model_len;
        let mut normal_builder = NormalLoaderBuilder::new(
            normal_cfg,
            chat_template.clone(),
            tokenizer_json.clone(),
            Some(model_id.clone()),
            no_kv_cache,
            jinja_explicit.clone(),
        );
        if let Some(ref path) = hf_cache_path {
            normal_builder = normal_builder.hf_cache_path(path.clone());
        }
        normal_builder = normal_builder.with_mtp(mtp);

        Box::new(AutoLoader {
            model_id,
            normal_builder: Mutex::new(Some(normal_builder)),
            loader: Mutex::new(None),
            hf_cache_path,
            hf_config_overrides,
            max_model_len,
        })
    }
}

#[derive(Deserialize)]
struct AutoConfig {
    #[serde(default)]
    architectures: Vec<String>,
}

struct ConfigArtifacts {
    contents: Option<String>,
    sentence_transformers_present: bool,
    repo_files: Vec<String>,
    remote_access_issue: Option<RemoteAccessIssue>,
}

enum Detected {
    Normal(NormalLoaderType),
}

impl AutoLoader {
    fn try_get_file(
        api: &ApiRepo,
        model_id: &Path,
        file: &str,
        revision: &str,
    ) -> std::result::Result<Option<PathBuf>, ApiError> {
        crate::pipeline::hf::try_get_file(api, model_id, file, revision)
    }

    fn list_local_repo_files(model_root: &Path) -> Vec<String> {
        fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    collect_files(root, &path, out)?;
                } else if let Ok(rel) = path.strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
            Ok(())
        }

        if !model_root.is_dir() {
            return Vec::new();
        }

        let mut files = Vec::new();
        if collect_files(model_root, model_root, &mut files).is_err() {
            return Vec::new();
        }
        files
    }

    fn read_config_from_path(&self, paths: &dyn ModelPaths) -> Result<ConfigArtifacts> {
        let config_path = paths.get_config_filename();
        let contents = match std::fs::read_to_string(config_path) {
            Ok(contents) => Some(contents),
            Err(err) if err.kind() == io::ErrorKind::NotFound => None,
            Err(err) => return Err(err.into()),
        };
        let contents = contents
            .map(|config| self.apply_config_overrides(&config))
            .transpose()?;
        let model_root = Path::new(&self.model_id);
        let repo_files = if model_root.exists() {
            Self::list_local_repo_files(model_root)
        } else {
            Vec::new()
        };
        let sentence_transformers_present = Self::has_sentence_transformers_sibling(config_path)
            || repo_files
                .iter()
                .any(|f| f == "config_sentence_transformers.json");
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue: None,
        })
    }

    fn read_config_from_hf(
        &self,
        revision: Option<String>,
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<ConfigArtifacts> {
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        let api = build_api_with_cache(token_source, !silent, Some(cache))?;
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = Path::new(&self.model_id);
        let mut remote_access_issue = None;
        let contents = match Self::try_get_file(&api, model_id, "config.json", &revision) {
            Ok(Some(path)) => Some(std::fs::read_to_string(&path)?),
            Ok(None) => None,
            Err(err) => {
                let issue =
                    remote_issue_from_api_error(model_id, Some("config.json"), &revision, &err);
                warn!(
                    "Auto loader could not fetch `config.json` for `{}`: {}",
                    self.model_id, issue.message
                );
                remote_access_issue = Some(issue);
                None
            }
        };
        let contents = contents
            .map(|config| self.apply_config_overrides(&config))
            .transpose()?;
        let sentence_transformers_present =
            model_id.join("config_sentence_transformers.json").exists()
                || Self::fetch_sentence_transformers_config(&api, model_id, &revision);
        let repo_files = if model_id.exists() {
            Self::list_local_repo_files(model_id)
        } else {
            crate::api_dir_list!(api, model_id, false, &revision).collect::<Vec<_>>()
        };
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue,
        })
    }

    fn has_sentence_transformers_sibling(config_path: &Path) -> bool {
        config_path
            .parent()
            .map(|parent| parent.join("config_sentence_transformers.json").exists())
            .unwrap_or(false)
    }

    fn apply_config_overrides(&self, config: &str) -> Result<String> {
        match self.hf_config_overrides.as_ref() {
            Some(overrides) => overrides.apply(config),
            None => Ok(config.to_string()),
        }
    }

    fn fetch_sentence_transformers_config(api: &ApiRepo, model_id: &Path, revision: &str) -> bool {
        match crate::pipeline::hf::try_get_file(
            api,
            model_id,
            "config_sentence_transformers.json",
            revision,
        ) {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(err) => {
                debug!(
                    "No `config_sentence_transformers.json` found for `{}`: {err}",
                    model_id.display()
                );
                false
            }
        }
    }

    fn detect(&self, artifacts: &ConfigArtifacts) -> Result<Detected> {
        // kitty-infer is text-only: diffusion/speech/embedding/multimodal
        // markers are refused honestly instead of routing to cut loaders.
        if artifacts.sentence_transformers_present {
            anyhow::bail!(
                "embedding models (`config_sentence_transformers.json`) are not supported by \
                 this text-only build"
            );
        }
        if artifacts.contents.is_none() && artifacts.repo_files.iter().any(|f| f == "params.json") {
            anyhow::bail!(
                "Mistral-native `params.json` models (Voxtral/speech) are not supported by \
                 this text-only build"
            );
        }

        let config = artifacts.contents.as_ref().ok_or_else(|| {
            if let Some(issue) = artifacts.remote_access_issue.as_ref() {
                hf_access_error(Path::new(&self.model_id), issue)
            } else {
                anyhow::anyhow!("Auto loader could not determine model type: missing `config.json`.")
            }
        })?;
        let cfg: AutoConfig = serde_json::from_str(config)?;
        if cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }
        let name = &cfg.architectures[0];
        let tp = NormalLoaderType::from_causal_lm_name(name).map_err(|_| {
            anyhow::anyhow!(
                "architecture `{name}` is not a supported text architecture in this build \
                 (multimodal/embedding/diffusion/speech were amputated, see F0-RECON.md)"
            )
        })?;
        Ok(Detected::Normal(tp))
    }

    fn ensure_loader(&self, detected: Detected) -> Result<()> {
        let mut guard = self.loader.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        match detected {
            Detected::Normal(tp) => {
                let builder = self
                    .normal_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(Some(tp))?;
                *guard = Some(loader);
            }
        }
        Ok(())
    }
}

fn device_map_for_detected(mapper: DeviceMapSetting, _detected: &Detected) -> DeviceMapSetting {
    // Text-only build: detection always yields `Detected::Normal`, so the
    // mapper passes through untouched.
    mapper
}

impl Loader for AutoLoader {
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
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = self.read_config_from_hf(revision.clone(), &token_source, silent)?;
        let detected = self.detect(&config)?;
        let mapper = device_map_for_detected(mapper, &detected);
        self.ensure_loader(detected)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_hf(
                revision,
                token_source,
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
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = self.read_config_from_path(paths)?;
        let detected = self.detect(&config)?;
        let mapper = device_map_for_detected(mapper, &detected);
        self.ensure_loader(detected)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_path(
                paths,
                dtype,
                device,
                silent,
                mapper,
                in_situ_quant,
                paged_attn_config,
            )
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .map(|l| l.get_kind())
            .unwrap_or(ModelKind::Normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> AutoLoader {
        AutoLoader {
            model_id: "Qwen/Qwen3.6-35B-A3B".to_string(),
            normal_builder: Mutex::new(None),
            loader: Mutex::new(None),
            hf_cache_path: None,
            hf_config_overrides: None,
            max_model_len: None,
        }
    }



}
