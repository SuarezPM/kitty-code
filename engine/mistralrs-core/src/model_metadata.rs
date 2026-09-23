//! Static per-architecture metadata powering the supported-models reference.
//! One architecture often serves several brand names (Qwen 3.5/3.6, LFM2/LFM2.5);
//! `families` carries those names while the loader enum stays the machine-matchable key.
//! `supported_models_matches_committed` keeps docs/.../reference/supported-models.md in sync;
//! refresh with `cargo test -p mistralrs-core regenerate_supported_models -- --ignored`.

use std::fmt::Write as _;

use strum::IntoEnumIterator;

use crate::pipeline::SupportedModality;
use crate::NormalLoaderType;

use SupportedModality::{Audio, Embedding, Text, Video, Vision};

#[derive(Clone, Copy)]
pub struct ModelExample {
    pub repo: &'static str,
    pub label: &'static str,
}

// A macro so each call expands to a struct literal, which const-promotes to `&'static`
// (a `const fn` call in an `&[..]` literal would not promote).
macro_rules! ex {
    ($repo:expr) => {
        ModelExample {
            repo: $repo,
            label: "",
        }
    };
    ($repo:expr, $label:expr) => {
        ModelExample {
            repo: $repo,
            label: $label,
        }
    };
}

pub struct ArchMetadata {
    pub families: &'static [&'static str],
    pub modalities: &'static [SupportedModality],
    pub examples: &'static [ModelExample],
}

impl NormalLoaderType {
    pub fn arch_metadata(&self) -> ArchMetadata {
        let m = &[Text][..];
        match self {
            Self::Mistral => ArchMetadata {
                families: &["Mistral"],
                modalities: m,
                examples: &[ex!("mistralai/Mistral-7B-Instruct-v0.3")],
            },
            Self::Gemma => ArchMetadata {
                families: &["Gemma"],
                modalities: m,
                examples: &[ex!("google/gemma-7b-it")],
            },
            Self::Mixtral => ArchMetadata {
                families: &["Mixtral"],
                modalities: m,
                examples: &[ex!("mistralai/Mixtral-8x7B-Instruct-v0.1")],
            },
            Self::Llama => ArchMetadata {
                families: &["Llama 2", "Llama 3.x"],
                modalities: m,
                examples: &[ex!("meta-llama/Llama-3.1-8B-Instruct")],
            },
            Self::Phi2 => ArchMetadata {
                families: &["Phi-2"],
                modalities: m,
                examples: &[ex!("microsoft/phi-2")],
            },
            Self::Phi3 => ArchMetadata {
                families: &["Phi-3", "Phi-3.5"],
                modalities: m,
                examples: &[ex!("microsoft/Phi-3-medium-4k-instruct")],
            },
            Self::Qwen2 => ArchMetadata {
                families: &["Qwen2", "Qwen2.5"],
                modalities: m,
                examples: &[
                    ex!("Qwen/Qwen2.5-7B-Instruct", "2.5"),
                    ex!("Qwen/Qwen2-7B-Instruct", "2"),
                ],
            },
            Self::Gemma2 => ArchMetadata {
                families: &["Gemma 2"],
                modalities: m,
                examples: &[ex!("google/gemma-2-9b-it")],
            },
            Self::Starcoder2 => ArchMetadata {
                families: &["Starcoder2"],
                modalities: m,
                examples: &[ex!("bigcode/starcoder2-7b")],
            },
            Self::Phi3_5MoE => ArchMetadata {
                families: &["Phi-3.5-MoE"],
                modalities: m,
                examples: &[ex!("microsoft/Phi-3.5-MoE-instruct")],
            },
            Self::DeepSeekV2 => ArchMetadata {
                families: &["DeepSeek-V2"],
                modalities: m,
                examples: &[ex!("deepseek-ai/DeepSeek-V2-Chat")],
            },
            Self::DeepSeekV3 => ArchMetadata {
                families: &["DeepSeek-V3", "DeepSeek-R1"],
                modalities: m,
                examples: &[
                    ex!("deepseek-ai/DeepSeek-V3", "V3"),
                    ex!("deepseek-ai/DeepSeek-R1", "R1"),
                ],
            },
            Self::Qwen3 => ArchMetadata {
                families: &["Qwen3"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-4B")],
            },
            Self::GLM4 => ArchMetadata {
                families: &["GLM-4"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4-32B-0414")],
            },
            Self::GLM4MoeLite => ArchMetadata {
                families: &["GLM-4.7-Flash"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4.7-Flash")],
            },
            Self::GLM4Moe => ArchMetadata {
                families: &["GLM-4.7"],
                modalities: m,
                examples: &[ex!("zai-org/GLM-4.7")],
            },
            Self::Qwen3Moe => ArchMetadata {
                families: &["Qwen3 MoE"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-30B-A3B")],
            },
            Self::SmolLm3 => ArchMetadata {
                families: &["SmolLM3"],
                modalities: m,
                examples: &[ex!("HuggingFaceTB/SmolLM3-3B")],
            },
            Self::GraniteMoeHybrid => ArchMetadata {
                families: &["Granite 4.0"],
                modalities: m,
                examples: &[ex!("ibm-granite/granite-4.0-micro")],
            },
            Self::GptOss => ArchMetadata {
                families: &["GPT-OSS"],
                modalities: m,
                examples: &[
                    ex!("openai/gpt-oss-20b", "20b"),
                    ex!("openai/gpt-oss-120b", "120b"),
                ],
            },
            Self::HunYuanDenseV1 => ArchMetadata {
                families: &["HunYuan"],
                modalities: m,
                examples: &[ex!("tencent/Hunyuan-7B-Instruct")],
            },
            Self::HunYuanMoEV1 => ArchMetadata {
                families: &["HunYuan MoE"],
                modalities: m,
                examples: &[ex!("tencent/Hunyuan-A13B-Instruct")],
            },
            Self::Qwen3Next => ArchMetadata {
                families: &["Qwen3-Next", "Qwen3-Coder-Next"],
                modalities: m,
                examples: &[ex!("Qwen/Qwen3-Next-80B-A3B-Instruct")],
            },
            Self::Qwen3_5 => ArchMetadata {
                families: &["Qwen3.5"],
                modalities: m,
                examples: &[],
            },
            Self::Lfm2 => ArchMetadata {
                families: &["LFM2", "LFM2.5"],
                modalities: m,
                examples: &[
                    ex!("LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5"),
                    ex!("LiquidAI/LFM2-1.2B", "LFM2"),
                ],
            },
            Self::Lfm2Moe => ArchMetadata {
                families: &["LFM2 MoE", "LFM2.5 MoE"],
                modalities: m,
                examples: &[
                    ex!("LiquidAI/LFM2.5-8B-A1B", "LFM2.5"),
                    ex!("LiquidAI/LFM2-8B-A1B", "LFM2"),
                ],
            },
            Self::K2Horizon => ArchMetadata {
                families: &["K2-Horizon"],
                modalities: m,
                examples: &[ex!("IFM/K2-Horizon-3.7B", "3.7B")],
            },
        }
    }
}

impl NormalLoaderType {
    pub fn config_arch(&self) -> &'static str {
        match self {
            Self::Mistral => "MistralForCausalLM",
            Self::Gemma => "GemmaForCausalLM",
            Self::Mixtral => "MixtralForCausalLM",
            Self::Llama => "LlamaForCausalLM",
            Self::Phi2 => "PhiForCausalLM",
            Self::Phi3 => "Phi3ForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Gemma2 => "Gemma2ForCausalLM",
            Self::Starcoder2 => "Starcoder2ForCausalLM",
            Self::Phi3_5MoE => "PhiMoEForCausalLM",
            Self::DeepSeekV2 => "DeepseekV2ForCausalLM",
            Self::DeepSeekV3 => "DeepseekV3ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
            Self::GLM4 => "Glm4ForCausalLM",
            Self::GLM4MoeLite => "Glm4MoeLiteForCausalLM",
            Self::GLM4Moe => "Glm4MoeForCausalLM",
            Self::Qwen3Moe => "Qwen3MoeForCausalLM",
            Self::SmolLm3 => "SmolLM3ForCausalLM",
            Self::GraniteMoeHybrid => "GraniteMoeHybridForCausalLM",
            Self::GptOss => "GptOssForCausalLM",
            Self::HunYuanDenseV1 => "HunYuanDenseV1ForCausalLM",
            Self::HunYuanMoEV1 => "HunYuanMoEV1ForCausalLM",
            Self::Qwen3Next => "Qwen3NextForCausalLM",
            Self::Qwen3_5 => "Qwen3_5ForCausalLM",
            Self::Lfm2 => "Lfm2ForCausalLM",
            Self::Lfm2Moe => "Lfm2MoeForCausalLM",
            Self::K2Horizon => "K2HorizonForCausalLM",
        }
    }
}

const HEADER: &str = r#"---
title: Supported models
description: Architectures supported by mistral.rs, and how to tell if yours is one of them.
---

<!-- Generated from the loader registry by mistralrs-core model_metadata. Do not edit by hand. -->

## Is my model supported?

mistral.rs auto-detects the architecture from a repo's `config.json`. To check yours:

1. Open the model's `config.json` on Hugging Face and read the `architectures` field (e.g. `"Qwen3ForCausalLM"`, `"Gemma4ForConditionalGeneration"`).
2. Find the matching row below. Each architecture covers every checkpoint that reports that class, including future fine-tunes and sizes, so the families and examples here are a sample, not the full list.
3. If the architecture is not listed, check the [GGUF compatibility reference](/reference/gguf-support/) or [request model support](https://github.com/EricLBuehler/mistral.rs/issues/156). Use `--arch` only when the checkpoint matches a known architecture.

```bash
mistralrs run -m <model>     # interactive
mistralrs serve -m <model>   # OpenAI-compatible server
```

Expand a listed example to copy a ready-to-run command. One loader often serves several brand names (Qwen 3.5 and 3.6 share `Qwen3_5`; LFM2 and LFM2.5 share `Lfm2`) - the `Model families` column lists them. Behavior that differs from the defaults is collected in [model family notes](/guides/models/model-family-notes/).

The `Architecture` column is the `config.json` `architectures` value. Per-family quantization, thinking, gated-repo, and tool-calling details live in [model family notes](/guides/models/model-family-notes/).

"#;

const FOOTER: &str = r#"## Format and quantization notes

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. See [GGUF support](/reference/gguf-support/) for GGUF compatibility; availability of [UQFF](/reference/uqff-format/), GPTQ, and AWQ artifacts varies by model on Hugging Face.

## Speculative decoding

| Mode | Target architecture | Assistant checkpoint family | Guide |
|---|---|---|---|
| MTP | `Gemma4` | Gemma 4 assistant checkpoints, PagedAttention required | [Speculative decoding (MTP)](/guides/perf/speculative-decoding/) |
| MTP | `Qwen3_5`, `Qwen3.8` | Built-in `mtp.*` head (`--mtp`), PagedAttention required | [Speculative decoding (MTP)](/guides/perf/speculative-decoding/) |
"#;

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|")
}

fn families_cell(meta: &ArchMetadata) -> String {
    md_cell(&meta.families.join(", "))
}

fn examples_cell(meta: &ArchMetadata) -> String {
    if meta.examples.is_empty() {
        return "No published example".to_string();
    }
    let label = |e: &ModelExample| {
        if e.label.is_empty() {
            format!("<code>{}</code>", e.repo)
        } else {
            format!("<code>{}</code> ({})", e.repo, e.label)
        }
    };
    let summary = meta
        .examples
        .iter()
        .map(label)
        .collect::<Vec<_>>()
        .join(", ");
    let body = meta
        .examples
        .iter()
        .map(|e| format!("<code>mistralrs run -m {}</code>", e.repo))
        .collect::<Vec<_>>()
        .join("<br>");
    md_cell(&format!(
        "<details><summary>{summary}</summary>{body}</details>"
    ))
}

fn simple_table(md: &mut String, rows: impl Iterator<Item = (String, ArchMetadata)>) {
    md.push_str("| Architecture | Model families | Example |\n|---|---|---|\n");
    for (name, meta) in rows {
        writeln!(
            md,
            "| `{name}` | {} | {} |",
            families_cell(&meta),
            examples_cell(&meta),
        )
        .unwrap();
    }
    md.push('\n');
}
fn variant_name<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
}

pub fn render_supported_models_markdown() -> String {
    let mut md = String::new();
    md.push_str(HEADER);

    md.push_str("## Text models\n\n");
    simple_table(
        &mut md,
        NormalLoaderType::iter().map(|t| (t.config_arch().to_string(), t.arch_metadata())),
    );

    md.push_str(FOOTER);
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const COMMITTED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/src/content/docs/reference/supported-models.md"
    );
    const REGEN_HINT: &str =
        "cargo test -p mistralrs-core regenerate_supported_models -- --ignored";

    #[test]
    fn supported_models_matches_committed() {
        let rendered = render_supported_models_markdown();
        // Windows runners check out with autocrlf, so normalize before comparing.
        let on_disk = std::fs::read_to_string(COMMITTED)
            .expect("supported-models.md exists")
            .replace("\r\n", "\n");
        assert!(
            on_disk == rendered,
            "committed supported-models.md is out of date; regenerate with: {REGEN_HINT}",
        );
    }

    #[test]
    #[ignore = "writes the committed supported-models.md"]
    fn regenerate_supported_models() {
        std::fs::write(Path::new(COMMITTED), render_supported_models_markdown()).unwrap();
    }

    #[test]
    fn config_archs_round_trip() {
        for t in NormalLoaderType::iter() {
            assert_eq!(
                NormalLoaderType::from_causal_lm_name(t.config_arch()).unwrap(),
                t,
                "{t:?} config_arch does not map back through from_causal_lm_name",
            );
        }
    }
}
