//! gbnf-probe — empirical GBNF measurement harness (plan todo 7 / ADR-GBNF-GO).
//!
//! Loads a GGUF via llama-cpp-2 and decodes every prompt TWICE:
//!   1. `with_grammar`   — the production sampler chain
//!      [`LlamaSampler::grammar_lazy`] + trigger `{"name"` built through
//!      `kitty_runtime::llama::grammar_lazy_sampler_for` (identical to what
//!      `LlamaCppClient` uses), followed by a dist sampler.
//!   2. `without_grammar` — dist-only baseline.
//!
//! Raw generations are dumped as JSON; benches/ab_gbnf.py validates them with
//! the same `GrammarConfig::validate` semantics as the spike. Nothing here
//! fabricates numbers: whatever decode produces is what gets reported.
//!
//! Usage:
//!   gbnf-probe --model PATH.gguf --prompts-file prompts.json --out out.json \
//!              [--max-tokens 128] [--seed 42] [--gpu-layers 999] [--n-ctx 2048]

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use kitty_runtime::llama::grammar_lazy_sampler_for;
use kitty_runtime::tools::ToolDefinition;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

const SYSTEM_PROMPT: &str = "You are a coding agent. When asked to perform a file or shell operation, reply with ONLY a JSON tool call object on a single line, shaped exactly {\"name\": \"...\", \"arguments\": { ... }}, using one of the tools: Read(path), Write(path, content), Edit(path, hash, new), Bash(command). No prose, no markdown fences.";

/// Tool set mirroring benches/ab_gbnf.py TINY_SCHEMAS required fields.
fn bench_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "Read",
            "Read a text file.",
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        ),
        ToolDefinition::new(
            "Write",
            "Write content to a file.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        ),
        ToolDefinition::new(
            "Edit",
            "Replace one line identified by its hash prefix.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "hash": { "type": "string" },
                    "new": { "type": "string" }
                },
                "required": ["path", "hash", "new"]
            }),
        ),
        ToolDefinition::new(
            "Bash",
            "Run a shell command.",
            serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        ),
    ]
}

/// Qwen2.5 chat template applied manually; `str_to_token` tokenizes the
/// `<|im_start|>`/`<|im_end|>` markers as special tokens (tokenize_special=true).
fn qwen_prompt(user_msg: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Which grammar constraint an arm uses.
#[derive(Clone, Copy, PartialEq)]
enum GrammarArm {
    /// Production wiring: LlamaSampler::grammar_lazy, trigger `{"name"`.
    Lazy,
    /// Diagnostic: grammar constrained from the very first token.
    Full,
    /// Baseline: unconstrained dist sampling.
    Off,
}

impl GrammarArm {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "lazy" => Some(Self::Lazy),
            "full" => Some(Self::Full),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Lazy => "with_grammar_lazy",
            Self::Full => "with_grammar_full",
            Self::Off => "without_grammar",
        }
    }
}

/// Decode one prompt to completion of max_tokens or EOG. A fresh context and
/// sampler are used per call so grammar state never leaks across prompts.
#[allow(clippy::too_many_arguments)]
fn generate(
    model: &LlamaModel,
    backend: &LlamaBackend,
    user_msg: &str,
    arm: GrammarArm,
    max_tokens: u32,
    seed: u32,
    n_ctx: u32,
) -> Result<String, String> {
    let prompt = qwen_prompt(user_msg);
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("tokenize: {e:?}"))?;
    if tokens.len() + max_tokens as usize + 8 > n_ctx as usize {
        return Err(format!(
            "prompt {} tokens + {} gen exceeds n_ctx {}",
            tokens.len(),
            max_tokens,
            n_ctx
        ));
    }

    let ctx_params =
        LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(n_ctx).expect("n_ctx > 0")));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("context: {e:?}"))?;

    let mut batch = LlamaBatch::new(tokens.len() + 1, 1);
    for (i, t) in tokens.iter().enumerate() {
        let last = i + 1 == tokens.len();
        batch
            .add(*t, i as i32, &[0], last)
            .map_err(|e| format!("batch add prompt: {e:?}"))?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("prefill: {e:?}"))?;

    // Same per-prompt seed across arms keeps the A/B comparable up to the
    // point where the lazy grammar starts constraining.
    let mut sampler = match arm {
        GrammarArm::Full => {
            let grammar = kitty_runtime::llama::tool_schemas_to_gbnf_pub(&bench_tools());
            LlamaSampler::chain_simple([
                LlamaSampler::grammar(model, &grammar, "root")
                    .map_err(|e| format!("grammar: {e:?}"))?,
                LlamaSampler::dist(seed),
            ])
        }
        GrammarArm::Lazy => LlamaSampler::chain_simple([
            grammar_lazy_sampler_for(model, &bench_tools())?,
            LlamaSampler::dist(seed),
        ]),
        GrammarArm::Off => LlamaSampler::chain_simple([LlamaSampler::dist(seed)]),
    };

    let mut out = String::new();
    let mut n_cur = tokens.len() as i32;
    for _ in 0..max_tokens {
        let idx = batch.n_tokens() - 1;
        let token = sampler.sample(&ctx, idx);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        // Buffer must be generous: a fixed small buffer silently drops longer
        // pieces (e.g. the literal token "arguments" is 9 bytes). Token
        // pieces are bounded well below 512 bytes; the error variant is
        // private so we cannot retry-by-size, only skip.
        if let Ok(bytes) = model.token_to_piece_bytes(token, 512, true, None) {
            out.push_str(&String::from_utf8_lossy(&bytes));
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("batch add gen: {e:?}"))?;
        n_cur += 1;
        ctx.decode(&mut batch).map_err(|e| format!("decode: {e:?}"))?;
    }
    Ok(out)
}

struct Args {
    model: PathBuf,
    prompts_file: PathBuf,
    out: PathBuf,
    max_tokens: u32,
    seed: u32,
    gpu_layers: u32,
    n_ctx: u32,
    arms: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut prompts_file = None;
    let mut out = PathBuf::from("gbnf_probe_out.json");
    let mut max_tokens = 128u32;
    let mut seed = 42u32;
    let mut gpu_layers = 999u32;
    let mut n_ctx = 2048u32;
    let mut arms_arg: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        let need = |name: &str, i: usize| -> Result<String, String> {
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name}: missing value"))
        };
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(need("--model", i)?));
            }
            "--prompts-file" => {
                i += 1;
                prompts_file = Some(PathBuf::from(need("--prompts-file", i)?));
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(need("--out", i)?);
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = need("--max-tokens", i)?.parse().map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = need("--seed", i)?.parse().map_err(|e| format!("{e}"))?;
            }
            "--gpu-layers" => {
                i += 1;
                gpu_layers = need("--gpu-layers", i)?.parse().map_err(|e| format!("{e}"))?;
            }
            "--n-ctx" => {
                i += 1;
                n_ctx = need("--n-ctx", i)?.parse().map_err(|e| format!("{e}"))?;
            }
            "--arms" => {
                i += 1;
                arms_arg = Some(need("--arms", i)?);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
        i += 1;
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompts_file: prompts_file.ok_or("--prompts-file is required")?,
        out,
        max_tokens,
        seed,
        gpu_layers,
        n_ctx,
        arms: arms_arg,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let prompts_raw = std::fs::read_to_string(&args.prompts_file)
        .map_err(|e| format!("read {}: {e}", args.prompts_file.display()))?;
    let prompts: Vec<String> = serde_json::from_str(&prompts_raw)
        .map_err(|e| format!("parse prompts file: {e}"))?;
    if prompts.is_empty() {
        return Err("prompts file is empty".into());
    }

    eprintln!(
        "[gbnf-probe] loading {} (gpu_layers={} n_ctx={})",
        args.model.display(),
        args.gpu_layers,
        args.n_ctx
    );
    let backend = LlamaBackend::init().map_err(|e| format!("backend init: {e:?}"))?;
    let mparams = LlamaModelParams::default().with_n_gpu_layers(args.gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &args.model, &mparams)
        .map_err(|e| format!("GGUF load failed: {e:?}"))?;
    eprintln!("[gbnf-probe] model loaded; decoding {} prompts", prompts.len());

    // Default arms: production lazy grammar + unconstrained baseline.
    let arm_list: Vec<GrammarArm> = match &args.arms {
        Some(spec) => spec
            .split(',')
            .map(|s| GrammarArm::parse(s.trim()).ok_or_else(|| format!("unknown arm '{s}'")))
            .collect::<Result<_, _>>()?,
        None => vec![GrammarArm::Lazy, GrammarArm::Off],
    };

    let mut arms: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for arm in arm_list {
        let mut rows = Vec::with_capacity(prompts.len());
        for (idx, user_msg) in prompts.iter().enumerate() {
            let per_prompt_seed = args.seed.wrapping_add(idx as u32);
            let text = generate(
                &model,
                &backend,
                user_msg,
                arm,
                args.max_tokens,
                per_prompt_seed,
                args.n_ctx,
            )
            .map_err(|e| format!("generate arm={} prompt#{idx}: {e}", arm.name()))?;
            if (idx + 1) % 10 == 0 || idx + 1 == prompts.len() {
                eprintln!("[gbnf-probe] {} {}/{}", arm.name(), idx + 1, prompts.len());
            }
            rows.push(serde_json::json!({ "index": idx, "text": text }));
        }
        arms.insert(arm.name().to_string(), serde_json::Value::Array(rows));
    }

    let result = serde_json::json!({
        "model_file": Path::new(&args.model)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown"),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "gpu_layers": args.gpu_layers,
        "n_ctx": args.n_ctx,
        "arms": arms,
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write {}: {e}", args.out.display()))?;
    eprintln!("[gbnf-probe] wrote {}", args.out.display());
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gbnf-probe error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
