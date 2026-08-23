//! gbnf-probe — empirical GBNF measurement harness (plan todo 7 / ADR-GBNF-GO).
//!
//! Loads a GGUF via llama-cpp-2 and decodes every prompt once per arm:
//!   1. `with_grammar_lazy`   — the production sampler chain
//!      [`LlamaSampler::grammar_lazy`] (word trigger `{"name"` +
//!      tokenizer-derived token-ID triggers) + `json_schema_to_grammar`
//!      over a oneOf of per-tool schemas, built through
//!      `kitty_runtime::llama::grammar_lazy_sampler_for` (identical to what
//!      `LlamaCppClient` uses), followed by a dist sampler.
//!   2. `without_grammar` — dist-only baseline.
//!
//! Raw generations are dumped as JSON; benches/ab_gbnf.py validates them with
//! the same `GrammarConfig::validate` semantics as the spike. Nothing here
//! fabricates numbers: whatever decode produces is what gets reported.
//!
//! NOTE: `LlamaSampler::sample` already accepts the sampled token into every
//! sampler of the chain (llama.cpp `llama_sampler_sample`). Never call
//! `accept` again in the generation loop — double-accepting mangles lazy
//! trigger buffers and crashes constrained grammars (task E4 finding).
//!
//! Usage:
//!   gbnf-probe --model PATH.gguf --prompts-file prompts.json --out out.json \
//!              [--max-tokens 128] [--seed 42] [--gpu-layers 999] [--n-ctx 2048]
//!
//! Diagnostics (task E4, ADR-GBNF-GO trigger fix):
//!   --diag-tokens        vocab-only load; prints how THIS tokenizer splits
//!                        the trigger phrases + derived trigger token IDs.
//!   --arms canary-word   lazy grammar `root ::= "{\"name\""` + word trigger;
//!                        output truncates to exactly `{"name"` IFF fired.
//!   --arms canary-tok    same canary grammar with token-ID triggers only.
//!   --arms lazytok       production grammar, token-ID triggers only.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use kitty_runtime::llama::{
    build_lazy_sampler_with, derive_trigger_tokens_pub, grammar_lazy_sampler_for,
    tool_schemas_to_gbnf_pub,
};
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
    /// Production wiring: LlamaSampler::grammar_lazy, word trigger `{"name"`
    /// + tokenizer-derived token-ID triggers.
    Lazy,
    /// Diagnostic: production grammar, token-ID triggers ONLY (no words) —
    /// isolates the token-trigger contribution.
    LazyTok,
    /// Fire-canary: lazy grammar whose root is exactly the literal
    /// `{"name"`. If the trigger fires, the replayed prefix completes the
    /// only rule, every stack empties and llama.cpp allows ONLY EOG — so a
    /// fired run truncates to exactly `{"name"`. Natural long output means
    /// the trigger never fired.
    CanaryWord,
    /// Same canary grammar, token-ID triggers only.
    CanaryTok,
    /// Diagnostic: grammar constrained from the very first token.
    Full,
    /// Baseline: unconstrained dist sampling.
    Off,
}

/// Canary grammar: root is exactly the trigger literal. Once triggered and
/// replayed, the rule completes and only EOG tokens remain allowed.
const CANARY_GRAMMAR: &str = "root ::= \"{\\\"name\\\"\"\n";

impl GrammarArm {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "lazy" => Some(Self::Lazy),
            "lazytok" => Some(Self::LazyTok),
            "canary-word" => Some(Self::CanaryWord),
            "canary-tok" => Some(Self::CanaryTok),
            "full" => Some(Self::Full),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Lazy => "with_grammar_lazy",
            Self::LazyTok => "with_grammar_lazy_tok",
            Self::CanaryWord => "canary_word_trigger",
            Self::CanaryTok => "canary_token_triggers",
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
    ctx.decode(&mut batch)
        .map_err(|e| format!("prefill: {e:?}"))?;

    // Same per-prompt seed across arms keeps the A/B comparable up to the
    // point where the lazy grammar starts constraining.
    let mut sampler = match arm {
        GrammarArm::Full => {
            let grammar = tool_schemas_to_gbnf_pub(&bench_tools());
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
        GrammarArm::LazyTok => {
            let grammar = tool_schemas_to_gbnf_pub(&bench_tools());
            let toks = derive_trigger_tokens_pub(model);
            eprintln!("[gbnf-probe] lazytok trigger tokens: {toks:?}");
            LlamaSampler::chain_simple([
                build_lazy_sampler_with(model, &grammar, &[], &toks)?,
                LlamaSampler::dist(seed),
            ])
        }
        GrammarArm::CanaryWord => LlamaSampler::chain_simple([
            build_lazy_sampler_with(
                model,
                CANARY_GRAMMAR,
                &["{\"name\"".to_string()],
                &[],
            )?,
            LlamaSampler::dist(seed),
        ]),
        GrammarArm::CanaryTok => {
            let toks = derive_trigger_tokens_pub(model);
            eprintln!("[gbnf-probe] canary-tok trigger tokens: {toks:?}");
            LlamaSampler::chain_simple([
                build_lazy_sampler_with(model, CANARY_GRAMMAR, &[], &toks)?,
                LlamaSampler::dist(seed),
            ])
        }
        GrammarArm::Off => LlamaSampler::chain_simple([LlamaSampler::dist(seed)]),
    };

    let mut out = String::new();
    for (gen_idx, _) in (0..max_tokens).enumerate() {
        let idx = batch.n_tokens() - 1;
        // LlamaSampler::sample == llama_sampler_sample: it APPLIES the chain
        // AND ACCEPTS the chosen token into every sampler in the chain
        // (llama-sampler.cpp llama_sampler_sample, final line). An explicit
        // `sampler.accept(token)` here would double-accept: lazy-grammar
        // trigger pieces get duplicated (`{"{"namename"...`) so word triggers
        // can never match, and constrained grammars get fed each token twice,
        // driving stacks empty -> GGML_ASSERT(!stacks.empty()) crash.
        let token = sampler.sample(&ctx, idx);
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
            .add(token, tokens.len() as i32 + gen_idx as i32, &[0], true)
            .map_err(|e| format!("batch add gen: {e:?}"))?;
        ctx.decode(&mut batch)
            .map_err(|e| format!("decode: {e:?}"))?;
    }
    Ok(out)
}

struct Args {
    model: PathBuf,
    prompts_file: Option<PathBuf>,
    out: PathBuf,
    max_tokens: u32,
    seed: u32,
    gpu_layers: u32,
    n_ctx: u32,
    arms: Option<String>,
    diag_tokens: bool,
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
    let mut diag_tokens = false;
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
            "--diag-tokens" => {
                diag_tokens = true;
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(need("--out", i)?);
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = need("--max-tokens", i)?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                seed = need("--seed", i)?.parse().map_err(|e| format!("{e}"))?;
            }
            "--gpu-layers" => {
                i += 1;
                gpu_layers = need("--gpu-layers", i)?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
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
        prompts_file,
        out,
        max_tokens,
        seed,
        gpu_layers,
        n_ctx,
        arms: arms_arg,
        diag_tokens,
    })
}

/// Tokenizer ground truth: how does THIS model split the trigger phrases?
/// Vocab-only load, prints JSON `{ phrase: [{id, piece}...] }` plus the
/// derived production trigger-token IDs.
fn run_diag_tokens(args: &Args) -> Result<(), String> {
    let backend = LlamaBackend::init().map_err(|e| format!("backend init: {e:?}"))?;
    let mparams = LlamaModelParams::default().with_vocab_only(true);
    let model = LlamaModel::load_from_file(&backend, &args.model, &mparams)
        .map_err(|e| format!("GGUF load failed: {e:?}"))?;

    let phrases = [
        "{\"name\"",
        "{\"name\": \"Read\"",
        "{\"",
        "{",
        " {",
        "\n{",
        "{\"name\":\"Read\",\"arguments\":{\"path\":\"x\"}}",
    ];
    let mut report = serde_json::Map::new();
    for p in phrases {
        let ids = model
            .str_to_token(p, AddBos::Never)
            .map_err(|e| format!("tokenize {p:?}: {e:?}"))?;
        let rows: Vec<serde_json::Value> = ids
            .iter()
            .map(|t| {
                let piece = model
                    .token_to_piece_bytes(*t, 512, true, None)
                    .map(|b| String::from_utf8_lossy(&b).to_string())
                    .unwrap_or_else(|_| "<err>".into());
                serde_json::json!({ "id": t.0, "piece": piece })
            })
            .collect();
        report.insert(p.to_string(), serde_json::Value::Array(rows));
    }
    let derived: Vec<i32> = derive_trigger_tokens_pub(&model).iter().map(|t| t.0).collect();
    let out = serde_json::json!({
        "model_file": args.model.display().to_string(),
        "tokenizations": report,
        "derived_trigger_token_ids": derived,
    });
    println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.diag_tokens {
        return run_diag_tokens(&args);
    }
    let prompts_file = args
        .prompts_file
        .as_ref()
        .ok_or("--prompts-file is required (or use --diag-tokens)")?;
    let prompts_raw =
        std::fs::read_to_string(prompts_file).map_err(|e| format!("read {}: {e}", prompts_file.display()))?;
    let prompts: Vec<String> =
        serde_json::from_str(&prompts_raw).map_err(|e| format!("parse prompts file: {e}"))?;
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
    eprintln!(
        "[gbnf-probe] model loaded; decoding {} prompts",
        prompts.len()
    );

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
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?,
    )
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
