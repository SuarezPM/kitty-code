//! Kitty CLI — Command-line interface for Kitty Code.

mod bench;

#[cfg(feature = "tui")]
mod tui;

mod claweval;
mod gpqa;
mod mcp;
mod polyglot;
mod tb2;

use anyhow::Result;
use clap::{Parser, Subcommand};
use kitty_harness::agent::{Agent, AgentConfig, RunOverrides};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::tools::standard_tool_definitions;
use kitty_harness::{ModelProfile, ThinkingMode};
use kitty_runtime::{ChatClient, OllamaClient};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// Default path for episodic memory database.
pub(crate) fn default_db_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{}/.kitty/episodes.db", home)
}

/// Resolve the project root used as the Bash-sandbox FS allow-root (Gate-4).
///
/// Walks up from the launch directory to the nearest ancestor containing a
/// project marker, so `kitty run` launched from a subdirectory scopes writes
/// to the actual project instead of that subdirectory. Falls back to the
/// launch directory itself only when NO marker exists anywhere above — and
/// says so loudly, because a silent `$HOME`-wide Landlock allow-root is the
/// exact failure mode this resolution exists to prevent.
pub(crate) fn resolve_project_root() -> PathBuf {
    const MARKERS: &[&str] = &[
        ".git",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "setup.py",
        "go.mod",
    ];
    let start = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut dir: Option<&std::path::Path> = Some(start.as_path());
    while let Some(d) = dir {
        if MARKERS.iter().any(|m| d.join(m).exists()) {
            return d.to_path_buf();
        }
        dir = d.parent();
    }
    warn!(
        cwd = %start.display(),
        "no project marker (.git/Cargo.toml/package.json/pyproject.toml/setup.py/go.mod) \
         found at or above CWD — sandbox fs_allow falls back to CWD itself"
    );
    start
}

// ── Session context-window resolution (f8 T2.2) ─────────────────────

/// Resolved session context window plus where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NumCtxResolution {
    num_ctx: u32,
    /// Provenance of the value: `"env"` | `"native"` | `"fallback"`.
    source: &'static str,
}

/// Floor for a natively probed window (tiny-profile sessions legitimately go
/// lower, but only via explicit env — a probe result is never trusted below).
const NUM_CTX_PROBE_FLOOR: u32 = 4096;
/// Ceiling applied to a probed native length (matches DEFAULT_NUM_CTX).
const NUM_CTX_PROBE_CEILING: u32 = 32768;
/// Short timeout for the one-shot `/api/show` probe: startup must not stall
/// on an unresponsive server.
const SHOW_PROBE_TIMEOUT_SECS: u64 = 5;

/// Pure core of the once-per-process resolution order (unit-tested without
/// env or network):
/// 1. explicit `KITTY_NUM_CTX` wins verbatim — NO probe call;
/// 2. else the probed native context length, clamped to [4096, 32768];
/// 3. else the runtime default ([`kitty_runtime::client::DEFAULT_NUM_CTX`]).
///
/// An env var that is set but unparsable is treated as absent here (the
/// caller warns); `native: None` covers both probe failure and parse failure.
fn resolve_effective_num_ctx(env_raw: Option<&str>, native: Option<u32>) -> NumCtxResolution {
    if let Some(raw) = env_raw {
        let parsed = raw.trim().parse::<u32>().ok().filter(|v| *v > 0);
        if let Some(v) = parsed {
            return NumCtxResolution {
                num_ctx: v,
                source: "env",
            };
        }
    }
    if let Some(n) = native {
        return NumCtxResolution {
            num_ctx: n.clamp(NUM_CTX_PROBE_FLOOR, NUM_CTX_PROBE_CEILING),
            source: "native",
        };
    }
    NumCtxResolution {
        num_ctx: kitty_runtime::client::DEFAULT_NUM_CTX,
        source: "fallback",
    }
}

/// Async wrapper around [`resolve_effective_num_ctx`]: reads the env override
/// and, only when unset-or-invalid, makes ONE short-timeout `show_model`
/// probe. Network errors and timeouts degrade to the fallback loudly.
async fn resolve_session_num_ctx(client: &OllamaClient, model: &str) -> NumCtxResolution {
    let env_raw = std::env::var(kitty_runtime::client::ENV_NUM_CTX).ok();
    let env_usable = env_raw
        .as_deref()
        .map(|raw| raw.trim().parse::<u32>().ok().filter(|v| *v > 0).is_some());
    match (env_raw.as_deref(), env_usable) {
        // 1. Explicit valid override: no probe at all.
        (Some(_), Some(true)) => return resolve_effective_num_ctx(env_raw.as_deref(), None),
        // Set but unparsable: warn, then fall through to probe/fallback.
        (Some(raw), _) => {
            warn!(raw = %raw, "KITTY_NUM_CTX is not a valid u32 — probing the model instead")
        }
        _ => {}
    }

    // 2. One-shot native probe with a short timeout.
    let probe = tokio::time::timeout(
        std::time::Duration::from_secs(SHOW_PROBE_TIMEOUT_SECS),
        client.show_model(model),
    )
    .await;
    match probe {
        Ok(Ok(payload)) => {
            let native = OllamaClient::parse_native_context_length(&payload);
            if native.is_none() {
                warn!("probe payload carried no usable .context_length — using default window");
            }
            return resolve_effective_num_ctx(None, native);
        }
        Ok(Err(e)) => {
            warn!(error = %e, "/api/show probe failed — using default window")
        }
        Err(_) => {
            warn!(
                timeout_secs = SHOW_PROBE_TIMEOUT_SECS,
                "/api/show probe timed out — using default window"
            )
        }
    }
    // 3. Fallback.
    resolve_effective_num_ctx(None, None)
}

#[derive(Parser)]
#[command(name = "kitty")]
#[command(about = "Local coding agent for ornith:9B on consumer GPUs")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Ollama API URL
    #[arg(long, default_value = "http://localhost:11434")]
    pub url: String,

    /// Model name
    #[arg(long, default_value = "ornith:9b")]
    pub model: String,

    /// Maximum thinking/reasoning tokens per turn (default: 2048)
    #[arg(long, default_value = "2048")]
    pub thinking_budget: usize,

    /// Enable knowledge injection (cards from knowledge/ directory)
    /// Recommended after T13: 5 cards, 500 tok/card limit, threshold 2.0
    #[arg(long, global = true)]
    pub knowledge: bool,

    /// Directory containing knowledge cards (default: knowledge/)
    #[arg(long, global = true, default_value = "knowledge")]
    pub knowledge_dir: String,

    /// Context window size in tokens (default: 16384)
    #[arg(long, default_value = "16384")]
    pub context_window: usize,

    /// Maximum iterations before forcing text response (default: 20)
    #[arg(long, default_value = "20")]
    pub max_iterations: usize,

    /// Bash command timeout in seconds (default: 60)
    #[arg(long, default_value = "60")]
    pub bash_timeout: u64,

    /// Model profile preset: nano|tiny|small|medium|large (default: auto —
    /// auto-detects small-tier 2-4B tool-callers and nano-tier sub-3B models
    /// by id shape)
    #[arg(long, global = true, default_value = "auto", value_parser = clap::builder::PossibleValuesParser::new(["nano", "tiny", "small", "medium", "large", "auto"]))]
    pub profile: String,

    /// Enable semantic memory (hybrid BM25+embedding RRF, medium/large only)
    #[arg(long, global = true)]
    pub semantic_memory: bool,

    /// Enable cloud fallback on failure (requires KITTY_CLOUD_API_KEY)
    #[arg(long, global = true)]
    pub cloud_fallback: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run the agent with a prompt
    Run {
        /// The prompt to send to the agent
        prompt: Vec<String>,
        /// Disable episodic memory recall/save
        #[arg(long)]
        no_memory: bool,
    },
    /// Run benchmarks
    Bench {
        /// Benchmark suite: polyglot (225 Exercism exercises with unit test grading),
        /// tb2 (terminal-bench), tb2-official (terminal-bench-2.1 via Harbor CLI),
        /// gpqa, gpqa_diamond, claweval
        /// (gpqa requires --pre-registration <PATH>;
        ///  claweval requires OPENROUTER_API_KEY + claw-eval CLI;
        ///  tb2-official requires harbor CLI + Docker)
        #[arg(default_value = "polyglot")]
        suite: String,
        /// Number of exercises to run
        #[arg(short, long, default_value = "15")]
        limit: usize,
        /// Autorun continuation hops for every exercise (default 0 = off).
        /// Each hop grants one extra full tool-loop cycle on iteration
        /// exhaustion; worst-case LLM calls ≈ iterations × 3 × (hops+1).
        #[arg(long, default_value = "0")]
        hops: usize,
        /// Per-exercise wall-clock cap in seconds. Overrides the
        /// KITTY_BENCH_TIMEOUT_SECS env fallback; both absent ⇒ manifest
        /// cell, else DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS (120). Fase 3:
        /// el default clap DEBE igualar la constante o el flag contaria
        /// siempre como explicito y el manifiesto nunca aplicaria.
        #[arg(long, default_value = "120")]
        timeout_secs: u64,
        /// Output file for results
        #[arg(short, long, default_value = "bench_results.json")]
        out: PathBuf,
        /// Path to JSON exercises file (overrides built-in exercises)
        #[arg(long)]
        exercises: Option<PathBuf>,
        /// Pre-registration manifest for GPQA (required for gpqa/gpqa_diamond). JSON must fix suite, n=198, dataset_id, 40-hex revision, 64-hex sha256, model, limits, grading, compatibility, preconditions, expected_artifacts.
        #[arg(long = "pre-registration", value_name = "PATH")]
        pre_registration: Option<PathBuf>,
        /// Disable episodic memory recall/save
        #[arg(long)]
        no_memory: bool,
        /// Number of ClawEval trials per task (default: 3)
        #[arg(long = "claweval-trials", default_value_t = 3)]
        claweval_trials: usize,
        /// ClawEval parallel tasks (default: 1)
        #[arg(long = "claweval-parallel", default_value_t = 1)]
        claweval_parallel: usize,
        /// Enable ClawEval sandbox mode (requires Docker)
        #[arg(long = "claweval-sandbox")]
        claweval_sandbox: bool,
    },
    /// Run A/B thinking-budget benchmark with interleaved execution
    #[command(name = "bench-ab")]
    BenchAb {
        /// Comma-separated thinking budgets (e.g. 1024,2048,4096)
        #[arg(long, default_value = "1024,2048,4096")]
        thinking: String,
        /// Number of exercises per budget
        #[arg(short, long, default_value = "50")]
        limit: usize,
        /// Autorun continuation hops for every exercise (default 0 = off).
        #[arg(long, default_value = "0")]
        hops: usize,
        /// Per-exercise wall-clock cap in seconds. Overrides the
        /// KITTY_BENCH_TIMEOUT_SECS env fallback; both absent ⇒
        /// DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS (120). Mismo patron
        /// sentinela que bench (D1 gate k>=3): el default clap debe
        /// igualar la constante.
        #[arg(long, default_value = "120")]
        timeout_secs: u64,
        /// Repetitions per (task,budget) pair
        #[arg(long, default_value = "1")]
        reps: usize,
        /// Benchmark suite: polyglot, tb2
        #[arg(long, default_value = "polyglot")]
        suite: String,
        /// Output file for results
        #[arg(short, long, default_value = "bench_ab.json")]
        out: PathBuf,
        /// Interleave execution order (shuffle with seed 42)
        #[arg(long, default_value = "true")]
        interleave: bool,
    },
    /// Show model info and status
    Info,
    /// Start MCP server over stdio (exposes Read/Edit/Write/Bash/Glob/Grep)
    Mcp,
    /// Launch the TUI status display (requires `tui` feature)
    #[cfg(feature = "tui")]
    Tui,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kitty=info,warn".into()),
        )
        .init();

    let cli = Cli::parse();

    // F2 embedded backend: KITTY_BACKEND=kitty-infer selects the in-process
    // kitty-infer GGUF client (KITTY_INFER_MODEL=<path.gguf>). Default (unset
    // or any other value) keeps the Ollama path byte-for-byte as before.
    let use_infer =
        kitty_runtime::backend_selected(std::env::var(kitty_runtime::ENV_BACKEND).ok().as_deref());
    if use_infer {
        info!("backend=kitty-infer (embedded GGUF)");
    }
    let ollama_client: Option<OllamaClient> = if use_infer {
        None
    } else {
        Some(OllamaClient::new(&cli.url))
    };

    // f8 T2.2: resolve the session context window ONCE per process, BEFORE
    // any Agent/client construction, shared by Run/Bench/BenchAb. The value
    // is applied via with_num_ctx() before the first request; the client
    // stays frozen afterward (num_ctx is never injected per-request).
    // Embedded backend skips the Ollama probe: ctx is owned by InferPolicy
    // (KITTY_INFER_CTX, default 4096).
    let ctx_resolution = if matches!(
        cli.command,
        Commands::Run { .. } | Commands::Bench { .. } | Commands::BenchAb { .. }
    ) {
        match ollama_client.as_ref() {
            Some(oc) => {
                let res = resolve_session_num_ctx(oc, &cli.model).await;
                info!(
                    num_ctx = res.num_ctx,
                    source = res.source,
                    "context window resolved"
                );
                Some(res)
            }
            None => {
                info!("context window owned by kitty-infer InferPolicy (KITTY_INFER_CTX)");
                None
            }
        }
    } else {
        None
    };
    // Boxed trait object: Ollama and embedded clients both implement
    // ChatClient, so Run/Info stay backend-agnostic from here on.
    let client: Box<dyn ChatClient> = match (&ollama_client, &ctx_resolution) {
        _ if use_infer => {
            #[cfg(feature = "kitty-infer")]
            {
                Box::new(kitty_runtime::KittyInferClient::from_env()?)
            }
            #[cfg(not(feature = "kitty-infer"))]
            {
                return Err(anyhow::anyhow!(kitty_runtime::unavailable_error()));
            }
        }
        (Some(oc), Some(res)) => Box::new(oc.clone().with_num_ctx(res.num_ctx)),
        (Some(oc), None) => Box::new(oc.clone()),
        (None, _) => unreachable!("ollama backend always builds its client"),
    };
    // Bench suites run over the backend-selected BenchClient (EMBEDDED-BENCH-SPEC
    // §diseño mínimo-1): Ollama default; infer opt-in por env, nunca default
    // silencioso. La sonda num_ctx solo corre en el brazo Ollama (resuelta
    // arriba); infer usa KITTY_INFER_CTX.
    let bench_client = || -> anyhow::Result<bench::BenchClient> {
        if use_infer {
            #[cfg(feature = "kitty-infer")]
            {
                Ok(bench::BenchClient::Infer(
                    kitty_runtime::KittyInferClient::from_env()?,
                ))
            }
            #[cfg(not(feature = "kitty-infer"))]
            {
                Err(anyhow::anyhow!(kitty_runtime::unavailable_error()))
            }
        } else {
            Ok(bench::BenchClient::Ollama(
                ollama_client
                    .clone()
                    .expect("ollama backend always builds its client"),
            ))
        }
    };
    let bench_backend: &'static str = if use_infer { "kitty-infer" } else { "ollama" };

    match cli.command {
        Commands::Run { prompt, no_memory } => {
            let prompt = prompt.join(" ");
            // F2 KV preflight, routed through the T2.2 resolver: the warning
            // must report the RESOLVED window (env/native/fallback), not a
            // raw env read — warn (never fail) when a large window risks OOM
            // on 8GB GPUs without acknowledged KV quantization.
            if let Some(res) = &ctx_resolution {
                if let Some(warning) = kitty_runtime::client::kv_cache_preflight_warning(
                    res.num_ctx,
                    std::env::var(kitty_runtime::client::ENV_KV_QUANT_ACK)
                        .ok()
                        .as_deref(),
                ) {
                    warn!("{}", warning);
                }
            }
            let memory_db_path = if no_memory {
                None
            } else {
                Some(default_db_path())
            };
            let base_config = AgentConfig {
                model: cli.model.clone(),
                // Gate-4: explicit project root for the Bash sandbox allow-root
                // (and bootstrap detection) — never the implicit process cwd.
                workspace: resolve_project_root(),
                enable_knowledge: cli.knowledge,
                knowledge_config: KnowledgeConfig {
                    enabled: cli.knowledge,
                    knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
                    ..Default::default()
                },
                memory_db_path,
                ..Default::default()
            }
            .with_profile(&cli.profile);
            let mut agent = Agent::new(client, base_config);
            // Per-run overrides (Fase 2.1): explicit CLI flags survive the
            // construction-time profile sync via one auditable override path.
            let mut overrides = RunOverrides::default();
            // f8 T1.1: interactive sessions may grant autorun continuation
            // hops via env (bench paths NEVER read this — they are flag-only).
            if let Ok(raw) = std::env::var(kitty_harness::config::ENV_AUTORUN_HOPS) {
                match raw.trim().parse::<usize>() {
                    Ok(v) => overrides.autorun_hops = Some(v),
                    Err(_) => warn!("KITTY_AUTORUN_HOPS={raw:?} is not a valid usize — ignoring"),
                }
            }
            // Overlay explicit CLI overrides (checked against clap defaults).
            if cli.context_window != 16384 {
                // f8 AD5: reduced window ⇒ re-derive the effective compaction
                // threshold against the profile value (and resize the budget,
                // which the pre-P3 overlay forgot) — done inside apply_overrides.
                overrides.context_window = Some(cli.context_window);
            }
            if cli.max_iterations != 20 {
                overrides.max_iterations = Some(cli.max_iterations);
            }
            if cli.thinking_budget != 2048 {
                let mut tuning = agent.config().tuning.clone();
                tuning.thinking = ThinkingMode::Budget(cli.thinking_budget as u32);
                overrides.tuning = Some(tuning);
            }
            if cli.bash_timeout != 60 {
                overrides.bash_timeout = Some(cli.bash_timeout);
            }
            if cli.semantic_memory {
                // G4-1: gate on the RESOLVED profile, not the raw flag string
                // (the "auto" default never equals "tiny" verbatim).
                let resolved = ModelProfile::from_cli(&cli.profile, &cli.model);
                if matches!(resolved.name, "tiny" | "nano" | "small") {
                    warn!(
                        "--semantic-memory requires medium/large profile, ignoring for {}",
                        resolved.name
                    );
                } else {
                    warn!("--semantic-memory enabled (hybrid BM25+embedding RRF)");
                }
            }
            if cli.cloud_fallback {
                if std::env::var(kitty_runtime::cloud::ENV_API_KEY).is_ok() {
                    warn!("--cloud-fallback enabled");
                } else {
                    warn!("--cloud-fallback enabled but KITTY_CLOUD_API_KEY not set, fallback will warn and be skipped");
                }
            }

            println!("Model: {}", cli.model);
            println!("Prompt: {}", prompt);
            println!();

            let start = Instant::now();
            let response = agent.run_with(&prompt, overrides).await?;
            let elapsed = start.elapsed();

            println!("{}", response);
            println!();
            println!("Completed in {:.1?}", elapsed);
            Ok(())
        }
        Commands::Bench {
            suite,
            limit,
            hops,
            timeout_secs,
            out,
            exercises: exercises_path,
            pre_registration,
            no_memory,
            claweval_trials,
            claweval_parallel,
            claweval_sandbox,
        } => {
            let cli_ctx = bench::BenchCliCtx {
                url: cli.url.clone(),
                model: cli.model.clone(),
                backend: bench_backend,
                thinking_budget: cli.thinking_budget,
                knowledge: cli.knowledge,
                knowledge_dir: cli.knowledge_dir.clone(),
                context_window: cli.context_window,
                max_iterations: cli.max_iterations,
                bash_timeout: cli.bash_timeout,
                profile: cli.profile.clone(),
            };
            bench::bench_command(
                cli_ctx,
                bench_client()?,
                bench::BenchArgs {
                    suite,
                    limit,
                    hops,
                    timeout_secs,
                    out,
                    exercises: exercises_path,
                    pre_registration,
                    no_memory,
                    claweval_trials,
                    claweval_parallel,
                    claweval_sandbox,
                },
            )
            .await
        }
        Commands::Info => {
            println!("Kitty Code v{}", env!("CARGO_PKG_VERSION"));
            println!("Local coding agent for ornith:9B on consumer GPUs");
            println!();
            if use_infer {
                println!("Backend: kitty-infer (embedded GGUF)");
                println!(
                    "Model: {}",
                    std::env::var(kitty_runtime::ENV_MODEL)
                        .unwrap_or_else(|_| "(KITTY_INFER_MODEL unset)".into())
                );
            } else {
                println!("Model: {}", cli.model);
                println!("URL:   {}", cli.url);
            }

            match client.health_check().await {
                Ok(()) => println!("Status: Connected"),
                Err(e) => println!("Status: Error: {}", e),
            }

            println!();
            let definitions = standard_tool_definitions();
            let tool_names: Vec<&str> = definitions
                .iter()
                .map(|d| d.function.name.as_str())
                .collect();
            println!("Tools: {}", tool_names.join(", "));
            println!("Edit:  Hashline (primary) + search-replace (fallback)");
            println!("Bench: polyglot (code), tb2 (terminal), tb2-official (terminal-bench-2.1 via Harbor), gpqa (reasoning, gated, requires --pre-registration and --limit 198), claweval (agent eval, requires claw-eval CLI + OPENROUTER_API_KEY)");
            Ok(())
        }
        Commands::Mcp => mcp::run().await,
        Commands::BenchAb {
            thinking,
            limit,
            hops,
            timeout_secs,
            reps,
            suite,
            out,
            interleave,
        } => {
            let cli_ctx = bench::BenchCliCtx {
                url: cli.url.clone(),
                model: cli.model.clone(),
                backend: bench_backend,
                thinking_budget: cli.thinking_budget,
                knowledge: cli.knowledge,
                knowledge_dir: cli.knowledge_dir.clone(),
                context_window: cli.context_window,
                max_iterations: cli.max_iterations,
                bash_timeout: cli.bash_timeout,
                profile: cli.profile.clone(),
            };
            bench::bench_ab_command(
                cli_ctx,
                bench_client()?,
                bench::BenchAbArgs {
                    thinking,
                    limit,
                    hops,
                    timeout_secs,
                    reps,
                    suite,
                    out,
                    interleave,
                },
            )
            .await
        }
        #[cfg(feature = "tui")]
        Commands::Tui => {
            let context_window = AgentConfig::default().context_window;
            tui::run_tui(&cli.model, &cli.url, context_window)?;
            Ok(())
        }
    }
}
#[cfg(test)]
mod num_ctx_resolution_tests {
    use super::{resolve_effective_num_ctx, NumCtxResolution};
    use kitty_runtime::client::DEFAULT_NUM_CTX;

    #[test]
    fn env_override_wins_without_probe() {
        // G3/T2.2 AC: env set → used verbatim, native value irrelevant
        // (None here models "no probe was made").
        assert_eq!(
            resolve_effective_num_ctx(Some("8192"), None),
            NumCtxResolution {
                num_ctx: 8192,
                source: "env"
            }
        );
        // Env even beats a (hypothetical) probe result.
        assert_eq!(
            resolve_effective_num_ctx(Some("4096"), Some(131072)),
            NumCtxResolution {
                num_ctx: 4096,
                source: "env"
            }
        );
    }

    #[test]
    fn native_probe_clamped_to_floor_and_ceiling() {
        // min(native, 32768), floored at 4096.
        assert_eq!(
            resolve_effective_num_ctx(None, Some(131072)),
            NumCtxResolution {
                num_ctx: 32768,
                source: "native"
            }
        );
        assert_eq!(
            resolve_effective_num_ctx(None, Some(2048)),
            NumCtxResolution {
                num_ctx: 4096,
                source: "native"
            }
        );
        assert_eq!(
            resolve_effective_num_ctx(None, Some(16384)),
            NumCtxResolution {
                num_ctx: 16384,
                source: "native"
            }
        );
    }

    #[test]
    fn fallback_when_env_invalid_and_probe_failed() {
        // Probe failure/network error → default window, loud fallback source.
        assert_eq!(
            resolve_effective_num_ctx(None, None),
            NumCtxResolution {
                num_ctx: DEFAULT_NUM_CTX,
                source: "fallback"
            }
        );
        // Set-but-unparsable env is treated as absent by the pure core.
        assert_eq!(
            resolve_effective_num_ctx(Some("not-a-number"), None),
            NumCtxResolution {
                num_ctx: DEFAULT_NUM_CTX,
                source: "fallback"
            }
        );
        assert_eq!(
            resolve_effective_num_ctx(Some("0"), None),
            NumCtxResolution {
                num_ctx: DEFAULT_NUM_CTX,
                source: "fallback"
            }
        );
    }
}
