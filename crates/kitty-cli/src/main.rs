//! Kitty CLI — Command-line interface for Kitty Code.

mod bench;

#[cfg(feature = "tui")]
mod tui;

mod claweval;
mod doctor;
mod gpqa;
mod journal;
mod mcp;
mod mcp_client;
mod plan;
mod polyglot;
mod skills;
mod tb2;
mod update;

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
#[command(name = "kittycode")]
#[command(about = "Local coding agent for ornith:9B on consumer GPUs")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

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
        /// W3 logprobs spike: send ONE `/api/chat` with logprobs on, print
        /// support + mean top1-top2 margin, and exit BEFORE the agent loop.
        /// Ollama-only; any other backend reports HELD (never faked).
        /// Nota probe: anotar (modelo,prompt); margen run-dependiente.
        #[arg(long)]
        logprobs_probe: bool,
        /// W3 plan mode: print what would run and exit WITHOUT entering the
        /// agent loop (zero file edits; `Edit`/`Write` stay blocked).
        #[arg(long)]
        plan: bool,
        /// Determinism seed (W7): same seed + same inputs ⇒ same retry
        /// backoffs and knowledge tie-breaks. Wired to the existing harness
        /// builder `Agent::with_seed` (signature untouched).
        /// `None` (default) = legacy exact behavior.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Run benchmarks
    Bench {
        /// Benchmark suite: polyglot (official dataset polyglot-benchmark, unit-test grading),
        /// tb2 (HOMEMADE 8 ejercicios locales, no oficial), tb2-official (OFFICIAL terminal-bench-2.1 via Harbor CLI),
        /// demo-12 (suite fija v1.0.0 con justicia embebida: 12 ejercicios, pineos, k>=3 por serie, gates_off; ignora --exercises/--limit),
        /// ppc (Harness² offline k=1, --limit ≤3, scored/offline, exit 0/1/2/3),
        /// gpqa, gpqa_diamond, claweval
        /// (gpqa requires --pre-registration <PATH>;
        ///  claweval requires OPENROUTER_API_KEY + claw-eval CLI;
        ///  tb2-official requires harbor CLI + Docker;
        ///  ppc requires --limit ≤3, default 3 cuando --limit es el global 15)
        #[arg(default_value = "polyglot")]
        suite: String,
        /// Number of exercises to run (ppc: ≤3, default 3 si se deja el global 15)
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
    /// Check for a newer release (fail-closed without network)
    Update {
        /// Print machine-readable JSON instead of human text
        #[arg(long)]
        json: bool,
    },
    /// Check the local environment (Ollama, model, VRAM, disk) — W1 doctor
    Doctor {
        /// Print the report as JSON instead of human text
        #[arg(long)]
        json: bool,
    },
    /// List discovered skills (SKILL.md walker) as prompt XML — W3 skills
    Skills {
        /// Extra skill roots (repeatable; default: KITTY_SKILLS_DIR, ./skills,
        /// ~/.config/opencode/skills)
        #[arg(long)]
        dir: Vec<PathBuf>,
    },
    /// Revert the most recent journaled edit — W3 undo
    Undo {
        /// Journal file (default ~/.kitty/journal.jsonl or KITTY_JOURNAL)
        #[arg(long)]
        journal: Option<PathBuf>,
    },
    /// List tools from an external MCP server spawned as a child — W3 client
    #[command(name = "mcp-client")]
    McpClient {
        /// Server program to spawn (e.g. `npx`)
        command: String,
        /// Call this tool after listing (default: only list)
        #[arg(long)]
        call: Option<String>,
        /// JSON object arguments for --call (default: {})
        #[arg(long, default_value = "{}")]
        args_json: String,
        /// Arguments for the server program
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Self-test probe MCP server over stdio (hidden; used by tests)
    #[command(hide = true, name = "probe-mcp-server")]
    ProbeMcpServer,
    /// Start MCP server over stdio (exposes Read/Edit/Write/Bash/Glob/Grep)
    Mcp {
        /// Plan mode: refuse Edit/Write over the MCP transport too
        #[arg(long)]
        plan: bool,
    },
    /// Launch the interactive TUI (default feature `tui`; same as bare `kittycode`)
    #[cfg(feature = "tui")]
    Tui,
}

/// T1.4 (SDD-82-10 Q2): coexistencia tokio-console + fmt en un solo dispatcher.
///
/// Con `RUSTFLAGS="--cfg tokio_unstable"` (SOLO sesion, nunca commiteado) el
/// registry compone `fmt + console` (gRPC 127.0.0.1:6669); sin la flag es un
/// `fmt` plano identico al anterior — delta no-op (mismo filtro
/// `kitty=info,warn`; `try_init` en vez de `init` para no panicar en doble
/// init). `init()` a secas con dos dispatchers panica: por eso una sola via.
#[cfg(tokio_unstable)]
fn init_tracing() {
    use tracing_subscriber::prelude::*;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "kitty=info,warn".into());
    let console_layer = console_subscriber::ConsoleLayer::builder()
        .with_default_env()
        .spawn();
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(filter))
        .with(console_layer)
        .try_init();
}

/// Via sin flag: comportamiento identico al `fmt().init()` previo.
#[cfg(not(tokio_unstable))]
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kitty=info,warn".into()),
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

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
    // W3 plan mode (`run --plan`) is an offline dry-run: it prints and exits
    // before any client/probe work, so it never touches the network.
    let plan_dry_run = matches!(&cli.command, Some(Commands::Run { plan: true, .. }));
    let ctx_resolution = if !plan_dry_run
        && matches!(
            cli.command,
            Some(Commands::Run { .. })
                | Some(Commands::Bench { .. })
                | Some(Commands::BenchAb { .. })
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
        // SDD §0: bare `kittycode` sin subcomando abre la TUI con la ventana
        // de contexto del CLI (sin red para abrir: el worker solo conecta al
        // enviar el primer turno o pedir /api/tags desde el picker).
        None => {
            #[cfg(feature = "tui")]
            {
                tui::run_tui(&cli.model, &cli.url, cli.context_window)?;
                Ok(())
            }
            #[cfg(not(feature = "tui"))]
            {
                Err(anyhow::anyhow!(
                    "no subcommand given: rebuild with `--features tui` for the interactive TUI (or run `kittycode --help`)"
                ))
            }
        }
        Some(Commands::Run {
            prompt,
            no_memory,
            logprobs_probe,
            plan,
            seed,
        }) => {
            let prompt = prompt.join(" ");
            // W3 plan mode: dry-run summary, NEVER the agent loop.
            if plan {
                println!("plan mode (--plan): no edits will be applied.");
                println!("Model: {}", cli.model);
                println!("URL:   {}", cli.url);
                println!("Prompt: {}", prompt);
                println!("Memory: {}", if no_memory { "off" } else { "on" });
                match seed {
                    Some(s) => println!("Seed: {}", s),
                    None => println!("Seed: none (legacy)"),
                }
                println!("Tools available: Read, Edit, Write, Bash, Glob, Grep");
                println!("Blocked in plan mode: Edit, Write");
                return Ok(());
            }
            // W3 spike probe (minimal CLI surface): one logprobs round-trip,
            // no agent loop, no defaults changed.
            if logprobs_probe {
                match ollama_client.as_ref() {
                    Some(oc) => {
                        let outcome = oc
                            .chat_with_logprobs(
                                &cli.model,
                                vec![kitty_runtime::Message::user(if prompt.is_empty() {
                                    "hi"
                                } else {
                                    &prompt
                                })],
                                None,
                                None,
                                5,
                            )
                            .await?;
                        match outcome.logprobs.as_deref() {
                            Some(lps) => println!(
                                "logprobs: SUPPORTED n_tokens={} mean_margin_top1_top2={:.3}",
                                lps.len(),
                                kitty_runtime::mean_margin(lps).unwrap_or(f32::NAN)
                            ),
                            None => println!("logprobs: NOT-RETURNED (server ignored the request)"),
                        }
                    }
                    None => println!(
                        "logprobs: HELD-backend (kitty-infer/LLM lane: not live-verified here)"
                    ),
                }
                return Ok(());
            }
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
            // C3 (Gate B): arm the agent-loop edit journal (`kitty undo`).
            // The hook lives in harness `ToolRegistry::execute`; this lane
            // only ensures `KITTY_JOURNAL` resolves (explicit value wins,
            // else the default) so the hook has a path to write to.
            // Format untouched: append-only JSONL, one entry per edit.
            let journal_path = journal::ensure_run_journal_env();
            info!(
                journal = %journal_path.display(),
                "edit journal armed for run"
            );
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
            // W1: Agent::new es sync por diseño (mkdir + SQLite + knowledge load
            // una vez por agente). Offload al blocking pool para no parkar el
            // reactor; constructores públicos siguen sync.
            let mut agent = tokio::task::spawn_blocking(move || Agent::new(client, base_config))
                .await
                .map_err(|e| anyhow::anyhow!("agent construction task failed: {e}"))?;
            // C2 (Gate B): determinism seed via the existing harness builder
            // (`Agent::with_seed`; signature untouched). `None` keeps the
            // legacy exact behavior.
            if let Some(s) = seed {
                agent = agent.with_seed(s);
            }
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
        Some(Commands::Bench {
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
        }) => {
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
        Some(Commands::Info) => {
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
            println!("Bench: polyglot (code), tb2 (terminal), tb2-official (terminal-bench-2.1 via Harbor), demo-12 (fija v1.0.0 k>=3), ppc (Harness² offline k=1 --limit ≤3), gpqa (reasoning, gated, requires --pre-registration and --limit 198), claweval (agent eval, requires claw-eval CLI + OPENROUTER_API_KEY)");
            Ok(())
        }
        Some(Commands::Update { json }) => update::run_update(json).await,
        Some(Commands::Mcp { plan }) => mcp::run(plan).await,
        Some(Commands::Doctor { json }) => {
            // W1: probe Ollama (/api/tags + /api/show via OllamaClient) + host.
            let report = doctor::run_doctor(&cli.url, &cli.model).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", doctor::format_human(&report));
            }
            Ok(())
        }
        Some(Commands::Skills { dir }) => {
            // W3: discover SKILL.md files, render the prompt XML block.
            let roots = if dir.is_empty() {
                skills::default_skill_dirs()
            } else {
                dir
            };
            let found = skills::discover_skills(&roots);
            println!("skills: {} found", found.len());
            for s in &found {
                println!(
                    "- {} — {} [{}]",
                    s.name,
                    s.description,
                    s.location.display()
                );
            }
            print!("{}", skills::format_skills_xml(&found));
            Ok(())
        }
        Some(Commands::Undo { journal }) => {
            // W3: revert the most recent journaled edit.
            let path = journal.unwrap_or_else(journal::default_journal_path);
            match journal::undo_last(&path)? {
                Some(reverted) => {
                    println!(
                        "undone: {} (journal {})",
                        reverted.display(),
                        path.display()
                    );
                }
                None => println!("journal empty: {} — nothing to undo", path.display()),
            }
            Ok(())
        }
        Some(Commands::McpClient {
            command,
            call,
            args_json,
            args,
        }) => {
            // W3: list tools from an external MCP server child process.
            let names = mcp_client::list_tools_via_child(&command, &args).await?;
            println!("tools: {} found", names.len());
            for n in &names {
                println!("- {n}");
            }
            if let Some(tool) = call {
                let arguments: serde_json::Value = serde_json::from_str(&args_json)?;
                let result =
                    mcp_client::call_tool_via_child(&command, &args, &tool, arguments).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Ok(())
        }
        Some(Commands::ProbeMcpServer) => mcp_client::serve_probe_stdio().await,
        Some(Commands::BenchAb {
            thinking,
            limit,
            hops,
            timeout_secs,
            reps,
            suite,
            out,
            interleave,
        }) => {
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
        Some(Commands::Tui) => {
            tui::run_tui(&cli.model, &cli.url, cli.context_window)?;
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

#[cfg(test)]
mod run_flags_tests {
    use super::*;
    use clap::Parser;
    use kitty_runtime::{ChatResponse, Message, MockClient, ToolCall, ToolCallFunction};

    #[test]
    fn run_seed_flag_parses_and_defaults_to_none() {
        // C2: `--seed` is a `Run`-only flag (bench keeps its seed-42 shuffle).
        let cli = Cli::try_parse_from(["kittycode", "run", "hello"]).unwrap();
        match cli.command {
            Some(Commands::Run { seed, plan, .. }) => {
                assert_eq!(seed, None);
                assert!(!plan);
            }
            _ => panic!("expected run"),
        }
        let cli = Cli::try_parse_from(["kittycode", "run", "--seed", "42", "hello"]).unwrap();
        match cli.command {
            Some(Commands::Run { seed, .. }) => assert_eq!(seed, Some(42)),
            _ => panic!("expected run"),
        }
    }

    #[test]
    fn run_plan_and_seed_combine() {
        let cli = Cli::try_parse_from(["kittycode", "run", "--plan", "--seed", "7", "hi"]).unwrap();
        match cli.command {
            Some(Commands::Run { plan, seed, .. }) => {
                assert!(plan);
                assert_eq!(seed, Some(7));
            }
            _ => panic!("expected run"),
        }
    }

    #[test]
    fn bare_no_subcommand_parses_to_none_with_cli_context_window() {
        // SDD §0: bare `kittycode` abre la TUI (command == None) con la
        // ventana del CLI, no AgentConfig::default().
        let cli = Cli::try_parse_from(["kittycode"]).unwrap();
        assert!(cli.command.is_none(), "bare must parse to None");
        assert_eq!(cli.context_window, 16384);
        let cli = Cli::try_parse_from(["kittycode", "--context-window", "8192"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.context_window, 8192);
    }

    // === B3: `kittycode bench ppc` parsea + default --limit ≤3 ===

    #[test]
    fn bench_ppc_invocacion_parsea_y_default_limit_3() {
        // B3: `kittycode bench ppc` debe parsear (suite ppc en help) y el
        // default global 15 no debe romper el cap PPC (≤3): bench/mod.rs lo
        // recorta a 3 cuando limit==15.
        let cli = Cli::try_parse_from(["kittycode", "bench", "ppc"]).expect("parse bench ppc");
        match cli.command {
            Some(Commands::Bench { suite, limit, .. }) => {
                assert_eq!(suite, "ppc", "suite debe ser ppc");
                assert_eq!(limit, 15, "default clap global sigue 15 (el recorte a 3 lo hace bench/mod.rs)");
                // El recorte efectivo: 15 (default) ⇒ 3 (cap PPC).
                let effective = if suite == "ppc" && limit == 15 {
                    crate::bench::ppc::PPC_MAX_LIMIT
                } else {
                    limit
                };
                assert_eq!(effective, 3, "default ppc efectivo ≤3");
                assert!(crate::bench::ppc::validate_limit(effective, 15).is_ok());
            }
            _ => panic!("expected Bench ppc"),
        }
        // Explícito --limit 2 pasa; --limit 4 rechaza (cap Oracle).
        let cli =
            Cli::try_parse_from(["kittycode", "bench", "ppc", "--limit", "2"]).expect("parse ppc limit 2");
        match cli.command {
            Some(Commands::Bench { suite, limit, .. }) => {
                assert_eq!((suite.as_str(), limit), ("ppc", 2));
            }
            _ => panic!("expected Bench ppc limit 2"),
        }
    }

    fn tool_resp(name: &str, args: serde_json::Value) -> ChatResponse {
        ChatResponse {
            model: "test-model".into(),
            message: Message::tool_calls(vec![ToolCall {
                id: None,
                function: ToolCallFunction {
                    name: name.to_string(),
                    arguments: args,
                },
            }]),
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }

    fn text_resp(text: &str) -> ChatResponse {
        ChatResponse {
            model: "test-model".into(),
            message: Message::assistant(text),
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }

    /// C3 end-to-end (offline, MockClient): the exact builder chain the `run`
    /// path uses (`Agent::new` + `with_seed`, journal armed via
    /// `ensure_run_journal_env`) records an agent-loop `Edit` in the journal,
    /// and `undo_last` — the same call the `kitty undo` command makes —
    /// reverts it. No journal-format change: `read_entries` parses what the
    /// harness hook wrote.
    // NB: the journal-env lock intentionally spans the `run_with` await
    // (the env must stay put for the whole agent loop); test-only.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn run_path_journals_agent_loop_edit_for_undo() {
        // `KITTY_JOURNAL` is the production wiring (Run path → harness
        // hook). No other test in this binary swaps it without holding the
        // shared lock; still restore on drop.
        let _lock = crate::journal::test_journal_env_lock();
        let prev = std::env::var(crate::journal::ENV_JOURNAL).ok();
        struct Restore(Option<String>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match &self.0 {
                    Some(v) => std::env::set_var(crate::journal::ENV_JOURNAL, v),
                    None => std::env::remove_var(crate::journal::ENV_JOURNAL),
                }
            }
        }
        let _restore = Restore(prev);

        let work = tempfile::tempdir().unwrap();
        let target = work.path().join("note.txt");
        std::fs::write(&target, "before").unwrap();
        let jdir = tempfile::tempdir().unwrap();
        let journal = jdir.path().join("journal.jsonl");

        // Same arming the `run` path performs (explicit value wins).
        std::env::set_var(crate::journal::ENV_JOURNAL, &journal);
        assert_eq!(crate::journal::ensure_run_journal_env(), journal);

        let path_str = target.to_str().unwrap().to_string();
        // F2a E1: el texto es terminal, así que el tercer turno es tool
        // (agota) y el forced aporta el terminal.
        let mock = MockClient::new(vec![
            tool_resp("Read", serde_json::json!({"path": path_str})),
            tool_resp(
                "Edit",
                serde_json::json!({
                    "path": path_str,
                    "old_string": "before",
                    "new_string": "AFTER",
                }),
            ),
            tool_resp("Glob", serde_json::json!({"pattern": "*.md"})),
            text_resp("final answer"),
        ]);
        let mut agent = Agent::new(
            mock,
            AgentConfig {
                workspace: work.path().to_path_buf(),
                ..Default::default()
            },
        )
        .with_seed(42);
        // Bound the loop through the public override path (same one the
        // `run` command uses for `--max-iterations`).
        let overrides = RunOverrides {
            max_iterations: Some(3),
            ..Default::default()
        };
        let out = agent.run_with("apply the edit", overrides).await.unwrap();
        assert_eq!(out, "final answer");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "AFTER");

        let entries = crate::journal::read_entries(&journal).unwrap();
        assert_eq!(entries.len(), 1, "got {entries:?}");
        assert_eq!(entries[0].before.as_deref(), Some("before"));

        let undone = crate::journal::undo_last(&journal).unwrap().expect("entry");
        assert_eq!(undone, target);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "before");
    }
}
