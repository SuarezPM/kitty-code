//! Kitty CLI — Command-line interface for Kitty Code.

#[cfg(feature = "tui")]
mod tui;

mod gpqa;
mod mcp;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kitty_harness::agent::{clamp_autorun_hops, Agent, AgentConfig};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::tools::standard_tool_definitions;
use kitty_harness::ContextBudget;
use kitty_harness::{ModelProfile, ThinkingMode};
use kitty_memory::{Episode, EpisodeStore};
use kitty_runtime::OllamaClient;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// Default path for episodic memory database.
fn default_db_path() -> String {
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

    /// Model profile preset: nano|tiny|medium|large (default: auto —
    /// auto-detects nano-tier sub-3B models by id shape)
    #[arg(long, global = true, default_value = "auto", value_parser = clap::builder::PossibleValuesParser::new(["nano", "tiny", "medium", "large", "auto"]))]
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
        /// Benchmark suite: polyglot, tb2, smoke, gpqa, gpqa_diamond (gpqa requires --pre-registration <PATH> with 40-hex revision + 64-hex sha256, n=198 and --limit 198)
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
        /// KITTY_BENCH_TIMEOUT_SECS env fallback; both absent ⇒ 30.
        #[arg(long, default_value = "30")]
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
        /// KITTY_BENCH_TIMEOUT_SECS env fallback; both absent ⇒ 30.
        #[arg(long, default_value = "30")]
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
    let client = OllamaClient::new(&cli.url);

    // f8 T2.2: resolve the session context window ONCE per process, BEFORE
    // any Agent/client construction, shared by Run/Bench/BenchAb. The value
    // is applied via with_num_ctx() before the first request; the client
    // stays frozen afterward (num_ctx is never injected per-request).
    let ctx_resolution = if matches!(
        cli.command,
        Commands::Run { .. } | Commands::Bench { .. } | Commands::BenchAb { .. }
    ) {
        let res = resolve_session_num_ctx(&client, &cli.model).await;
        info!(
            num_ctx = res.num_ctx,
            source = res.source,
            "context window resolved"
        );
        Some(res)
    } else {
        None
    };
    let client = match &ctx_resolution {
        Some(res) => client.with_num_ctx(res.num_ctx),
        None => client,
    };

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
            // f8 T1.1: interactive sessions may grant autorun continuation
            // hops via env (bench paths NEVER read this — they are flag-only).
            if let Ok(raw) = std::env::var("KITTY_AUTORUN_HOPS") {
                match raw.trim().parse::<usize>() {
                    Ok(v) => agent.config_mut().autorun_hops = clamp_autorun_hops(v),
                    Err(_) => warn!("KITTY_AUTORUN_HOPS={raw:?} is not a valid usize — ignoring"),
                }
            }
            // Overlay explicit CLI overrides after profile sync (profile wins by default,
            // but explicit CLI flags should survive). We check against clap defaults.
            if cli.context_window != 16384 {
                agent.config_mut().context_window = cli.context_window;
                // f8 AD5: reduced window ⇒ re-derive the effective compaction
                // threshold against the profile value (and resize the budget,
                // which the pre-P3 overlay forgot).
                let profile_threshold = agent.config().compact_threshold;
                let budget = agent.budget_mut();
                budget.set_max(cli.context_window);
                budget.set_threshold(ContextBudget::derive_compaction_threshold(
                    profile_threshold,
                    cli.context_window,
                ));
            }
            if cli.max_iterations != 20 {
                agent.config_mut().max_iterations = cli.max_iterations;
            }
            if cli.thinking_budget != 2048 {
                // G2-FIX3: tuning.thinking is the LIVE thinking axis — the
                // agent re-seeds its budget tracker from it at every run().
                agent.config_mut().tuning.thinking =
                    ThinkingMode::Budget(cli.thinking_budget as u32);
            }
            if cli.bash_timeout != 60 {
                agent.config_mut().bash_timeout_secs = cli.bash_timeout;
            }
            if cli.semantic_memory {
                // G4-1: gate on the RESOLVED profile, not the raw flag string
                // (the "auto" default never equals "tiny" verbatim).
                let resolved = ModelProfile::from_cli(&cli.profile, &cli.model);
                if resolved.name == "tiny" || resolved.name == "nano" {
                    warn!(
                        "--semantic-memory requires medium/large profile, ignoring for {}",
                        resolved.name
                    );
                } else {
                    warn!("--semantic-memory enabled (hybrid BM25+embedding RRF)");
                }
            }
            if cli.cloud_fallback {
                if std::env::var("KITTY_CLOUD_API_KEY").is_ok() {
                    warn!("--cloud-fallback enabled");
                } else {
                    warn!("--cloud-fallback enabled but KITTY_CLOUD_API_KEY not set, fallback will warn and be skipped");
                }
            }

            println!("Model: {}", cli.model);
            println!("Prompt: {}", prompt);
            println!();

            let start = Instant::now();
            let response = agent.run(&prompt).await?;
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
        } => {
            // Pre-bench CWD contamination guard (Gate 2 hardening):
            // leftover solution.py in CWD contaminates grading (agent may write to CWD
            // instead of tmpdir). Warn and remove stale files if present.
            for stale in ["./solution.py", "./solution.js", "./solution.sh"] {
                if std::path::Path::new(stale).exists() {
                    warn!("{stale} in CWD will contaminate bench tmpdir; removing stale file");
                    let _ = std::fs::remove_file(stale);
                }
            }

            // Ensure memory DB directory exists
            if !no_memory {
                let db_path = default_db_path();
                if let Some(db_dir) = std::path::Path::new(&db_path).parent() {
                    std::fs::create_dir_all(db_dir).ok();
                }
            }

            // Gate 2 tuning: bench uses max_iterations=5 for sub-20s smoke (default 20 remains for `run`).
            // If CLI left default 20, bench overrides to 5; explicit CLI value is respected.
            let bench_max_iterations = if cli.max_iterations == 20 {
                5
            } else {
                cli.max_iterations
            };
            let is_small_model = is_small_model(&cli.profile, &cli.model);
            let effective_no_memory = no_memory || is_small_model;
            if is_small_model && !no_memory {
                warn!(model = %cli.model, "small model detected — auto-disabling episodic memory for bench (use explicit --no-memory to silence)");
            }
            // f9 calibration: per-exercise timeout — flag > env > default.
            let (exercise_timeout_secs, env_invalid) = resolve_timeout_from_flag(timeout_secs);
            if env_invalid {
                warn!(
                    "per-exercise timeout falling back to {DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS}s"
                );
            }

            let cfg = BenchConfig {
                client: client.clone(),
                model: cli.model.clone(),
                limit,
                exercise_timeout_secs,
                use_memory: !effective_no_memory,
                thinking_budget: cli.thinking_budget,
                context_window: cli.context_window,
                max_iterations: bench_max_iterations,
                autorun_hops: kitty_harness::agent::clamp_autorun_hops(hops),
                bash_timeout: cli.bash_timeout,
                profile: ModelProfile::from_cli(&cli.profile, &cli.model),
                enable_knowledge: cli.knowledge,
                knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
            };

            // --- GPQA Diamond local approximation (F1: offline-first, gated, pre-registro explícito G1) ---
            if suite == "gpqa" || suite == "gpqa_diamond" {
                return run_gpqa_suite(&cfg, &out, limit, pre_registration).await;
            }

            println!("Kitty Code Benchmark");
            println!("  Model: {}", cli.model);
            println!("  Suite: {} (limit: {})", suite, limit);
            println!(
                "  Memory: {}",
                if effective_no_memory { "off" } else { "on" }
            );
            println!(
                "  Knowledge: {}",
                if cli.knowledge {
                    format!("on ({})", cli.knowledge_dir)
                } else {
                    "off (use --knowledge to enable, 5 cards)".to_string()
                }
            );
            if bench_max_iterations != cli.max_iterations {
                println!(
                    "  Max iterations: {} (bench override, CLI default 20)",
                    bench_max_iterations
                );
            } else {
                println!("  Max iterations: {}", bench_max_iterations);
            }
            println!("  Thinking budget: {}", cli.thinking_budget);
            println!("  Context window: {}", cli.context_window);
            if let Some(ref path) = exercises_path {
                println!("  Exercises JSON: {}", path.display());
            }
            println!();

            // If --exercises provided, load from JSON; otherwise use built-in
            let results = if let Some(ref path) = exercises_path {
                let json_exercises = load_json_exercises(path)?;
                let bench_items: Vec<BenchItem> = json_exercises
                    .iter()
                    .filter_map(json_to_bench_item)
                    .collect();
                if bench_items.is_empty() {
                    anyhow::bail!("No valid exercises found in {}", path.display());
                }
                println!("  Loaded {} exercises from JSON", bench_items.len());
                run_bench(&bench_items, &cfg).await?
            } else {
                match suite.as_str() {
                    "tb2" => run_bench(&tb2_exercises(), &cfg).await?,
                    _ => run_bench(&exercises(), &cfg).await?,
                }
            };

            let passed = results.iter().filter(|r| r.passed).count();
            let total = results.len();
            let rate = if total > 0 {
                100.0 * passed as f64 / total as f64
            } else {
                0.0
            };

            println!();
            println!("RESULTS");
            println!("Passed: {}/{} ({:.1}%)", passed, total, rate);

            // Per-language breakdown: language parsed from polyglot task-name
            // prefixes (polyglot_python_* → python), label fallback otherwise.
            let by_lang = tally_by_language(&results);
            for (lang, t) in &by_lang {
                let lang_rate = if t.attempted > 0 {
                    100.0 * t.passed as f64 / t.attempted as f64
                } else {
                    0.0
                };
                println!(
                    "  {lang}: attempted={} passed={} failed={} errored={} ({:.1}%)",
                    t.attempted, t.passed, t.failed, t.errored, lang_rate
                );
            }

            let gate = 50.0;
            if rate >= gate {
                println!();
                println!("GATE PASSED: {:.1}% >= {:.0}%", rate, gate);
            } else {
                println!();
                println!("GATE FAILED: {:.1}% < {:.0}%", rate, gate);
            }

            let output = serde_json::json!({
                "model": cli.model,
                "suite": suite,
                "total": total,
                "passed": passed,
                "rate_pct": rate,
                "memory": cfg.use_memory,
                "by_language": by_lang.iter().map(|(lang, t)| (lang.clone(), serde_json::json!({
                    "attempted": t.attempted,
                    "passed": t.passed,
                    "failed": t.failed,
                    "errored": t.errored,
                }))).collect::<std::collections::BTreeMap<String, serde_json::Value>>(),
                "details": results.iter().map(|r| serde_json::json!({
                    "id": r.id,
                    "lang": r.lang,
                    "passed": r.passed,
                    "verdict": r.verdict,
                    "latency_ms": r.latency_ms,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
            println!("Results saved to {}", out.display());
            // bench-reproducibility: exit-code 0=pass/improved, 1=gate failed, 2=noise (no significant change)
            let baseline_rate: Option<f64> = std::fs::read_to_string(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benches/baseline.json"),
            )
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
            .and_then(|v| v.get("rate_pct").and_then(|r| r.as_f64()));
            let exit_code = bench_exit_code(rate, gate, baseline_rate);
            std::process::exit(exit_code);
        }
        Commands::Info => {
            println!("Kitty Code v{}", env!("CARGO_PKG_VERSION"));
            println!("Local coding agent for ornith:9B on consumer GPUs");
            println!();
            println!("Model: {}", cli.model);
            println!("URL:   {}", cli.url);

            match client.health_check().await {
                Ok(true) => println!("Status: Connected"),
                Ok(false) => println!("Status: Not connected"),
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
            println!("Bench: polyglot (code), tb2 (terminal), gpqa (reasoning, gated, requires --pre-registration and --limit 198)");
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
            let budgets = parse_thinking_budgets(&thinking)?;
            let plan = build_ab_plan(limit, &budgets, reps, interleave);

            // Same tuning as `bench`: 5 iterations for sub-20s smoke unless CLI overrides.
            let bench_max_iterations = if cli.max_iterations == 20 {
                5
            } else {
                cli.max_iterations
            };
            let is_small_model = is_small_model(&cli.profile, &cli.model);
            let effective_no_memory = is_small_model;
            if is_small_model {
                warn!(model = %cli.model, "small model detected — auto-disabling episodic memory for bench-ab");
            }
            // f9 calibration: per-exercise timeout — flag > env > default.
            let (exercise_timeout_secs, env_invalid) = resolve_timeout_from_flag(timeout_secs);
            if env_invalid {
                warn!(
                    "per-exercise timeout falling back to {DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS}s"
                );
            }

            let base_cfg = BenchConfig {
                client: client.clone(),
                model: cli.model.clone(),
                limit,
                exercise_timeout_secs,
                use_memory: !effective_no_memory,
                thinking_budget: 2048, // overridden per run below
                context_window: cli.context_window,
                max_iterations: bench_max_iterations,
                autorun_hops: kitty_harness::agent::clamp_autorun_hops(hops),
                bash_timeout: cli.bash_timeout,
                profile: ModelProfile::from_cli(&cli.profile, &cli.model),
                enable_knowledge: cli.knowledge,
                knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
            };

            let items: Vec<BenchItem> = match suite.as_str() {
                "tb2" => tb2_exercises(),
                _ => exercises(),
            };

            let store = if base_cfg.use_memory {
                Some(EpisodeStore::open(&default_db_path())?)
            } else {
                None
            };

            println!("Kitty Code A/B Benchmark");
            println!("  Model: {}", cli.model);
            println!("  Suite: {} (limit: {}, reps: {})", suite, limit, reps);
            println!("  Budgets: {:?}", budgets);
            println!("  Interleave: {}", interleave);
            println!("  Runs: {}", plan.len());
            println!();

            let mut per_budget: std::collections::HashMap<usize, Vec<BenchResult>> =
                std::collections::HashMap::new();
            for (i, (task_idx, budget)) in plan.iter().enumerate() {
                let ex = &items[*task_idx % items.len()];
                let mut cfg = base_cfg.clone();
                cfg.thinking_budget = *budget;
                print!(
                    "[{:2}/{:2}] budget={:4} task={:2} ",
                    i + 1,
                    plan.len(),
                    budget,
                    task_idx
                );
                let r = run_one_bench(ex, &cfg, store.as_ref()).await?;
                per_budget.entry(*budget).or_default().push(r);
            }

            let mut budgets_sorted: Vec<usize> = per_budget.keys().copied().collect();
            budgets_sorted.sort_unstable();

            let mut per_budget_out = Vec::new();
            for budget in &budgets_sorted {
                per_budget_out.push(budget_summary(*budget, &per_budget[budget]));
            }

            let mut pairwise = Vec::new();
            for i in 0..budgets_sorted.len() {
                for j in (i + 1)..budgets_sorted.len() {
                    let a = &per_budget_out[i];
                    let b = &per_budget_out[j];
                    let a_rate = a["rate"].as_f64().unwrap_or(0.0);
                    let b_rate = b["rate"].as_f64().unwrap_or(0.0);
                    let winner = if a_rate >= b_rate {
                        a["budget"].clone()
                    } else {
                        b["budget"].clone()
                    };
                    pairwise.push(serde_json::json!({
                        "a": a["budget"],
                        "b": b["budget"],
                        "a_rate": a_rate,
                        "b_rate": b_rate,
                        "winner": winner,
                    }));
                }
            }

            println!();
            println!("RESULTS");
            for entry in &per_budget_out {
                let b = entry["budget"].as_u64().unwrap_or(0);
                println!(
                    "  budget={:4}  {}/{} ({:.1}%)  wilson95=[{:.1}%, {:.1}%]  p50={:.0}ms",
                    b,
                    entry["passed"].as_u64().unwrap_or(0),
                    entry["total"].as_u64().unwrap_or(0),
                    entry["rate"].as_f64().unwrap_or(0.0),
                    entry["wilson_ci"][0].as_f64().unwrap_or(0.0) * 100.0,
                    entry["wilson_ci"][1].as_f64().unwrap_or(0.0) * 100.0,
                    entry["p50_ms"].as_f64().unwrap_or(0.0),
                );
            }

            let output = serde_json::json!({
                "model": cli.model,
                "suite": suite,
                "limit": limit,
                "reps": reps,
                "interleave": interleave,
                "budgets": budgets_sorted,
                "per_budget": per_budget_out,
                "pairwise": pairwise,
            });
            std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
            println!("Results saved to {}", out.display());
            Ok(())
        }
        #[cfg(feature = "tui")]
        Commands::Tui => {
            let context_window = AgentConfig::default().context_window;
            tui::run_tui(&cli.model, &cli.url, context_window)?;
            Ok(())
        }
    }
}

// ── Benchmark ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
struct BenchResult {
    id: String,
    lang: String,
    passed: bool,
    /// Outcome verdict: "pass" | "fail" | "error" | "timeout".
    verdict: String,
    latency_ms: f64,
}

/// Language for a bench result: parse polyglot task-name prefixes
/// (`polyglot_python_list-ops__koxBfT3` → `python`, `polyglot_go_pig-latin__…`
/// → `go`), falling back to the exercise language label when the id does not
/// carry a `polyglot_<lang>_` prefix.
fn language_of(id: &str, lang_label: &str) -> String {
    if let Some(rest) = id.strip_prefix("polyglot_") {
        if let Some(idx) = rest.find('_') {
            let candidate = &rest[..idx];
            if !candidate.is_empty() {
                return candidate.to_string();
            }
        }
    }
    lang_label.to_string()
}

/// Per-language outcome tally for the result summary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LangTally {
    attempted: usize,
    passed: usize,
    failed: usize,
    errored: usize,
}

/// Aggregate results by language (parsed from task names, label fallback).
/// `errored` counts error/timeout outcomes; `failed` counts completed runs
/// that did not pass.
fn tally_by_language(results: &[BenchResult]) -> std::collections::BTreeMap<String, LangTally> {
    let mut tally: std::collections::BTreeMap<String, LangTally> =
        std::collections::BTreeMap::new();
    for r in results {
        let lang = language_of(&r.id, &r.lang);
        let e = tally.entry(lang).or_default();
        e.attempted += 1;
        if r.passed {
            e.passed += 1;
        } else if r.verdict == "error" || r.verdict == "timeout" {
            e.errored += 1;
        } else {
            e.failed += 1;
        }
    }
    tally
}

/// A benchmark exercise loaded from JSON.
#[derive(Debug, Clone, serde::Deserialize)]
struct JsonExercise {
    id: String,
    #[serde(default)]
    lang: String,
    prompt: String,
    /// String to search for in the check_file (polyglot format).
    #[serde(default)]
    check_str: String,
    /// File to check for the check_str.
    #[serde(default)]
    check_file: String,
    /// Check type for TB2 tasks: "file_exists", "file_content_exact", etc.
    #[serde(default)]
    check_type: String,
    /// Relative path for the check (TB2 format).
    #[serde(default)]
    check_path: String,
    /// Expected content for file_content_exact checks.
    #[serde(default)]
    expected: String,
    /// Setup category for TB2 tasks: "file_ops", "text_proc", etc.
    #[serde(default)]
    setup: String,
}

/// Convert a JsonExercise to BenchItem (for polyglot + TB2 JSON formats).
fn json_to_bench_item(ex: &JsonExercise) -> Option<BenchItem> {
    if !ex.check_str.is_empty() && !ex.check_file.is_empty() {
        let check_str = ex.check_str.clone();
        let check_file = ex.check_file.clone();
        let cf = check_file.clone();
        let cf_leaked = Box::leak(check_file.into_boxed_str());
        Some(BenchItem {
            id: Box::leak(ex.id.clone().into_boxed_str()),
            label: Box::leak(ex.lang.clone().into_boxed_str()),
            prompt: Box::leak(ex.prompt.clone().into_boxed_str()),
            check: Box::new(move |d| file_contains(d, &cf, &check_str)),
            check_file: cf_leaked,
        })
    } else if !ex.check_type.is_empty() {
        let check_path = ex.check_path.clone();
        let check_type = ex.check_type.clone();
        let expected = ex.expected.clone();
        let cf = check_path.clone();
        let check_file_leaked: &'static str = if check_path.is_empty() {
            "solution.py"
        } else {
            Box::leak(check_path.into_boxed_str())
        };
        let check: Box<dyn Fn(&str) -> bool> = match check_type.as_str() {
            "file_exists" => {
                let p = cf.clone();
                Box::new(move |d| std::path::Path::new(d).join(&p).exists())
            }
            "file_not_exists" => {
                let p = cf.clone();
                Box::new(move |d| !std::path::Path::new(d).join(&p).exists())
            }
            "file_executable" => {
                let p = cf.clone();
                Box::new(move |d| file_is_executable(d, &p))
            }
            "file_content_exact" => {
                let p = cf.clone();
                let exp = expected.clone();
                Box::new(move |d| {
                    let path = std::path::Path::new(d).join(&p);
                    match std::fs::read_to_string(&path) {
                        Ok(c) => c.trim() == exp.trim(),
                        Err(_) => false,
                    }
                })
            }
            "always_pass" => Box::new(|_| true),
            _ => {
                let p = cf.clone();
                Box::new(move |d| std::path::Path::new(d).join(&p).exists())
            }
        };
        let label_str = if !ex.setup.is_empty() {
            ex.setup.clone()
        } else if !ex.lang.is_empty() {
            ex.lang.clone()
        } else {
            "tb2".to_string()
        };
        Some(BenchItem {
            id: Box::leak(ex.id.clone().into_boxed_str()),
            label: Box::leak(label_str.into_boxed_str()),
            prompt: Box::leak(ex.prompt.clone().into_boxed_str()),
            check,
            check_file: check_file_leaked,
        })
    } else {
        None
    }
}

/// Load exercises from a JSON file.
fn load_json_exercises(path: &std::path::Path) -> Result<Vec<JsonExercise>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read exercises file: {}", path.display()))?;
    let exercises: Vec<JsonExercise> = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse exercises JSON: {}", path.display()))?;
    Ok(exercises)
}

struct BenchItem {
    id: &'static str,
    /// Grouping label shown in results (language or category).
    label: &'static str,
    prompt: &'static str,
    check: Box<dyn Fn(&str) -> bool>,
    check_file: &'static str,
}

/// Helper to build a BenchItem with a boxed closure.
fn bench(
    id: &'static str,
    label: &'static str,
    prompt: &'static str,
    check: impl Fn(&str) -> bool + 'static,
) -> BenchItem {
    BenchItem {
        id,
        label,
        prompt,
        check: Box::new(check),
        check_file: "solution.py",
    }
}

fn bench_file(
    id: &'static str,
    label: &'static str,
    prompt: &'static str,
    check_file: &'static str,
    check: impl Fn(&str) -> bool + 'static,
) -> BenchItem {
    BenchItem {
        id,
        label,
        prompt,
        check: Box::new(check),
        check_file,
    }
}

fn exercises() -> Vec<BenchItem> {
    vec![
        bench("py-fib", "python", "Write a function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers. Save to solution.py.", |d| file_contains(d, "solution.py", "def fibonacci")),
        bench("py-class", "python", "Create a BankAccount class with deposit() and withdraw() methods. Save to solution.py.", |d| file_contains(d, "solution.py", "class BankAccount")),
        bench("py-sort", "python", "Write merge_sort(arr: list[int]) -> list[int]. Save to solution.py.", |d| file_contains(d, "solution.py", "def merge_sort")),
        bench("py-decorator", "python", "Write a retry decorator that retries up to 3 times on exception. Save to solution.py.", |d| file_contains(d, "solution.py", "def retry")),
        bench("py-context", "python", "Write a context manager that times code execution. Save to solution.py.", |d| file_contains(d, "solution.py", "__enter__")),
        bench_file("js-promise", "javascript", "Write fetchWithTimeout(url, timeout) using AbortController. Save to solution.js.", "solution.js", |d| file_contains(d, "solution.js", "AbortController")),
        bench_file("js-debounce", "javascript", "Write a debounce function. Save to solution.js.", "solution.js", |d| file_contains(d, "solution.js", "debounce")),
        bench_file("js-async", "javascript", "Write an async function to read a file. Save to solution.js.", "solution.js", |d| file_contains(d, "solution.js", "async")),
        bench_file("ts-interface", "typescript", "Write a TypeScript interface User with id, name, email. Save to solution.ts.", "solution.ts", |d| file_contains(d, "solution.ts", "interface User")),
        bench_file("ts-generic", "typescript", "Write generic first<T>(arr: T[]): T | undefined. Save to solution.ts.", "solution.ts", |d| file_contains(d, "solution.ts", "function first")),
        bench_file("sh-backup", "bash", "Write a bash script that backs up a directory to timestamped tar.gz. Save to solution.sh.", "solution.sh", |d| file_contains(d, "solution.sh", "tar")),
        bench_file("sh-replace", "bash", "Write a bash script to find and replace text in .txt files. Save to solution.sh.", "solution.sh", |d| file_contains(d, "solution.sh", "find")),
        bench_file("rust-struct", "rust", "Write a Rust struct Point with x,y f64 and new() + distance_to(). Save to solution.rs.", "solution.rs", |d| file_contains(d, "solution.rs", "struct Point")),
        bench_file("rust-enum", "rust", "Write enum Shape with Circle/Rectangle/Triangle and area(). Save to solution.rs.", "solution.rs", |d| file_contains(d, "solution.rs", "enum Shape")),
        bench_file("go-struct", "go", "Write Go struct Person with Name/Age and String() method. Save to solution.go.", "solution.go", |d| file_contains(d, "solution.go", "type Person struct")),
    ]
}

fn file_contains(dir: &str, filename: &str, pattern: &str) -> bool {
    let path = std::path::Path::new(dir).join(filename);
    match std::fs::read_to_string(&path) {
        Ok(content) => content.contains(pattern),
        Err(_) => false,
    }
}

/// Shared benchmark configuration.
#[derive(Clone)]
struct BenchConfig {
    client: OllamaClient,
    model: String,
    limit: usize,
    use_memory: bool,
    thinking_budget: usize,
    context_window: usize,
    max_iterations: usize,
    /// Autorun continuation hops (f8 T1.3): FLAG-ONLY on bench paths — the
    /// harness never reads `KITTY_AUTORUN_HOPS`, so ambient env cannot leak
    /// into benchmark runs. Default 0.
    autorun_hops: usize,
    /// Per-exercise wall-clock cap in seconds (f9 calibration: resolved via
    /// flag > KITTY_BENCH_TIMEOUT_SECS > default 30).
    exercise_timeout_secs: u64,
    bash_timeout: u64,
    profile: &'static ModelProfile,
    enable_knowledge: bool,
    knowledge_dir: PathBuf,
}

/// Shared benchmark driver used by every exercise set.
async fn run_bench(items: &[BenchItem], cfg: &BenchConfig) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();

    let store = if cfg.use_memory {
        Some(EpisodeStore::open(&default_db_path())?)
    } else {
        None
    };

    for (i, ex) in items.iter().take(cfg.limit).enumerate() {
        print!("[{:2}/{:2}] ", i + 1, cfg.limit.min(items.len()));
        results.push(run_one_bench(ex, cfg, store.as_ref()).await?);
    }

    Ok(results)
}

/// Run a single benchmark exercise with the given config.
/// Shared by `run_bench` and the A/B thinking-budget benchmark.
/// G3-FIX1: re-seed an agent's budget compaction threshold from its
/// EFFECTIVE post-overlay context window (`agent.config().context_window`),
/// never from a raw CLI argument. AD5 derivation semantics apply: the result
/// never raises above `profile_threshold` and equals it on windows where the
/// ratio already exceeds the profile value.
fn sync_budget_threshold(agent: &mut Agent, profile_threshold: f32) {
    let effective_window = agent.config().context_window;
    agent
        .budget_mut()
        .set_threshold(ContextBudget::derive_compaction_threshold(
            profile_threshold,
            effective_window,
        ));
}

/// Default per-exercise bench wall-clock cap (seconds). Calibration showed
/// real 9B-model tasks on split-GPU hardware routinely exceed this; raise
/// via `--timeout-secs` or `KITTY_BENCH_TIMEOUT_SECS`.
pub const DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS: u64 = 30;

/// Env fallback for the per-exercise bench timeout (f9 calibration).
pub const ENV_BENCH_TIMEOUT_SECS: &str = "KITTY_BENCH_TIMEOUT_SECS";

/// Pure resolver for the per-exercise timeout — precedence: explicit CLI
/// flag > `KITTY_BENCH_TIMEOUT_SECS` env > default 30.
///
/// Returns `(resolved_secs, env_invalid)` so the caller can warn exactly
/// once when the env var was present but unparsable (or non-positive): such
/// values fall back to the default rather than producing a zero timeout.
#[must_use]
fn resolve_exercise_timeout_secs(explicit_flag: Option<u64>, env_raw: Option<&str>) -> (u64, bool) {
    if let Some(secs) = explicit_flag {
        return (secs, false);
    }
    match env_raw.map(str::trim) {
        Some(raw) => match raw.parse::<u64>() {
            Ok(secs) if secs > 0 => (secs, false),
            _ => (DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS, true),
        },
        None => (DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS, false),
    }
}

/// Resolve the effective timeout for a bench command from the parsed flag:
/// the flag counts as EXPLICIT only when it differs from the clap default
/// (same sentinel pattern as the other CLI overlays in this binary).
fn resolve_timeout_from_flag(flag_secs: u64) -> (u64, bool) {
    let explicit = (flag_secs != DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS).then_some(flag_secs);
    let env_raw = std::env::var(ENV_BENCH_TIMEOUT_SECS).ok();
    let (resolved, invalid) = resolve_exercise_timeout_secs(explicit, env_raw.as_deref());
    if invalid {
        warn!(
            raw = ?env_raw,
            default = DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS,
            "{ENV_BENCH_TIMEOUT_SECS} is not a positive integer — using default"
        );
    }
    (resolved, invalid)
}

async fn run_one_bench(
    ex: &BenchItem,
    cfg: &BenchConfig,
    store: Option<&EpisodeStore>,
) -> Result<BenchResult> {
    print!("{:20}...", ex.id);

    // Recall similar past episodes
    if let Some(store) = store {
        let past = store.recall_similar(ex.prompt, 3)?;
        if !past.is_empty() {
            print!(" ({} past)", past.len());
        }
    }

    let tmpdir = tempfile::tempdir().context("Failed to create temp dir")?;
    let tmpdir_str = tmpdir.path().to_string_lossy().to_string();

    // Prompt hardening: use real check_file per exercise (fix .py forced for js/ts/rust/go)
    // Gate 2 absolute path prevents CWD contamination.
    let target_file = if ex.check_file.is_empty() {
        "solution.py"
    } else {
        ex.check_file
    };
    let prompt = format!(
        "{}\n\nWork in directory: {} — Save to {}/{} (use absolute path)",
        ex.prompt, tmpdir_str, tmpdir_str, target_file
    );

    let start = Instant::now();
    // Bench determinism gate: exercises are single-file throwaway workspaces
    // graded via `ex.check`; if one gained a Cargo.toml/pyproject.toml
    // mid-task, the agent verification loop would run workspace tests plus
    // revision cycles against the 30s exercise cap. Bench never wants that.
    std::env::set_var(kitty_harness::agent::ENV_VERIFY_LOOP, "off");

    let base_config = AgentConfig {
        model: cfg.model.clone(),
        profile: cfg.profile,
        enable_knowledge: cfg.enable_knowledge,
        knowledge_config: KnowledgeConfig {
            enabled: cfg.enable_knowledge,
            knowledge_dir: cfg.knowledge_dir.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut agent = Agent::new(cfg.client.clone(), base_config);
    // Bench smoke override: bench uses 5 iterations even for medium (20) to stay <20s.
    // Agent::new syncs profile fields, so patch explicit bench overrides here.
    // Copy Run overlay logic to preserve profile tiny/medium/large unless CLI explicitly differs.
    agent.config_mut().max_iterations = cfg.max_iterations;
    // f8 T1.3: bench autorun is FLAG-ONLY — set explicitly (default 0) so an
    // ambient KITTY_AUTORUN_HOPS can never leak into a benchmark run.
    agent.config_mut().autorun_hops = clamp_autorun_hops(cfg.autorun_hops);
    if cfg.context_window != 16384 {
        agent.config_mut().context_window = cfg.context_window;
        agent.budget_mut().set_max(cfg.context_window);
    }
    if cfg.thinking_budget != 2048 {
        // G2-FIX3: tuning.thinking is the LIVE thinking axis — the agent
        // re-seeds its budget tracker from it at every run(). This is what
        // makes bench-ab A/B arms actually differ.
        agent.config_mut().tuning.thinking = ThinkingMode::Budget(cfg.thinking_budget as u32);
    }
    if cfg.bash_timeout != 60 {
        agent.config_mut().bash_timeout_secs = cfg.bash_timeout;
    }
    // Ensure compact threshold follows profile (tiny 0.7, medium 0.8, large 0.8)
    agent.config_mut().compact_threshold = cfg.profile.compact_threshold;
    // G3-FIX1: re-sync derives from the agent's EFFECTIVE post-overlay
    // context window (profile preset unless --context-window overrode it) —
    // NOT the raw CLI argument, which silently undid AD5's small-window
    // protection on tiny-profile bench runs.
    sync_budget_threshold(&mut agent, cfg.profile.compact_threshold);
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(cfg.exercise_timeout_secs),
        agent.run(&prompt),
    )
    .await;

    // Determine pass/fail and persist an episode for memory.
    let (passed, verdict, latency_ms) = match outcome {
        Ok(Ok(_)) => {
            let passed = (ex.check)(&tmpdir_str);
            (
                passed,
                if passed { "pass" } else { "fail" },
                start.elapsed().as_millis() as f64,
            )
        }
        Ok(Err(e)) => {
            println!(" error: {}", e);
            (false, "error", start.elapsed().as_millis() as f64)
        }
        Err(_) => {
            println!(" timeout");
            (false, "timeout", 30000.0)
        }
    };

    let status = if passed { "PASS" } else { "FAIL" };
    println!("{} ({:.0}ms)", status, latency_ms);

    if let Some(store) = store {
        let files_touched = if verdict == "pass" || verdict == "fail" {
            list_files(&tmpdir_str)
        } else {
            vec![]
        };
        let ep = Episode::new(
            ex.prompt.to_string(),
            files_touched,
            Some(verdict.into()),
            format!("{verdict} (latency {latency_ms:.0}ms)"),
        );
        store.save(&ep).ok();
    }

    Ok(BenchResult {
        id: ex.id.to_string(),
        lang: ex.label.to_string(),
        passed,
        verdict: verdict.to_string(),
        latency_ms,
    })
}

// ── GPQA Diamond local approximation (F1 + G1) ─────────────────

/// GPQA Diamond local suite: offline-first, gated, pre-registro explícito (G1).
/// Carga solo materializado validado contra pre-registro; no hace red, no registra prompts, no usa HF_TOKEN.
/// Resultado etiquetado como aproximación local no oficial con proveniencia completa.
/// Si falta/inválido el pre-registro ⇒ salida `blocked` y nunca invoca el modelo.
async fn run_gpqa_suite(
    cfg: &BenchConfig,
    out: &PathBuf,
    limit: usize,
    pre_registration: Option<PathBuf>,
) -> Result<()> {
    println!("Kitty Code GPQA Diamond (local approximation)");
    println!("  Model: {}", cfg.model);

    // Preflight con pre-registro explícito (G1): bloqueado sin invocar modelo si falta/inválido
    let (pre, manifest) = match &pre_registration {
        None => {
            let help = crate::gpqa::pre_registration_help_text();
            eprintln!("\nGPQA preflight FAILED (offline-first, gated):\nPre-registration required but not provided.\n{help}\n");
            eprintln!("Nota: GPQA Diamond es gated; este harness nunca descarga automáticamente ni usa HF_TOKEN.");
            // Escribir salida blocked para trazabilidad
            let blocked = serde_json::json!({
                "suite": "gpqa_diamond_local",
                "status": "blocked",
                "reason": "missing --pre-registration",
                "help": help,
                "label": "aproximación local no oficial — no leaderboard oficial",
                "model": cfg.model,
                "limit": limit,
            });
            let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
            println!("Results (blocked) saved to {}", out.display());
            anyhow::bail!("GPQA blocked: missing --pre-registration");
        }
        Some(p) => match crate::gpqa::preflight_gpqa_with_prereg(p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\nGPQA preflight FAILED (offline-first, gated):\n{e}\n");
                eprintln!("Nota: GPQA Diamond es gated; este harness nunca descarga automáticamente ni usa HF_TOKEN.");
                eprintln!("El resultado GPQA siempre se etiqueta como \"aproximación local no oficial\" (no leaderboard).");
                let blocked = serde_json::json!({
                    "suite": "gpqa_diamond_local",
                    "status": "blocked",
                    "reason": e.to_string(),
                    "label": "aproximación local no oficial — no leaderboard oficial",
                    "model": cfg.model,
                    "limit": limit,
                    "pre_registration": p.display().to_string(),
                });
                let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
                println!("Results (blocked) saved to {}", out.display());
                anyhow::bail!("GPQA preflight blocked: {e}");
            }
        },
    };

    let cache_path = crate::gpqa::gpqa_cache_path();

    // G1.1: GPQA solo puede ejecutar las 198 preguntas; bloquea antes de crear Agent si limit !=198
    if limit != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: --limit must be {} for GPQA Diamond (got {}). Usa --limit 198. Pre-registro n/limits.limit también debe ser {}.",
            crate::gpqa::GPQA_NUM_QUESTIONS, limit, crate::gpqa::GPQA_NUM_QUESTIONS
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "help": "Ejecuta: kitty bench --suite gpqa --pre-registration <PATH> --limit 198",
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
            "pre_registration": pre_registration.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        });
        let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!(msg);
    }
    if pre.limits.limit != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: pre-registration limits.limit must be {} (got {}). Corrige el pre-registro y usa --limit 198.",
            crate::gpqa::GPQA_NUM_QUESTIONS, pre.limits.limit
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
        });
        let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
        anyhow::bail!(msg);
    }

    // G1.2: pre-registro debe ser vinculante: model y límites relevantes deben coincidir con BenchConfig efectivos
    let mut binding_errors: Vec<String> = Vec::new();
    if pre.model != cfg.model {
        binding_errors.push(format!("model mismatch: pre-registration '{}' != bench '{}'", pre.model, cfg.model));
    }
    if let Some(v) = pre.limits.timeout_secs {
        if v != cfg.exercise_timeout_secs {
            binding_errors.push(format!("timeout_secs mismatch: pre-registration {} != bench {}", v, cfg.exercise_timeout_secs));
        }
    }
    if let Some(v) = pre.limits.max_iterations {
        if v != cfg.max_iterations {
            binding_errors.push(format!("max_iterations mismatch: pre-registration {} != bench {}", v, cfg.max_iterations));
        }
    }
    if let Some(v) = pre.limits.thinking_budget {
        if v != cfg.thinking_budget {
            binding_errors.push(format!("thinking_budget mismatch: pre-registration {} != bench {}", v, cfg.thinking_budget));
        }
    }
    if let Some(v) = pre.limits.context_window {
        if v != cfg.context_window {
            binding_errors.push(format!("context_window mismatch: pre-registration {} != bench {}", v, cfg.context_window));
        }
    }
    if !binding_errors.is_empty() {
        let msg = format!("GPQA blocked: pre-registration no vinculante: {}", binding_errors.join("; "));
        eprintln!("\n{msg}\nPre-registro debe fijar model y límites efectivos de la ejecución. Corrige el JSON o los flags CLI.\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
            "pre_registration": pre_registration.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        });
        let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!(msg);
    }

    let questions = crate::gpqa::load_questions(&cache_path).map_err(|e| {
        anyhow::anyhow!("Failed to load GPQA questions from {}: {e}", cache_path.display())
    })?;

    let total_available = questions.len();
    // load_questions ya garantiza 198, pero mantenemos chequeo defensivo
    if total_available != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: cache must contain exactly {} questions, found {}. Usa --limit 198 y materializa la revisión correcta.",
            crate::gpqa::GPQA_NUM_QUESTIONS, total_available
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
        });
        let _ = std::fs::write(out, serde_json::to_string_pretty(&blocked).unwrap_or_default());
        anyhow::bail!(msg);
    }

    let take = crate::gpqa::GPQA_NUM_QUESTIONS;
    println!(
        "  Dataset: {} rev {} ({} questions, limit {})",
        manifest.dataset_id,
        &manifest.revision[..8.min(manifest.revision.len())],
        total_available,
        take
    );
    println!("  Grading: {} | Permutation: {} | Template: {}", manifest.grading_method, manifest.permutation, manifest.template);
    println!("  Cache: {} (outside repo, sha256 {}...)", cache_path.display(), &manifest.sha256[..8.min(manifest.sha256.len())]);
    println!("  Label: {}", "aproximación local no oficial — no leaderboard oficial");
    println!();

    let mut results: Vec<BenchResult> = Vec::with_capacity(take);
    let mut correct = 0usize;

    let store = if cfg.use_memory {
        Some(EpisodeStore::open(&default_db_path())?)
    } else {
        None
    };

    for (idx, q) in questions.iter().take(take).enumerate() {
        // Permutación determinista de 4 opciones (correcta + 3 distractores) con revisión validada
        let distractors = q.distractors.clone();
        let (permuted, correct_idx) = crate::gpqa::permute_four(&q.id, &pre.revision, q.correct_answer.clone(), distractors)
            .map_err(|e| anyhow::anyhow!("permute failed for {}: {e}", q.id))?;
        let correct_letter = crate::gpqa::idx_to_letter(correct_idx)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompt = crate::gpqa::format_prompt(&q.question, &permuted)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        print!("[{:3}/{:3}] {} ...", idx + 1, take, q.id);

        let start = Instant::now();
        let outcome = {
            let base_config = AgentConfig {
                model: cfg.model.clone(),
                profile: cfg.profile,
                enable_knowledge: cfg.enable_knowledge,
                knowledge_config: KnowledgeConfig {
                    enabled: cfg.enable_knowledge,
                    knowledge_dir: cfg.knowledge_dir.clone(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut agent = Agent::new(cfg.client.clone(), base_config);
            agent.config_mut().max_iterations = cfg.max_iterations;
            agent.config_mut().autorun_hops = clamp_autorun_hops(cfg.autorun_hops);
            if cfg.context_window != 16384 {
                agent.config_mut().context_window = cfg.context_window;
                agent.budget_mut().set_max(cfg.context_window);
            }
            if cfg.thinking_budget != 2048 {
                agent.config_mut().tuning.thinking = ThinkingMode::Budget(cfg.thinking_budget as u32);
            }
            if cfg.bash_timeout != 60 {
                agent.config_mut().bash_timeout_secs = cfg.bash_timeout;
            }
            agent.config_mut().compact_threshold = cfg.profile.compact_threshold;
            sync_budget_threshold(&mut agent, cfg.profile.compact_threshold);
            std::env::set_var(kitty_harness::agent::ENV_VERIFY_LOOP, "off");
            tokio::time::timeout(
                std::time::Duration::from_secs(cfg.exercise_timeout_secs),
                agent.run(&prompt),
            )
            .await
        };

        let (passed, verdict, latency_ms, normalized) = match outcome {
            Ok(Ok(response)) => {
                let norm = crate::gpqa::normalize_answer(&response);
                let ok = crate::gpqa::grade_answer(norm, correct_letter);
                // No logueamos el prompt ni la respuesta cruda más allá de la letra normalizada
                (ok, if ok { "pass" } else { "fail" }, start.elapsed().as_millis() as f64, norm)
            }
            Ok(Err(e)) => {
                println!(" error: {}", e);
                (false, "error", start.elapsed().as_millis() as f64, None)
            }
            Err(_) => {
                println!(" timeout");
                (false, "timeout", 30000.0, None)
            }
        };

        if passed {
            correct += 1;
        }
        let status = if passed { "PASS" } else { "FAIL" };
        // Solo mostramos letra normalizada, nunca el prompt
        match normalized {
            Some(l) => println!("{} ({:.0}ms) -> {}", status, latency_ms, l),
            None => println!("{} ({:.0}ms) -> ?", status, latency_ms),
        }

        // Episode sin prompt (no registrar prompts GPQA)
        if let Some(store) = &store {
            let ep = Episode::new(
                format!("gpqa:{}", q.id),
                vec![],
                Some(verdict.to_string()),
                format!("gpqa {} {} (latency {:.0}ms)", q.id, verdict, latency_ms),
            );
            store.save(&ep).ok();
        }

        results.push(BenchResult {
            id: q.id.clone(),
            lang: "gpqa".to_string(),
            passed,
            verdict: verdict.to_string(),
            latency_ms,
        });
    }

    let rate = if take > 0 { 100.0 * correct as f64 / take as f64 } else { 0.0 };
    println!();
    println!("RESULTS (GPQA Diamond local approximation)");
    println!("Passed: {}/{} ({:.1}%)", correct, take, rate);
    println!("Label: aproximación local no oficial — no leaderboard oficial");
    println!("Provenance: dataset={} rev={} sha256={}... grading={} permutation={}",
        manifest.dataset_id, &manifest.revision[..8.min(manifest.revision.len())], &manifest.sha256[..8.min(manifest.sha256.len())], manifest.grading_method, manifest.permutation);

    let pre_path = pre_registration.as_ref().expect("pre-registration validated");
    let pre_sha = crate::gpqa::sha256_file(pre_path).unwrap_or_else(|_| "unknown".to_string());
    let gpqa_result = crate::gpqa::GpqaResult::new_with_prereg(&manifest, &pre, pre_path, &pre_sha, take, correct);
    let output = serde_json::json!({
        "model": cfg.model,
        "suite": gpqa_result.suite,
        "version": gpqa_result.version,
        "dataset_id": gpqa_result.dataset_id,
        "revision": gpqa_result.revision,
        "sha256": gpqa_result.sha256,
        "license": gpqa_result.license,
        "attribution": gpqa_result.attribution,
        "grading_method": gpqa_result.grading_method,
        "compatibility": gpqa_result.compatibility,
        "permutation": gpqa_result.permutation,
        "template": gpqa_result.template,
        "label": gpqa_result.label,
        "preconditions": gpqa_result.preconditions,
        "pre_registration_path": gpqa_result.pre_registration_path,
        "pre_registration_sha256": gpqa_result.pre_registration_sha256,
        "limits": gpqa_result.limits,
        "expected_artifacts": gpqa_result.expected_artifacts,
        "note": gpqa_result.note,
        "total": gpqa_result.total,
        "correct": gpqa_result.correct,
        "accuracy": gpqa_result.accuracy,
        "rate_pct": rate,
        "details": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "passed": r.passed,
            "verdict": r.verdict,
            "latency_ms": r.latency_ms,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    println!("Nota: este es un resultado local aproximado, no comparable al leaderboard oficial de GPQA.");
    Ok(())
}

// ── TB2: Terminal-Bench 2.0 ──────────────────────────────────────────

fn tb2_exercises() -> Vec<BenchItem> {
    vec![
        bench_file("tb2-bash-count", "bash", "Create a bash script called count_lines.sh that counts the number of lines in all .rs files in the current directory recursively. Save it to count_lines.sh and make it executable.", "count_lines.sh", |d| file_is_executable(d, "count_lines.sh") && file_contains(d, "count_lines.sh", "find")),
        bench_file("tb2-py-csv", "python", "Write a Python script called csv_to_json.py that reads a CSV file (input path as first argument) and outputs JSON to stdout. Save it to csv_to_json.py.", "csv_to_json.py", |d| file_contains(d, "csv_to_json.py", "csv") && file_contains(d, "csv_to_json.py", "json")),
        bench_file("tb2-makefile", "makefile", "Create a Makefile with three targets: test (runs cargo test), build (runs cargo build), and clean (removes target directory). Save to Makefile.", "Makefile", |d| file_contains(d, "Makefile", "test:") && file_contains(d, "Makefile", "build:") && file_contains(d, "Makefile", "clean:")),
        bench_file("tb2-dockerfile", "docker", "Write a Dockerfile for a Rust project that uses multi-stage builds: builder stage compiles, runtime stage copies binary. Save to Dockerfile.", "Dockerfile", |d| file_contains(d, "Dockerfile", "FROM") && file_contains(d, "Dockerfile", "COPY")),
        bench_file("tb2-gh-actions", "ci", "Create a GitHub Actions CI workflow at .github/workflows/ci.yml that runs cargo test and cargo build on push/PR. Save to .github/workflows/ci.yml.", ".github/workflows/ci.yml", |d| {
            let path = std::path::Path::new(d).join(".github/workflows/ci.yml");
            match std::fs::read_to_string(&path) {
                Ok(c) => c.contains("cargo test") || c.contains("cargo build"),
                Err(_) => false,
            }
        }),
        bench_file("tb2-sh-find-modified", "bash", "Write a bash script called find_recent.sh that finds all files modified in the last 24 hours in a given directory (first argument). Save to find_recent.sh.", "find_recent.sh", |d| file_is_executable(d, "find_recent.sh") && file_contains(d, "find_recent.sh", "find")),
        bench_file("tb2-py-dirsize", "python", "Write a Python script called dirsize.py that calculates and prints the total size of each subdirectory in the current directory. Save to dirsize.py.", "dirsize.py", |d| file_contains(d, "dirsize.py", "os") && file_contains(d, "dirsize.py", "path")),
        bench_file("tb2-sh-backup", "bash", "Write a bash script called backup.sh that backs up files from a source directory (first argument) to a destination with a timestamp suffix (e.g. backup_20260817_120000). Save to backup.sh.", "backup.sh", |d| file_is_executable(d, "backup.sh") && file_contains(d, "backup.sh", "cp")),
    ]
}

fn file_is_executable(dir: &str, filename: &str) -> bool {
    let path = std::path::Path::new(dir).join(filename);
    match std::fs::metadata(&path) {
        Ok(meta) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                meta.permissions().mode() & 0o111 != 0
            }
            #[cfg(not(unix))]
            {
                meta.is_file()
            }
        }
        Err(_) => false,
    }
}

fn list_files(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                files.push(name.to_string());
            }
        }
    }
    files
}

/// Pure helper for bench exit-code (P1b reproducibility spec).
/// 0 = pass/improved, 1 = gate failed or baseline regression, 2 = noise (|delta|<2).
fn bench_exit_code(rate: f64, gate: f64, baseline: Option<f64>) -> i32 {
    if rate < gate {
        1
    } else if let Some(b) = baseline {
        if rate < b {
            1
        } else if (rate - b).abs() < 2.0 {
            2
        } else {
            0
        }
    } else {
        0
    }
}

#[cfg(test)]
mod bench_exit_tests {
    use super::bench_exit_code;

    #[test]
    fn gate_failed_returns_1() {
        assert_eq!(bench_exit_code(40.0, 50.0, None), 1);
        assert_eq!(bench_exit_code(40.0, 50.0, Some(96.0)), 1);
    }

    #[test]
    fn baseline_regression_returns_1() {
        assert_eq!(bench_exit_code(90.0, 50.0, Some(96.44)), 1);
    }

    #[test]
    fn noise_within_2pct_returns_2() {
        assert_eq!(bench_exit_code(97.0, 50.0, Some(96.44)), 2);
        assert_eq!(bench_exit_code(96.44, 50.0, Some(96.44)), 2);
    }

    #[test]
    fn improved_returns_0() {
        assert_eq!(bench_exit_code(99.0, 50.0, Some(96.44)), 0);
    }

    #[test]
    fn no_baseline_pass_returns_0() {
        assert_eq!(bench_exit_code(80.0, 50.0, None), 0);
    }

    #[test]
    fn test_baseline_path_cargo_manifest() {
        let manifest_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../benches/baseline.json");
        assert!(
            manifest_path.exists(),
            "baseline not found via CARGO_MANIFEST_DIR: {} (exists={})",
            manifest_path.display(),
            manifest_path.exists()
        );
        let content =
            std::fs::read_to_string(&manifest_path).expect("read baseline via CARGO_MANIFEST_DIR");
        let v: serde_json::Value = serde_json::from_str(&content).expect("parse baseline");
        assert!(v.get("rate_pct").is_some(), "baseline missing rate_pct");
        let cwd = std::env::current_dir().unwrap_or_default();
        if cwd.ends_with("crates") || cwd.ends_with("kitty-cli") {
            assert!(
                !std::path::Path::new("benches/baseline.json").exists(),
                "old relative path unexpectedly exists from subdir CWD={}",
                cwd.display()
            );
        }
    }
}

fn parse_thinking_budgets(s: &str) -> anyhow::Result<Vec<usize>> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        anyhow::bail!("thinking budgets string empty");
    }
    let mut out = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let v: usize = p
            .parse()
            .with_context(|| format!("invalid thinking budget: {p}"))?;
        if v == 0 {
            anyhow::bail!("thinking budget must be > 0: {v}");
        }
        out.push(v);
    }
    if out.is_empty() {
        anyhow::bail!("no valid budgets parsed from: {s}");
    }
    Ok(out)
}

fn build_ab_plan(
    limit: usize,
    budgets: &[usize],
    reps: usize,
    interleave: bool,
) -> Vec<(usize, usize)> {
    let mut plan = Vec::new();
    let reps = reps.max(1);
    for &budget in budgets {
        for task_idx in 0..limit {
            for _ in 0..reps {
                plan.push((task_idx, budget));
            }
        }
    }
    if interleave {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        plan.shuffle(&mut rng);
    }
    plan
}

/// Aggregate one budget arm's results into its `per_budget` JSON entry
/// (G4-FIX1: the runbook's central errored-trials gate needs serialized
/// error counts, which the output previously lacked).
///
/// Verdict policy, documented: `"error"` and `"timeout"` are counted in
/// SEPARATE keys — `errored` (in-run failures) vs `timed_out` (task-level
/// harness-limit outcomes, a different failure class). The README gate
/// applies to `errored`; consumers apply their own timeout policy. All
/// pre-existing fields are preserved untouched for backward compatibility.
fn budget_summary(budget: usize, results: &[BenchResult]) -> serde_json::Value {
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let rate = if total > 0 {
        100.0 * passed as f64 / total as f64
    } else {
        0.0
    };
    let (lo, hi) = wilson_ci(passed, total);
    let p50 = p50_ms(results.iter().map(|r| r.latency_ms).collect());
    let errored = results.iter().filter(|r| r.verdict == "error").count();
    let timed_out = results.iter().filter(|r| r.verdict == "timeout").count();
    serde_json::json!({
        "budget": budget,
        "passed": passed,
        "total": total,
        "rate": rate,
        "wilson_ci": [lo, hi],
        "p50_ms": p50,
        "errored": errored,
        "timed_out": timed_out,
    })
}

fn wilson_ci(passed: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f64;
    let p = passed as f64 / n;
    let z: f64 = 1.96;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let inner = (p * (1.0 - p) + z2 / (4.0 * n)) / n;
    let delta = z * inner.sqrt();
    let lo = (centre - delta) / denom;
    let hi = (centre + delta) / denom;
    (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0))
}

fn p50_ms(mut latencies: Vec<f64>) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = latencies.len();
    if n % 2 == 1 {
        latencies[n / 2]
    } else {
        (latencies[n / 2 - 1] + latencies[n / 2]) / 2.0
    }
}

fn is_small_model(profile: &str, model: &str) -> bool {
    // G4-1: gate on the RESOLVED profile name — nano sessions (auto-detected
    // or explicit) must get the same small-model bench guard as tiny, or
    // SQLite recall stalls a 4096-token session.
    matches!(ModelProfile::from_cli(profile, model).name, "tiny" | "nano")
}

#[cfg(test)]
mod bench_ab_tests {
    use super::{
        build_ab_plan, parse_thinking_budgets, sync_budget_threshold, wilson_ci, Agent,
        AgentConfig, Cli, Commands, ModelProfile, OllamaClient,
    };
    use clap::Parser;

    #[test]
    fn test_bench_ab_parses() {
        let cli = Cli::try_parse_from([
            "kitty",
            "bench-ab",
            "--thinking",
            "1024,2048,4096",
            "--limit",
            "50",
        ])
        .expect("parse bench-ab");
        match cli.command {
            Commands::BenchAb {
                thinking, limit, ..
            } => {
                assert_eq!(limit, 50);
                let budgets = parse_thinking_budgets(&thinking).expect("parse budgets");
                assert_eq!(budgets, vec![1024, 2048, 4096]);
            }
            _ => panic!("expected BenchAb"),
        }
    }

    /// G3-FIX1 AC: the bench threshold re-sync derives from the agent's
    /// EFFECTIVE window — a tiny-profile bench agent WITHOUT an explicit
    /// --context-window keeps AD5's small-window protection (0.62 exactly).
    #[test]
    fn bench_threshold_resync_derives_from_effective_window() {
        // Tiny preset ⇒ effective window 4096; NO --context-window override.
        let mut agent = Agent::new(
            OllamaClient::new("http://localhost:11434"),
            AgentConfig::default().with_profile("tiny"),
        );
        assert_eq!(agent.config().context_window, 4096);
        sync_budget_threshold(&mut agent, ModelProfile::tiny().compact_threshold);
        assert!(
            (agent.budget().threshold() - 0.62).abs() < 1e-6,
            "tiny bench agent must keep derived 0.62, got {}",
            agent.budget().threshold()
        );

        // Control: medium at its default window stays the profile value.
        let mut agent = Agent::new(
            OllamaClient::new("http://localhost:11434"),
            AgentConfig::default().with_profile("medium"),
        );
        sync_budget_threshold(&mut agent, ModelProfile::medium().compact_threshold);
        assert!((agent.budget().threshold() - 0.8).abs() < 1e-6);

        // Explicit reduced override derives against THAT window.
        let mut agent = Agent::new(
            OllamaClient::new("http://localhost:11434"),
            AgentConfig::default().with_profile("medium"),
        );
        agent.config_mut().context_window = 4096;
        sync_budget_threshold(&mut agent, ModelProfile::medium().compact_threshold);
        assert!((agent.budget().threshold() - 0.62).abs() < 1e-6);
    }

    /// f9 calibration AC: per-exercise timeout resolution honors
    /// flag > env > default, with invalid env falling back to the default.
    #[test]
    fn resolve_exercise_timeout_flag_wins_over_env_over_default() {
        const DEFAULT: u64 = super::DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS;

        // Explicit flag wins over ANY env (valid or garbage).
        let (got, invalid) = super::resolve_exercise_timeout_secs(Some(900), Some("60"));
        assert_eq!(got, 900);
        assert!(!invalid);
        let (got, invalid) = super::resolve_exercise_timeout_secs(Some(900), Some("garbage"));
        assert_eq!(got, 900);
        assert!(!invalid);

        // No flag + valid env (trimmed) ⇒ env value.
        let (got, invalid) = super::resolve_exercise_timeout_secs(None, Some("900"));
        assert_eq!(got, 900);
        assert!(!invalid);
        let (got, invalid) = super::resolve_exercise_timeout_secs(None, Some(" 600 "));
        assert_eq!(got, 600);
        assert!(!invalid);

        // No flag + INVALID env (unparsable, empty, zero, negative) ⇒
        // default + invalid marker so the caller warns once.
        for bad in ["abc", "", "   ", "0", "-5", "12.5"] {
            let (got, invalid) = super::resolve_exercise_timeout_secs(None, Some(bad));
            assert_eq!(got, DEFAULT, "bad env {bad:?} must fall back to default");
            assert!(invalid, "bad env {bad:?} must be flagged invalid");
        }

        // Neither flag nor env ⇒ default, not invalid.
        let (got, invalid) = super::resolve_exercise_timeout_secs(None, None);
        assert_eq!(got, DEFAULT);
        assert!(!invalid);
    }

    /// f8 T1.3 AC: `--hops` propagates; absent flag ⇒ 0.
    #[test]
    fn test_bench_hops_flag_propagation() {
        // Explicit --hops 3 propagates.
        let cli = Cli::try_parse_from(["kitty", "bench-ab", "--hops", "3"])
            .expect("parse bench-ab --hops 3");
        match cli.command {
            Commands::BenchAb { hops, .. } => assert_eq!(hops, 3),
            _ => panic!("expected BenchAb"),
        }
        // Absent flag defaults to 0 (byte-identical legacy behavior).
        let cli = Cli::try_parse_from(["kitty", "bench-ab"]).expect("parse bench-ab default");
        match cli.command {
            Commands::BenchAb { hops, .. } => assert_eq!(hops, 0),
            _ => panic!("expected BenchAb"),
        }
        // Same on the Bench subcommand.
        let cli =
            Cli::try_parse_from(["kitty", "bench", "--hops", "2"]).expect("parse bench --hops 2");
        match cli.command {
            Commands::Bench { hops, .. } => assert_eq!(hops, 2),
            _ => panic!("expected Bench"),
        }
    }

    #[test]
    fn test_bench_ab_interleave() {
        let budgets = vec![1024, 2048];
        let plan = build_ab_plan(2, &budgets, 1, true);
        assert_eq!(plan.len(), 4);
        let plan_no_shuffle = build_ab_plan(2, &budgets, 1, false);
        assert_ne!(plan, plan_no_shuffle, "interleaved should be shuffled");
        let plan2 = build_ab_plan(2, &budgets, 1, true);
        assert_eq!(plan, plan2, "shuffled with seed 42 should be deterministic");
        let t4096 = wilson_ci(25, 50);
        assert!(
            t4096.0 < 0.5 && t4096.1 > 0.5,
            "wilson ci should surround p=0.5"
        );
    }

    /// G4-FIX1: mixed-verdict results must serialize `errored` and
    /// `timed_out` counts per budget (the runbook gate reads these), with
    /// all pre-existing fields preserved untouched.
    #[test]
    fn budget_summary_serializes_errored_and_timed_out_counts() {
        fn mk(id: &str, passed: bool, verdict: &str) -> super::BenchResult {
            super::BenchResult {
                id: id.into(),
                lang: "python".into(),
                passed,
                verdict: verdict.into(),
                latency_ms: 100.0,
            }
        }
        // Mixed fixture: 2 pass + 1 fail + 2 error + 1 timeout.
        let results = vec![
            mk("a", true, "pass"),
            mk("b", true, "pass"),
            mk("c", false, "fail"),
            mk("d", false, "error"),
            mk("e", false, "error"),
            mk("f", false, "timeout"),
        ];
        let entry = super::budget_summary(2048, &results);

        // New counters: error and timeout counted SEPARATELY.
        assert_eq!(entry["errored"], serde_json::json!(2));
        assert_eq!(entry["timed_out"], serde_json::json!(1));

        // Backward compatibility: existing fields untouched.
        assert_eq!(entry["budget"], serde_json::json!(2048));
        assert_eq!(entry["passed"], serde_json::json!(2));
        assert_eq!(entry["total"], serde_json::json!(6));
        assert!((entry["rate"].as_f64().unwrap() - 100.0 * 2.0 / 6.0).abs() < 1e-9);
        assert!(entry.get("wilson_ci").is_some());
        assert!(entry.get("p50_ms").is_some());

        // Empty arm: counters are zero, not absent.
        let empty = super::budget_summary(1024, &[]);
        assert_eq!(empty["errored"], serde_json::json!(0));
        assert_eq!(empty["timed_out"], serde_json::json!(0));
    }
}

#[cfg(test)]
mod profile_heuristic_tests {
    use super::is_small_model;

    #[test]
    fn test_is_small_uses_profile() {
        assert!(
            is_small_model("tiny", "qwen3:4b"),
            "qwen3:4b+tiny should be small (profile tiny)"
        );
        assert!(
            !is_small_model("medium", "ornith:9b"),
            "ornith:9b+medium should NOT be small"
        );
        assert!(
            is_small_model("tiny", "smollm2:1.7b"),
            "smollm2:1.7b+tiny should be small"
        );
        assert!(
            is_small_model("tiny", "ornith:9b"),
            "ornith:9b+tiny should be small (profile tiny) — diverges from old heuristic"
        );
    }
}

#[cfg(test)]
mod language_breakdown_tests {
    use super::{tally_by_language, BenchResult, LangTally};

    fn result(id: &str, lang: &str, passed: bool, verdict: &str) -> BenchResult {
        BenchResult {
            id: id.to_string(),
            lang: lang.to_string(),
            passed,
            verdict: verdict.to_string(),
            latency_ms: 0.0,
        }
    }

    #[test]
    fn language_parsed_from_polyglot_task_names() {
        assert_eq!(
            super::language_of("polyglot_python_list-ops__koxBfT3", "x"),
            "python"
        );
        assert_eq!(
            super::language_of("polyglot_go_pig-latin__jGEaYgv", "x"),
            "go"
        );
        assert_eq!(
            super::language_of("polyglot_rust_acronym__H3Wcvcj", "x"),
            "rust"
        );
        assert_eq!(
            super::language_of("polyglot_cpp_clock__JqkaQ4P", "x"),
            "cpp"
        );
        assert_eq!(
            super::language_of("polyglot_java_phone-number__o26caVU", "x"),
            "java"
        );
        assert_eq!(
            super::language_of("polyglot_javascript_wordy__9zXJF8V", "x"),
            "javascript"
        );
    }

    #[test]
    fn language_falls_back_to_label_without_prefix() {
        assert_eq!(super::language_of("py-fibonacci", "python"), "python");
        assert_eq!(super::language_of("tb2-makefile", "makefile"), "makefile");
        // Degenerate prefix (no language segment) → label fallback
        assert_eq!(super::language_of("polyglot_", "go"), "go");
        assert_eq!(super::language_of("polyglot_weird", "go"), "go");
    }

    #[test]
    fn tally_counts_attempted_passed_failed_errored() {
        let results = vec![
            result("polyglot_python_a__1", "python", true, "pass"),
            result("polyglot_python_b__2", "python", false, "fail"),
            result("polyglot_python_c__3", "python", false, "error"),
            result("polyglot_python_d__4", "python", false, "timeout"),
            result("polyglot_go_e__5", "go", true, "pass"),
            result("py-fib", "python", true, "pass"),
        ];
        let tally = tally_by_language(&results);
        assert_eq!(
            tally.get("python"),
            Some(&LangTally {
                attempted: 5,
                passed: 2,
                failed: 1,
                errored: 2,
            })
        );
        assert_eq!(
            tally.get("go"),
            Some(&LangTally {
                attempted: 1,
                passed: 1,
                failed: 0,
                errored: 0,
            })
        );
        assert_eq!(tally.len(), 2, "label fallback merges into parsed language");
    }

    #[test]
    fn tally_empty_results_is_empty() {
        assert!(tally_by_language(&[]).is_empty());
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
mod small_model_gate_tests {
    use super::is_small_model;

    #[test]
    fn nano_sessions_count_as_small_models_for_bench_gating() {
        // G4-1 ACs: nano must trip the same memory-off guard as tiny.
        assert!(
            is_small_model("auto", "qwen2.5-3b"),
            "auto-profile + nano-detected id ⇒ small"
        );
        assert!(
            is_small_model("nano", "some-model-id"),
            "explicit --profile nano ⇒ small"
        );
        assert!(
            is_small_model("tiny", "some-model-id"),
            "tiny parity preserved"
        );
        // Medium parity preserved: these stay OUT of the guard.
        assert!(
            !is_small_model("auto", "ornith-1.5:9b"),
            "1.5 without trailing b never matched nano detection"
        );
        assert!(!is_small_model("auto", "qwen3:8b"));
        assert!(!is_small_model("medium", "qwen2.5-3b"));
    }
}
