//! Kitty CLI — Command-line interface for Kitty Code.

#[cfg(feature = "tui")]
mod tui;

mod mcp;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kitty_harness::agent::{Agent, AgentConfig};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::tools::standard_tool_definitions;
use kitty_harness::{ModelProfile, ThinkingMode};
use kitty_memory::{Episode, EpisodeStore};
use kitty_runtime::OllamaClient;
use std::path::PathBuf;
use std::time::Instant;
use tracing::warn;

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

    /// Model profile preset: tiny|medium|large (default: medium)
    #[arg(long, global = true, default_value = "medium", value_parser = clap::builder::PossibleValuesParser::new(["tiny", "medium", "large"]))]
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
        /// Benchmark suite: polyglot, tb2, smoke
        #[arg(default_value = "polyglot")]
        suite: String,
        /// Number of exercises to run
        #[arg(short, long, default_value = "15")]
        limit: usize,
        /// Output file for results
        #[arg(short, long, default_value = "bench_results.json")]
        out: PathBuf,
        /// Path to JSON exercises file (overrides built-in exercises)
        #[arg(long)]
        exercises: Option<PathBuf>,
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

    match cli.command {
        Commands::Run { prompt, no_memory } => {
            let prompt = prompt.join(" ");
            // F2 KV preflight: the client cannot introspect the Ollama
            // server's env, so warn (never fail) when a large explicit window
            // risks OOM on 8GB GPUs without acknowledged KV quantization.
            let num_ctx = kitty_runtime::client::parse_num_ctx(
                std::env::var(kitty_runtime::client::ENV_NUM_CTX)
                    .ok()
                    .as_deref(),
            );
            if let Some(warning) = kitty_runtime::client::kv_cache_preflight_warning(
                num_ctx,
                std::env::var(kitty_runtime::client::ENV_KV_QUANT_ACK)
                    .ok()
                    .as_deref(),
            ) {
                warn!("{}", warning);
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
            // Overlay explicit CLI overrides after profile sync (profile wins by default,
            // but explicit CLI flags should survive). We check against clap defaults.
            if cli.context_window != 16384 {
                agent.config_mut().context_window = cli.context_window;
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
                if cli.profile == "tiny" {
                    warn!("--semantic-memory requires medium/large profile, ignoring for tiny");
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
            out,
            exercises: exercises_path,
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

            let cfg = BenchConfig {
                client: client.clone(),
                model: cli.model.clone(),
                limit,
                use_memory: !effective_no_memory,
                thinking_budget: cli.thinking_budget,
                context_window: cli.context_window,
                max_iterations: bench_max_iterations,
                bash_timeout: cli.bash_timeout,
                profile: ModelProfile::from_cli(&cli.profile),
                enable_knowledge: cli.knowledge,
                knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
            };

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
            println!("Bench: polyglot (code), tb2 (terminal)");
            Ok(())
        }
        Commands::Mcp => mcp::run().await,
        Commands::BenchAb {
            thinking,
            limit,
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

            let base_cfg = BenchConfig {
                client: client.clone(),
                model: cli.model.clone(),
                limit,
                use_memory: !effective_no_memory,
                thinking_budget: 2048, // overridden per run below
                context_window: cli.context_window,
                max_iterations: bench_max_iterations,
                bash_timeout: cli.bash_timeout,
                profile: ModelProfile::from_cli(&cli.profile),
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
                let results = &per_budget[budget];
                let passed = results.iter().filter(|r| r.passed).count();
                let total = results.len();
                let rate = if total > 0 {
                    100.0 * passed as f64 / total as f64
                } else {
                    0.0
                };
                let (lo, hi) = wilson_ci(passed, total);
                let p50 = p50_ms(results.iter().map(|r| r.latency_ms).collect());
                per_budget_out.push(serde_json::json!({
                    "budget": budget,
                    "passed": passed,
                    "total": total,
                    "rate": rate,
                    "wilson_ci": [lo, hi],
                    "p50_ms": p50,
                }));
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
    // Re-sync budget threshold from profile (ensures tiny 0.7 vs medium 0.8)
    agent
        .budget_mut()
        .set_threshold(cfg.profile.compact_threshold);
    let outcome =
        tokio::time::timeout(std::time::Duration::from_secs(30), agent.run(&prompt)).await;

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

fn is_small_model(profile: &str, _model: &str) -> bool {
    ModelProfile::from_cli(profile).name == "tiny"
}

#[cfg(test)]
mod bench_ab_tests {
    use super::{build_ab_plan, parse_thinking_budgets, wilson_ci, Cli, Commands};
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
