//! Kitty CLI — Command-line interface for Kitty Code.

#[cfg(feature = "tui")]
mod tui;

mod mcp;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kitty_harness::agent::{Agent, AgentConfig};
use kitty_harness::tools::standard_tool_definitions;
use kitty_harness::ModelProfile;
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
    #[arg(long)]
    pub knowledge: bool,

    /// Directory containing knowledge cards (default: knowledge/)
    #[arg(long, default_value = "knowledge")]
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
            let memory_db_path = if no_memory {
                None
            } else {
                Some(default_db_path())
            };
            let base_config = AgentConfig {
                model: cli.model.clone(),
                enable_knowledge: cli.knowledge,
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
                agent.config_mut().max_thinking_tokens = Some(cli.thinking_budget);
            }
            if cli.bash_timeout != 60 {
                agent.config_mut().bash_timeout_secs = cli.bash_timeout;
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
            let bench_max_iterations = if cli.max_iterations == 20 { 5 } else { cli.max_iterations };
            // Gate 2: --no-memory by default for small models (e.g. qwen3:4b) to avoid retrieval overhead.
            // Effective memory off if either --no-memory flag set or model looks small.
            let is_small_model = cli.model.contains(":4b")
                || cli.model.contains(":2b")
                || cli.model.contains(":1b")
                || cli.model.contains("1.7b")
                || cli.model.contains("0.5b");
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
            };

            println!("Kitty Code Benchmark");
            println!("  Model: {}", cli.model);
            println!("  Suite: {} (limit: {})", suite, limit);
            println!("  Memory: {}", if effective_no_memory { "off" } else { "on" });
            if bench_max_iterations != cli.max_iterations {
                println!("  Max iterations: {} (bench override, CLI default 20)", bench_max_iterations);
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

            let mut by_lang: std::collections::HashMap<String, (usize, usize)> =
                std::collections::HashMap::new();
            for r in &results {
                let entry = by_lang.entry(r.lang.clone()).or_insert((0, 0));
                entry.0 += 1;
                if r.passed {
                    entry.1 += 1;
                }
            }
            for (lang, (t, p)) in by_lang.iter() {
                println!("  {}: {}/{} ({:.0}%)", lang, p, t, 100.0 * *p as f64 / *t as f64);
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
                "details": results.iter().map(|r| serde_json::json!({
                    "id": r.id,
                    "lang": r.lang,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
            println!("Results saved to {}", out.display());
            // bench-reproducibility: exit-code 0=pass/improved, 1=gate failed, 2=noise (no significant change)
            let baseline_rate: Option<f64> = std::fs::read_to_string("benches/baseline.json")
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
    latency_ms: f64,
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
fn bench(id: &'static str, label: &'static str, prompt: &'static str, check: impl Fn(&str) -> bool + 'static) -> BenchItem {
    BenchItem { id, label, prompt, check: Box::new(check), check_file: "solution.py" }
}

fn bench_file(id: &'static str, label: &'static str, prompt: &'static str, check_file: &'static str, check: impl Fn(&str) -> bool + 'static) -> BenchItem {
    BenchItem { id, label, prompt, check: Box::new(check), check_file }
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
}

/// Shared benchmark driver used by every exercise set.
async fn run_bench(
    items: &[BenchItem],
    cfg: &BenchConfig,
) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();

    let store = if cfg.use_memory {
        Some(EpisodeStore::open(&default_db_path())?)
    } else {
        None
    };

    for (i, ex) in items.iter().take(cfg.limit).enumerate() {
        print!("[{:2}/{:2}] {:20}...", i + 1, cfg.limit.min(items.len()), ex.id);

        // Recall similar past episodes
        if let Some(ref store) = store {
            let past = store.recall_similar(ex.prompt, 3)?;
            if !past.is_empty() {
                print!(" ({} past)", past.len());
            }
        }

        let tmpdir = tempfile::tempdir().context("Failed to create temp dir")?;
        let tmpdir_str = tmpdir.path().to_string_lossy().to_string();

        // Prompt hardening: use real check_file per exercise (fix .py forced for js/ts/rust/go)
        // Gate 2 absolute path prevents CWD contamination.
        let target_file = if ex.check_file.is_empty() { "solution.py" } else { ex.check_file };
        let prompt = format!(
            "{}\n\nWork in directory: {} — Save to {}/{} (use absolute path)",
            ex.prompt, tmpdir_str, tmpdir_str, target_file
        );

        let start = Instant::now();
        let base_config = AgentConfig {
            model: cfg.model.clone(),
            profile: cfg.profile,
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
            agent.config_mut().max_thinking_tokens = Some(cfg.thinking_budget);
        }
        if cfg.bash_timeout != 60 {
            agent.config_mut().bash_timeout_secs = cfg.bash_timeout;
        }
        // Ensure compact threshold follows profile (tiny 0.7, medium 0.8, large 0.8)
        agent.config_mut().compact_threshold = cfg.profile.compact_threshold;
        // Re-sync budget threshold from profile (ensures tiny 0.7 vs medium 0.8)
        agent.budget_mut().set_threshold(cfg.profile.compact_threshold);
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent.run(&prompt),
        ).await;

        // Determine pass/fail and persist an episode for memory.
        let (passed, verdict, latency_ms) = match outcome {
            Ok(Ok(_)) => {
                let passed = (ex.check)(&tmpdir_str);
                (passed, if passed { "pass" } else { "fail" }, start.elapsed().as_millis() as f64)
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

        if let Some(ref store) = store {
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

        results.push(BenchResult {
            id: ex.id.to_string(),
            lang: ex.label.to_string(),
            passed,
            latency_ms,
        });
    }

    Ok(results)
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
}
