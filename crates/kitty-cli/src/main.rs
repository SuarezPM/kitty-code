//! Kitty CLI — Command-line interface for Kitty Code.

#[cfg(feature = "tui")]
mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kitty_harness::agent::{Agent, AgentConfig};
use kitty_harness::tools::standard_tool_definitions;
use kitty_memory::{Episode, EpisodeStore};
use kitty_runtime::OllamaClient;
use std::path::PathBuf;
use std::time::Instant;

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
        /// Disable episodic memory recall/save
        #[arg(long)]
        no_memory: bool,
    },
    /// Show model info and status
    Info,
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
            let config = AgentConfig {
                model: cli.model.clone(),
                max_thinking_tokens: Some(cli.thinking_budget),
                max_iterations: cli.max_iterations,
                context_window: cli.context_window,
                enable_knowledge: cli.knowledge,
                bash_timeout_secs: cli.bash_timeout,
                memory_db_path,
                ..Default::default()
            };
            let mut agent = Agent::new(client, config);

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
            no_memory,
        } => {
            // Ensure memory DB directory exists
            if !no_memory {
                let db_path = default_db_path();
                if let Some(db_dir) = std::path::Path::new(&db_path).parent() {
                    std::fs::create_dir_all(db_dir).ok();
                }
            }

            println!("Kitty Code Benchmark");
            println!("  Model: {}", cli.model);
            println!("  Suite: {} (limit: {})", suite, limit);
            println!("  Memory: {}", if no_memory { "off" } else { "on" });
            println!("  Thinking budget: {}", cli.thinking_budget);
            println!("  Context window: {}", cli.context_window);
            println!();

            let results = match suite.as_str() {
                "tb2" => run_tb2_bench(&client, &cli.model, limit, !no_memory, cli.thinking_budget, cli.context_window, cli.max_iterations, cli.bash_timeout).await?,
                _ => run_polyglot_bench(&client, &cli.model, limit, !no_memory, cli.thinking_budget, cli.context_window, cli.max_iterations, cli.bash_timeout).await?,
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

            let gate = 45.0;
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
                "memory": !no_memory,
                "details": results.iter().map(|r| serde_json::json!({
                    "id": r.id,
                    "lang": r.lang,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                })).collect::<Vec<_>>(),
            });
            std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
            println!("Results saved to {}", out.display());
            Ok(())
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

struct BenchItem {
    id: &'static str,
    /// Grouping label shown in results (language or category).
    label: &'static str,
    prompt: &'static str,
    check: fn(&str) -> bool,
}

fn exercises() -> Vec<BenchItem> {
    vec![
        BenchItem { id: "py-fib", label: "python", prompt: "Write a function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers. Save to solution.py.", check: |d| file_contains(d, "solution.py", "def fibonacci") },
        BenchItem { id: "py-class", label: "python", prompt: "Create a BankAccount class with deposit() and withdraw() methods. Save to solution.py.", check: |d| file_contains(d, "solution.py", "class BankAccount") },
        BenchItem { id: "py-sort", label: "python", prompt: "Write merge_sort(arr: list[int]) -> list[int]. Save to solution.py.", check: |d| file_contains(d, "solution.py", "def merge_sort") },
        BenchItem { id: "py-decorator", label: "python", prompt: "Write a retry decorator that retries up to 3 times on exception. Save to solution.py.", check: |d| file_contains(d, "solution.py", "def retry") },
        BenchItem { id: "py-context", label: "python", prompt: "Write a context manager that times code execution. Save to solution.py.", check: |d| file_contains(d, "solution.py", "__enter__") },
        BenchItem { id: "js-promise", label: "javascript", prompt: "Write fetchWithTimeout(url, timeout) using AbortController. Save to solution.js.", check: |d| file_contains(d, "solution.js", "AbortController") },
        BenchItem { id: "js-debounce", label: "javascript", prompt: "Write a debounce function. Save to solution.js.", check: |d| file_contains(d, "solution.js", "debounce") },
        BenchItem { id: "js-async", label: "javascript", prompt: "Write an async function to read a file. Save to solution.js.", check: |d| file_contains(d, "solution.js", "async") },
        BenchItem { id: "ts-interface", label: "typescript", prompt: "Write a TypeScript interface User with id, name, email. Save to solution.ts.", check: |d| file_contains(d, "solution.ts", "interface User") },
        BenchItem { id: "ts-generic", label: "typescript", prompt: "Write generic first<T>(arr: T[]): T | undefined. Save to solution.ts.", check: |d| file_contains(d, "solution.ts", "function first") },
        BenchItem { id: "sh-backup", label: "bash", prompt: "Write a bash script that backs up a directory to timestamped tar.gz. Save to solution.sh.", check: |d| file_contains(d, "solution.sh", "tar") },
        BenchItem { id: "sh-replace", label: "bash", prompt: "Write a bash script to find and replace text in .txt files. Save to solution.sh.", check: |d| file_contains(d, "solution.sh", "find") },
        BenchItem { id: "rust-struct", label: "rust", prompt: "Write a Rust struct Point with x,y f64 and new() + distance_to(). Save to solution.rs.", check: |d| file_contains(d, "solution.rs", "struct Point") },
        BenchItem { id: "rust-enum", label: "rust", prompt: "Write enum Shape with Circle/Rectangle/Triangle and area(). Save to solution.rs.", check: |d| file_contains(d, "solution.rs", "enum Shape") },
        BenchItem { id: "go-struct", label: "go", prompt: "Write Go struct Person with Name/Age and String() method. Save to solution.go.", check: |d| file_contains(d, "solution.go", "type Person struct") },
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

        let prompt = format!(
            "{}\n\nWork in directory: {}",
            ex.prompt, tmpdir_str
        );

        let start = Instant::now();
        let config = AgentConfig {
            model: cfg.model.clone(),
            max_thinking_tokens: Some(cfg.thinking_budget),
            context_window: cfg.context_window,
            max_iterations: cfg.max_iterations,
            bash_timeout_secs: cfg.bash_timeout,
            ..Default::default()
        };
        let mut agent = Agent::new(cfg.client.clone(), config);
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(120),
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
                (false, "timeout", 120000.0)
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

#[allow(clippy::too_many_arguments)]
async fn run_polyglot_bench(
    client: &OllamaClient,
    model: &str,
    limit: usize,
    use_memory: bool,
    thinking_budget: usize,
    context_window: usize,
    max_iterations: usize,
    bash_timeout: u64,
) -> Result<Vec<BenchResult>> {
    let cfg = BenchConfig {
        client: client.clone(),
        model: model.to_string(),
        limit,
        use_memory,
        thinking_budget,
        context_window,
        max_iterations,
        bash_timeout,
    };
    run_bench(&exercises(), &cfg).await
}

// ── TB2: Terminal-Bench 2.0 ──────────────────────────────────────────

fn tb2_exercises() -> Vec<BenchItem> {
    vec![
        BenchItem {
            id: "tb2-bash-count",
            label: "bash",
            prompt: "Create a bash script called count_lines.sh that counts the number of lines in all .rs files in the current directory recursively. Save it to count_lines.sh and make it executable.",
            check: |d| file_is_executable(d, "count_lines.sh") && file_contains(d, "count_lines.sh", "find"),
        },
        BenchItem {
            id: "tb2-py-csv",
            label: "python",
            prompt: "Write a Python script called csv_to_json.py that reads a CSV file (input path as first argument) and outputs JSON to stdout. Save it to csv_to_json.py.",
            check: |d| file_contains(d, "csv_to_json.py", "csv") && file_contains(d, "csv_to_json.py", "json"),
        },
        BenchItem {
            id: "tb2-makefile",
            label: "makefile",
            prompt: "Create a Makefile with three targets: test (runs cargo test), build (runs cargo build), and clean (removes target directory). Save to Makefile.",
            check: |d| file_contains(d, "Makefile", "test:") && file_contains(d, "Makefile", "build:") && file_contains(d, "Makefile", "clean:"),
        },
        BenchItem {
            id: "tb2-dockerfile",
            label: "docker",
            prompt: "Write a Dockerfile for a Rust project that uses multi-stage builds: builder stage compiles, runtime stage copies binary. Save to Dockerfile.",
            check: |d| file_contains(d, "Dockerfile", "FROM") && file_contains(d, "Dockerfile", "COPY"),
        },
        BenchItem {
            id: "tb2-gh-actions",
            label: "ci",
            prompt: "Create a GitHub Actions CI workflow at .github/workflows/ci.yml that runs cargo test and cargo build on push/PR. Save to .github/workflows/ci.yml.",
            check: |d| {
                let path = std::path::Path::new(d).join(".github/workflows/ci.yml");
                match std::fs::read_to_string(&path) {
                    Ok(c) => c.contains("cargo test") || c.contains("cargo build"),
                    Err(_) => false,
                }
            },
        },
        BenchItem {
            id: "tb2-sh-find-modified",
            label: "bash",
            prompt: "Write a bash script called find_recent.sh that finds all files modified in the last 24 hours in a given directory (first argument). Save to find_recent.sh.",
            check: |d| file_is_executable(d, "find_recent.sh") && file_contains(d, "find_recent.sh", "find"),
        },
        BenchItem {
            id: "tb2-py-dirsize",
            label: "python",
            prompt: "Write a Python script called dirsize.py that calculates and prints the total size of each subdirectory in the current directory. Save to dirsize.py.",
            check: |d| file_contains(d, "dirsize.py", "os") && file_contains(d, "dirsize.py", "path"),
        },
        BenchItem {
            id: "tb2-sh-backup",
            label: "bash",
            prompt: "Write a bash script called backup.sh that backs up files from a source directory (first argument) to a destination with a timestamp suffix (e.g. backup_20260817_120000). Save to backup.sh.",
            check: |d| file_is_executable(d, "backup.sh") && file_contains(d, "backup.sh", "cp"),
        },
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

#[allow(clippy::too_many_arguments)]
async fn run_tb2_bench(
    client: &OllamaClient,
    model: &str,
    limit: usize,
    use_memory: bool,
    thinking_budget: usize,
    context_window: usize,
    max_iterations: usize,
    bash_timeout: u64,
) -> Result<Vec<BenchResult>> {
    let cfg = BenchConfig {
        client: client.clone(),
        model: model.to_string(),
        limit,
        use_memory,
        thinking_budget,
        context_window,
        max_iterations,
        bash_timeout,
    };
    run_bench(&tb2_exercises(), &cfg).await
}
