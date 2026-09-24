//! Shared benchmark types, exercise loaders, and pure helpers.
//!
//! Extracted from `main.rs` (audit 2026-09 Fase 1.1): the CLI binary owns
//! benchmarking (see kitty-cli AGENTS.md) while `main.rs` stays a thin
//! dispatch layer.

use anyhow::{Context, Result};
use kitty_harness::agent::{Agent, AgentConfig};
use kitty_harness::config::ENV_BENCH_TIMEOUT_SECS;
use kitty_harness::ModelProfile;
#[cfg(feature = "kitty-infer")]
use kitty_runtime::KittyInferClient;
use kitty_runtime::{
    client::RequestOptions, ChatClient, ChatResponse, Message, OllamaClient, ToolDefinition,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use tracing::warn;

// ── Benchmark ────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct BenchResult {
    pub(crate) id: String,
    pub(crate) lang: String,
    pub(crate) passed: bool,
    /// Outcome verdict: "pass" | "fail" | "error" | "timeout".
    pub(crate) verdict: String,
    pub(crate) latency_ms: f64,
}

/// Language for a bench result: parse polyglot task-name prefixes
/// (`polyglot_python_list-ops__koxBfT3` → `python`, `polyglot_go_pig-latin__…`
/// → `go`), falling back to the exercise language label when the id does not
/// carry a `polyglot_<lang>_` prefix.
pub(crate) fn language_of(id: &str, lang_label: &str) -> String {
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
pub(crate) struct LangTally {
    pub(crate) attempted: usize,
    pub(crate) passed: usize,
    pub(crate) failed: usize,
    pub(crate) errored: usize,
}

/// Aggregate results by language (parsed from task names, label fallback).
/// `errored` counts error/timeout outcomes; `failed` counts completed runs
/// that did not pass.
pub(crate) fn tally_by_language(
    results: &[BenchResult],
) -> std::collections::BTreeMap<String, LangTally> {
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
pub(crate) struct JsonExercise {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) lang: String,
    pub(crate) prompt: String,
    /// String to search for in the check_file (polyglot format).
    #[serde(default)]
    pub(crate) check_str: String,
    /// File to check for the check_str.
    #[serde(default)]
    pub(crate) check_file: String,
    /// Check type for TB2 tasks: "file_exists", "file_content_exact", etc.
    #[serde(default)]
    pub(crate) check_type: String,
    /// Relative path for the check (TB2 format).
    #[serde(default)]
    pub(crate) check_path: String,
    /// Expected content for file_content_exact checks.
    #[serde(default)]
    pub(crate) expected: String,
    /// Setup category for TB2 tasks: "file_ops", "text_proc", etc.
    #[serde(default)]
    pub(crate) setup: String,
}

/// Convert a JsonExercise to BenchItem (for polyglot + TB2 JSON formats).
pub(crate) fn json_to_bench_item(ex: &JsonExercise) -> Option<BenchItem> {
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
pub(crate) fn load_json_exercises(path: &std::path::Path) -> Result<Vec<JsonExercise>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read exercises file: {}", path.display()))?;
    let exercises: Vec<JsonExercise> = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse exercises JSON: {}", path.display()))?;
    Ok(exercises)
}

pub(crate) struct BenchItem {
    pub(crate) id: &'static str,
    /// Grouping label shown in results (language or category).
    pub(crate) label: &'static str,
    pub(crate) prompt: &'static str,
    pub(crate) check: Box<dyn Fn(&str) -> bool>,
    pub(crate) check_file: &'static str,
}

/// Helper to build a BenchItem with a boxed closure.
pub(crate) fn bench(
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

pub(crate) fn bench_file(
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

/// Homemade 15-ejercicio fallback (NO oficial): se usa solo cuando la suite
/// no matchea polyglot/tb2/tb2-official/gpqa/claweval ni --exercises JSON.
/// T8.1: nombre explícito homemade vs official (antes `exercises()`).
pub(crate) fn homemade_exercises() -> Vec<BenchItem> {
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

pub(crate) fn file_contains(dir: &str, filename: &str, pattern: &str) -> bool {
    let path = std::path::Path::new(dir).join(filename);
    match std::fs::read_to_string(&path) {
        Ok(content) => content.contains(pattern),
        Err(_) => false,
    }
}

/// Backend-selected bench client (EMBEDDED-BENCH-SPEC §diseño mínimo-1):
/// Ollama default; infer opt-in por env, nunca default silencioso.
/// La sonda `num_ctx` solo corre en el brazo Ollama (infer usa
/// `KITTY_INFER_CTX`); el resto de métricas es idéntico.
#[derive(Clone)]
pub(crate) enum BenchClient {
    Ollama(OllamaClient),
    #[cfg(feature = "kitty-infer")]
    Infer(KittyInferClient),
}

impl ChatClient for BenchClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>> {
        match self {
            Self::Ollama(c) => ChatClient::chat(c, model, messages, tools),
            #[cfg(feature = "kitty-infer")]
            Self::Infer(c) => ChatClient::chat(c, model, messages, tools),
        }
    }

    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>> {
        match self {
            Self::Ollama(c) => ChatClient::chat_with_options(c, model, messages, tools, options),
            #[cfg(feature = "kitty-infer")]
            Self::Infer(c) => ChatClient::chat_with_options(c, model, messages, tools, options),
        }
    }

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<usize>> + Send + 'a>> {
        match self {
            Self::Ollama(c) => ChatClient::count_tokens(c, model, text),
            #[cfg(feature = "kitty-infer")]
            Self::Infer(c) => ChatClient::count_tokens(c, model, text),
        }
    }

    fn health_check<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<()>> + Send + 'a>> {
        match self {
            Self::Ollama(c) => ChatClient::health_check(c),
            #[cfg(feature = "kitty-infer")]
            Self::Infer(c) => ChatClient::health_check(c),
        }
    }
}

/// Shared benchmark configuration.
#[derive(Clone)]
pub(crate) struct BenchConfig {
    pub(crate) client: BenchClient,
    pub(crate) model: String,
    pub(crate) limit: usize,
    pub(crate) use_memory: bool,
    pub(crate) thinking_budget: usize,
    pub(crate) context_window: usize,
    pub(crate) max_iterations: usize,
    /// Autorun continuation hops (f8 T1.3): FLAG-ONLY on bench paths — the
    /// harness never reads `KITTY_AUTORUN_HOPS`, so ambient env cannot leak
    /// into benchmark runs. Default 0.
    pub(crate) autorun_hops: usize,
    /// Per-exercise wall-clock cap in seconds (f9 calibration: resolved via
    /// flag > KITTY_BENCH_TIMEOUT_SECS > manifest cell > default).
    pub(crate) exercise_timeout_secs: u64,
    pub(crate) bash_timeout: u64,
    pub(crate) profile: &'static ModelProfile,
    pub(crate) enable_knowledge: bool,
    pub(crate) knowledge_dir: PathBuf,
}
/// W1: `Agent::new` es sync por diseño (mkdir + SQLite open + knowledge
/// load corren una vez por agente, no por tool-call). En paths async
/// (bench/run/polyglot/gpqa) la construcción se offloadea al blocking pool
/// para no parkar el reactor; los constructores públicos siguen sync.
pub(crate) async fn build_bench_agent(client: BenchClient, config: AgentConfig) -> Result<Agent> {
    tokio::task::spawn_blocking(move || Agent::new(client, config))
        .await
        .context("agent construction task failed")
}
/// Default per-exercise bench wall-clock cap (seconds). Calibration showed
/// real 9B-model tasks on split-GPU hardware routinely exceed this; raise
/// via `--timeout-secs` or `KITTY_BENCH_TIMEOUT_SECS`.
pub const DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS: u64 = 120;

/// Pure resolver for the per-exercise timeout — precedence: explicit CLI
/// flag > `KITTY_BENCH_TIMEOUT_SECS` env > default constant.
///
/// Returns `(resolved_secs, env_invalid)` so the caller can warn exactly
/// once when the env var was present but unparsable (or non-positive): such
/// values fall back to the default rather than producing a zero timeout.
#[must_use]
pub(crate) fn resolve_exercise_timeout_secs(
    explicit_flag: Option<u64>,
    env_raw: Option<&str>,
) -> (u64, bool) {
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
pub(crate) fn resolve_timeout_from_flag(flag_secs: u64) -> (u64, bool) {
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

// ── Per-(model, backend) exercise timeouts (Fase 1 SDD) ─────────────
// El cap único mataba a MiniCPM5-infer (~138s/turno, calidad OK): cada
// celda (modelo, backend) resuelve su timeout desde `timeouts.json`
// (embebido vía `include_str!`, sin IO en el bench). Precedencia en el
// path `bench`: flag explícito > env > celda del manifiesto > default
// seguro (= actual). Gate 50%, semillas e interleave: intocados.

/// One manifest entry: seconds plus measured provenance. `origin` is
/// returned to the caller so the run log records WHY the timeout
/// applies (audit trail, Gate C).
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct TimeoutEntry {
    pub(crate) secs: u64,
    #[serde(default)]
    pub(crate) origin: String,
}

/// Embedded `timeouts.json`: `models[model][backend]` takes precedence over
/// `backends[backend]`, then `default_secs`. Unknown keys fall through; a zero `secs` (or zero
/// `default_secs`) normalizes to [`DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS`].
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct TimeoutManifest {
    #[serde(default)]
    pub(crate) default_secs: u64,
    #[serde(default)]
    pub(crate) backends: std::collections::HashMap<String, TimeoutEntry>,
    #[serde(default)]
    pub(crate) models:
        std::collections::HashMap<String, std::collections::HashMap<String, TimeoutEntry>>,
}

const EMBEDDED_TIMEOUT_MANIFEST: &str = include_str!("timeouts.json");

/// Load the embedded manifest; on parse error warn once and fall back to
/// the safe default (a manifest typo must never fail the bench).
pub(crate) fn load_timeout_manifest() -> TimeoutManifest {
    match serde_json::from_str::<TimeoutManifest>(EMBEDDED_TIMEOUT_MANIFEST) {
        Ok(manifest) => manifest,
        Err(e) => {
            warn!(error = %e, "timeouts.json invalid — using safe default timeout");
            TimeoutManifest::default()
        }
    }
}

/// Resolve `(secs, origin)` for one cell: exact model+backend entry, else
/// backend default, else manifest default, else
/// [`DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS`]. Model match is trimmed +
/// lowercased; backend verbatim (`ollama` | `kitty-infer`).
pub(crate) fn timeout_for_manifest(
    model: &str,
    backend: &str,
    manifest: &TimeoutManifest,
) -> (u64, Option<String>) {
    let key = model.trim().to_lowercase();
    if let Some(per_backend) = manifest.models.get(&key) {
        if let Some(entry) = per_backend.get(backend) {
            if entry.secs > 0 {
                return (entry.secs, Some(entry.origin.clone()));
            }
        }
    }
    if let Some(entry) = manifest.backends.get(backend) {
        if entry.secs > 0 {
            return (entry.secs, Some(entry.origin.clone()));
        }
    }
    if manifest.default_secs > 0 {
        return (manifest.default_secs, None);
    }
    (DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS, None)
}

/// Pure bench-path resolver — precedence: explicit CLI flag >
/// `KITTY_BENCH_TIMEOUT_SECS` env > manifest cell. Same invalid-env
/// contract as [`resolve_exercise_timeout_secs`], falling back to the
/// manifest cell instead of the constant.
#[must_use]
pub(crate) fn resolve_bench_timeout_secs(
    explicit_flag: Option<u64>,
    env_raw: Option<&str>,
    manifest_secs: u64,
) -> (u64, bool) {
    if let Some(secs) = explicit_flag {
        return (secs, false);
    }
    match env_raw.map(str::trim) {
        Some(raw) => match raw.parse::<u64>() {
            Ok(secs) if secs > 0 => (secs, false),
            _ => (manifest_secs, true),
        },
        None => (manifest_secs, false),
    }
}

/// Full `bench`-command resolution: manifest cell for `(model, backend)`,
/// overlaid by explicit flag/env. Returns `(secs, env_invalid, origin)`
/// so the caller warns exactly once on bad env and logs the provenance.
pub(crate) fn resolve_timeout_for_bench(
    flag_secs: u64,
    model: &str,
    backend: &str,
) -> (u64, bool, Option<String>) {
    let manifest = load_timeout_manifest();
    let (manifest_secs, origin) = timeout_for_manifest(model, backend, &manifest);
    let explicit = (flag_secs != DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS).then_some(flag_secs);
    let env_raw = std::env::var(ENV_BENCH_TIMEOUT_SECS).ok();
    let (resolved, invalid) =
        resolve_bench_timeout_secs(explicit, env_raw.as_deref(), manifest_secs);
    if invalid {
        warn!(
            raw = ?env_raw,
            manifest_secs,
            "{ENV_BENCH_TIMEOUT_SECS} is not a positive integer — using manifest cell"
        );
    }
    (resolved, invalid, origin)
}
pub(crate) fn file_is_executable(dir: &str, filename: &str) -> bool {
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

pub(crate) fn list_files(dir: &str) -> Vec<String> {
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
pub(crate) fn bench_exit_code(rate: f64, gate: f64, baseline: Option<f64>) -> i32 {
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
pub(crate) fn is_small_model(profile: &str, model: &str) -> bool {
    // G4-1: gate on the RESOLVED profile name — nano/small sessions
    // (auto-detected or explicit) must get the same small-model bench guard
    // as tiny, or SQLite recall stalls a ≤8K-token session.
    matches!(
        ModelProfile::from_cli(profile, model).name,
        "tiny" | "nano" | "small"
    )
}
#[cfg(test)]
mod homemade_golden_tests {
    // T8.1 goldens patrón ab.rs:315 (seed golden): pin ids/orden homemade.
    // Si cambia un ejercicio a conciencia, actualizar el golden tras revisión,
    // nunca a ciegas. Suite strings (`tb2`, `polyglot`) intactos por compat.
    #[test]
    fn homemade_15_ids_golden() {
        let ids: Vec<&str> = super::homemade_exercises().iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            vec![
                "py-fib",
                "py-class",
                "py-sort",
                "py-decorator",
                "py-context",
                "js-promise",
                "js-debounce",
                "js-async",
                "ts-interface",
                "ts-generic",
                "sh-backup",
                "sh-replace",
                "rust-struct",
                "rust-enum",
                "go-struct",
            ]
        );
    }

    #[test]
    fn tb2_homemade_8_ids_golden() {
        let ids: Vec<&str> = super::super::run::tb2_homemade_exercises()
            .iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(
            ids,
            vec![
                "tb2-bash-count",
                "tb2-py-csv",
                "tb2-makefile",
                "tb2-dockerfile",
                "tb2-gh-actions",
                "tb2-sh-find-modified",
                "tb2-py-dirsize",
                "tb2-sh-backup",
            ]
        );
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

    #[test]
    fn small_sessions_count_as_small_models_for_bench_gating() {
        // Multi-model lane: small-tier ids trip the same memory-off guard.
        assert!(
            is_small_model("auto", "minicpm5:2b"),
            "auto-profile + small-detected id ⇒ small"
        );
        assert!(
            is_small_model("small", "some-model-id"),
            "explicit --profile small ⇒ small"
        );
        assert!(
            is_small_model("auto", "IFM/K2-Horizon-3.7B"),
            "K2 HF id ⇒ small"
        );
        assert!(
            !is_small_model("medium", "minicpm5:2b"),
            "explicit medium beats auto-detection"
        );
    }
}

#[cfg(test)]
mod timeout_manifest_tests {
    use super::{
        load_timeout_manifest, resolve_bench_timeout_secs, timeout_for_manifest, TimeoutManifest,
        DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS,
    };

    #[test]
    fn embedded_manifest_parses_with_safe_default() {
        let manifest = load_timeout_manifest();
        assert_eq!(
            manifest.default_secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS,
            "manifest default must equal the actual default (safe default)"
        );
        assert!(!manifest.models.is_empty(), "measured models present");
    }

    #[test]
    fn each_cell_uses_its_own_timeout_not_the_uniform() {
        let manifest = load_timeout_manifest();
        // Measured 2026-09-23 (Gate C): minicpm5 trivial turn 138.4s on
        // kitty-infer, 22-41s passes on ollama.
        let (infer_secs, infer_origin) =
            timeout_for_manifest("minicpm5:2b", "kitty-infer", &manifest);
        assert_eq!(infer_secs, 150, "minicpm5-infer cell");
        assert!(
            infer_origin.is_some_and(|o| !o.is_empty()),
            "cell carries its origin"
        );
        let (ollama_secs, _) = timeout_for_manifest("minicpm5:2b", "ollama", &manifest);
        assert_eq!(ollama_secs, 60, "minicpm5-ollama cell");
        // Non-uniform by construction: the infer cell differs from default.
        assert_ne!(
            infer_secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS,
            "a cell must not silently equal the uniform default here"
        );
        // Measured passes 95s/114s on kitty-infer.
        let (qwen_secs, _) = timeout_for_manifest("qwen3:4b", "kitty-infer", &manifest);
        assert_eq!(qwen_secs, 180, "qwen3-infer cell");
    }

    #[test]
    fn unknown_cells_fall_back_to_safe_default() {
        let manifest = load_timeout_manifest();
        // Unknown model, known backend: backend default (with its origin).
        let (secs, origin) = timeout_for_manifest("nope:99b", "ollama", &manifest);
        assert_eq!(secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS);
        assert!(
            origin.is_some_and(|o| !o.is_empty()),
            "backend default carries its origin"
        );
        // Unknown model AND backend: manifest default, no cell origin.
        let (secs, origin) = timeout_for_manifest("nope:99b", "nope", &manifest);
        assert_eq!(secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS);
        assert!(origin.is_none(), "default carries no cell origin");
        // Unknown backend for a known model: backend default, then default.
        let (secs, _) = timeout_for_manifest("minicpm5:2b", "weird-backend", &manifest);
        assert_eq!(secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS);
        // Zero entries normalize to the safe default, never to 0.
        let zero = TimeoutManifest {
            default_secs: 0,
            ..Default::default()
        };
        let (secs, _) = timeout_for_manifest("nope:99b", "nope", &zero);
        assert_eq!(secs, DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS);
    }

    #[test]
    fn model_key_match_is_case_insensitive() {
        let manifest = load_timeout_manifest();
        let (secs, _) = timeout_for_manifest("MiniCPM5:2B", "kitty-infer", &manifest);
        assert_eq!(secs, 150);
    }

    #[test]
    fn explicit_flag_and_env_beat_the_manifest_cell() {
        // Explicit flag wins over everything.
        assert_eq!(
            resolve_bench_timeout_secs(Some(45), Some("60"), 150),
            (45, false)
        );
        // Valid env wins over the manifest cell.
        assert_eq!(
            resolve_bench_timeout_secs(None, Some("60"), 150),
            (60, false)
        );
        // Invalid env falls back to the manifest cell (flagged).
        assert_eq!(
            resolve_bench_timeout_secs(None, Some("garbage"), 150),
            (150, true)
        );
        // Neither: the manifest cell applies silently.
        assert_eq!(resolve_bench_timeout_secs(None, None, 150), (150, false));
    }
}
