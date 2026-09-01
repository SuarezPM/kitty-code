//! ClawEval integration (F2).
//!
//! - Preflight checks: CLI availability, Docker, API key, fixtures, sandbox image.
//! - Wrapper execution: spawns `claw-eval batch` as subprocess, waits for completion.
//! - Artifact extraction: parses `batch_summary.json` + `batch_results.json`.
//! - Results tagged as "local approximation — not official ClawEval leaderboard".

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Wrapper version ─────────────────────────────────────────────────

/// Local wrapper version (not upstream claw-eval version).
pub const CLAWEVAL_WRAPPER_VERSION: &str = "0.1.0-claweval-local-v1";

/// Compatibility label — honest tag for results.
pub const CLAWEVAL_COMPATIBILITY: &str = "local_approximation_not_official";

/// Default number of trials per task.
pub const CLAWEVAL_DEFAULT_TRIALS: usize = 3;

/// Default parallelism.
#[allow(dead_code)]
pub const CLAWEVAL_DEFAULT_PARALLEL: usize = 1;

// ── Preflight ──────────────────────────────────────────────────────

/// Status of ClawEval preflight checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClawEvalPreflight {
    /// `claw-eval` CLI found on PATH (`command -v claw-eval`).
    pub cli_available: bool,
    /// Docker daemon reachable (`docker info`).
    pub docker_available: bool,
    /// `OPENROUTER_API_KEY` env var set and non-empty.
    pub api_key_set: bool,
    /// Fixtures archive present (`data/fixtures.tar.gz` or HuggingFace download).
    pub fixtures_available: bool,
    /// Sandbox Docker image present (`claw-eval-agent:latest`).
    pub sandbox_image: bool,
    /// Output of `claw-eval --version`.
    pub version: Option<String>,
}

/// Run all preflight checks and return the status.
pub fn preflight_claweval() -> ClawEvalPreflight {
    let cli_available = command_exists("claw-eval");
    let docker_available = check_docker();
    let api_key_set = check_api_key();
    let fixtures_available = check_fixtures();
    let sandbox_image = check_sandbox_image();
    let version = get_version();

    ClawEvalPreflight {
        cli_available,
        docker_available,
        api_key_set,
        fixtures_available,
        sandbox_image,
        version,
    }
}

/// Minimum viable: CLI available AND API key set.
pub fn is_ready(pf: &ClawEvalPreflight) -> bool {
    pf.cli_available && pf.api_key_set
}

/// Human-readable list of what is missing.
pub fn blocked_reasons(pf: &ClawEvalPreflight) -> Vec<String> {
    let mut reasons = Vec::new();
    if !pf.cli_available {
        reasons.push("claw-eval CLI not found on PATH (pip install claw-eval)".to_string());
    }
    if !pf.docker_available {
        reasons.push("Docker daemon not reachable (docker info failed)".to_string());
    }
    if !pf.api_key_set {
        reasons.push("OPENROUTER_API_KEY env var not set or empty".to_string());
    }
    if !pf.fixtures_available {
        reasons.push(
            "ClawEval fixtures not found (data/fixtures.tar.gz or HuggingFace download required)"
                .to_string(),
        );
    }
    if !pf.sandbox_image {
        reasons.push(
            "Sandbox image claw-eval-agent:latest not found (docker pull claw-ai/claw-eval-agent:latest)"
                .to_string(),
        );
    }
    reasons
}

// ── Preflight helpers ──────────────────────────────────────────────

/// Check if a command exists on PATH.
fn command_exists(cmd: &str) -> bool {
    std::process::Command::new("command")
        .args(["-v", cmd])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check Docker daemon via `docker info`.
fn check_docker() -> bool {
    std::process::Command::new("docker")
        .args(["info"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Check `OPENROUTER_API_KEY` env var is set and non-empty.
fn check_api_key() -> bool {
    std::env::var("OPENROUTER_API_KEY")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// Check if fixtures archive exists.
fn check_fixtures() -> bool {
    // Check data/fixtures.tar.gz relative to CWD or known locations
    let candidates = [
        PathBuf::from("data/fixtures.tar.gz"),
        PathBuf::from("fixtures.tar.gz"),
    ];
    for p in &candidates {
        if p.exists() {
            return true;
        }
    }
    // Also check if a fixtures dir exists
    let dirs = [PathBuf::from("data/fixtures"), PathBuf::from("fixtures")];
    for d in &dirs {
        if d.is_dir() {
            return true;
        }
    }
    false
}

/// Check if sandbox Docker image exists.
fn check_sandbox_image() -> bool {
    std::process::Command::new("docker")
        .args(["image", "inspect", "claw-eval-agent:latest"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Get `claw-eval --version` output.
fn get_version() -> Option<String> {
    let output = std::process::Command::new("claw-eval")
        .arg("--version")
        .output()
        .ok()?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            None
        } else {
            Some(stdout)
        }
    } else {
        None
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors produced by ClawEval wrapper.
#[derive(Debug, Clone)]
pub enum ClawEvalError {
    /// Preflight failed: cannot run.
    Blocked { reasons: Vec<String> },
    /// `claw-eval batch` subprocess failed.
    SubprocessFailed {
        exit_code: Option<i32>,
        stderr: String,
    },
    /// Output directory does not exist or is not writable.
    OutputDirError { path: PathBuf, help: String },
    /// `batch_summary.json` not found or invalid.
    BatchSummaryMissing { path: PathBuf, help: String },
    /// `batch_results.json` not found or invalid.
    BatchResultsMissing { path: PathBuf, help: String },
    /// JSON parsing error.
    ParseError(String),
    /// IO error.
    Io(String),
}

impl std::fmt::Display for ClawEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blocked { reasons } => {
                write!(f, "ClawEval blocked:\n{}", reasons.join("\n"))
            }
            Self::SubprocessFailed { exit_code, stderr } => {
                write!(
                    f,
                    "claw-eval batch failed (exit code {:?}):\n{}",
                    exit_code, stderr
                )
            }
            Self::OutputDirError { path, help } => {
                write!(f, "ClawEval output dir error at {}\n{}", path.display(), help)
            }
            Self::BatchSummaryMissing { path, help } => {
                write!(
                    f,
                    "batch_summary.json not found or invalid at {}\n{}",
                    path.display(),
                    help
                )
            }
            Self::BatchResultsMissing { path, help } => {
                write!(
                    f,
                    "batch_results.json not found or invalid at {}\n{}",
                    path.display(),
                    help
                )
            }
            Self::ParseError(s) => write!(f, "ClawEval parse error: {s}"),
            Self::Io(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for ClawEvalError {}

// ── Run configuration ──────────────────────────────────────────────

/// Configuration for a ClawEval batch run.
#[derive(Debug, Clone)]
pub struct ClawEvalRunConfig {
    /// Model identifier (e.g. "openrouter/anthropic/claude-3-opus").
    pub model: String,
    /// OpenRouter API key.
    pub api_key: String,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Number of trials per task (default 3).
    pub trials: usize,
    /// Parallel tasks (default 1).
    pub parallel: usize,
    /// Use sandbox mode (default false).
    pub sandbox: bool,
    /// Specific task IDs to run (None = all tasks).
    pub task_ids: Option<Vec<String>>,
    /// Optional path to claw-eval config file.
    pub config_path: Option<PathBuf>,
    /// Output directory for results.
    pub output_dir: PathBuf,
}

// ── Result types ───────────────────────────────────────────────────

/// Aggregated result from a ClawEval batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClawEvalResult {
    pub model: String,
    pub total_tasks: usize,
    pub trials_per_task: usize,
    pub average_score: f64,
    pub pass_at_k: f64,
    pub pass_power_k: f64,
    #[serde(default)]
    pub per_task: Vec<ClawEvalTaskResult>,
}

/// Per-task result from ClawEval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClawEvalTaskResult {
    pub task_id: String,
    pub trials: Vec<ClawEvalTrialResult>,
}

/// Per-trial result from ClawEval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClawEvalTrialResult {
    pub trial: usize,
    pub trace_file: Option<String>,
    pub score: f64,
    pub passed: bool,
    pub completion: f64,
    pub safety: f64,
    pub robustness: f64,
}

/// Status of a ClawEval run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClawEvalStatus {
    /// Missing CLI, Docker, or API key — cannot run.
    Blocked { reasons: Vec<String> },
    /// Executed locally — results are local approximation, not official.
    Executed { result: ClawEvalResult, official: bool },
}

// ── Execution ──────────────────────────────────────────────────────

/// Run `claw-eval batch` as a subprocess and return the parsed result.
///
/// No network calls in this function; the subprocess handles all API communication.
pub async fn run_claweval_batch(cfg: &ClawEvalRunConfig) -> Result<ClawEvalResult, ClawEvalError> {
    // Preflight: check readiness
    let pf = preflight_claweval();
    if !is_ready(&pf) {
        let reasons = blocked_reasons(&pf);
        return Err(ClawEvalError::Blocked { reasons });
    }

    // Ensure output dir exists
    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| ClawEvalError::OutputDirError {
        path: cfg.output_dir.clone(),
        help: format!("failed to create output dir: {e}"),
    })?;

    // Build command
    let mut cmd = std::process::Command::new("claw-eval");
    cmd.arg("batch");
    cmd.args(["--model", &cfg.model]);
    cmd.args(["--trials", &cfg.trials.to_string()]);
    cmd.args(["--parallel", &cfg.parallel.to_string()]);
    cmd.args(["--output-dir", &cfg.output_dir.to_string_lossy()]);

    if cfg.sandbox {
        cmd.arg("--sandbox");
    }

    if let Some(ref base_url) = cfg.base_url {
        cmd.args(["--base-url", base_url]);
    }

    if let Some(ref task_ids) = cfg.task_ids {
        let ids = task_ids.join(",");
        cmd.args(["--tasks", &ids]);
    }

    if let Some(ref config_path) = cfg.config_path {
        cmd.args(["--config", &config_path.to_string_lossy()]);
    }

    // Set API key in environment
    cmd.env("OPENROUTER_API_KEY", &cfg.api_key);

    // Spawn and wait
    let output = cmd
        .output()
        .map_err(|e| ClawEvalError::Io(format!("failed to spawn claw-eval: {e}")))?;

    if !output.status.success() {
        let exit_code = output.status.code();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(ClawEvalError::SubprocessFailed { exit_code, stderr });
    }

    // Parse results from output directory
    let summary_path = cfg.output_dir.join("batch_summary.json");
    let results_path = cfg.output_dir.join("batch_results.json");

    parse_batch_output(&summary_path, &results_path)
}

/// Parse `batch_summary.json` and `batch_results.json` into `ClawEvalResult`.
pub fn parse_batch_output(
    summary_path: &Path,
    results_path: &Path,
) -> Result<ClawEvalResult, ClawEvalError> {
    let summary_json = std::fs::read_to_string(summary_path).map_err(|e| {
        ClawEvalError::BatchSummaryMissing {
            path: summary_path.to_path_buf(),
            help: format!("failed to read: {e}"),
        }
    })?;

    let per_task = parse_batch_results(results_path)?;

    let summary: serde_json::Value = serde_json::from_str(&summary_json)
        .map_err(|e| ClawEvalError::ParseError(format!("batch_summary.json: {e}")))?;

    let model = summary
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let total_tasks = summary
        .get("total_tasks")
        .and_then(|v| v.as_u64())
        .unwrap_or(per_task.len() as u64) as usize;

    let trials_per_task = summary
        .get("trials_per_task")
        .and_then(|v| v.as_u64())
        .unwrap_or(CLAWEVAL_DEFAULT_TRIALS as u64) as usize;

    let average_score = summary
        .get("average_score")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let pass_at_k = summary
        .get("pass_at_k")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let pass_power_k = summary
        .get("pass_power_k")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Ok(ClawEvalResult {
        model,
        total_tasks,
        trials_per_task,
        average_score,
        pass_at_k,
        pass_power_k,
        per_task,
    })
}

/// Parse per-task results from `batch_results.json`.
pub fn parse_batch_results(path: &Path) -> Result<Vec<ClawEvalTaskResult>, ClawEvalError> {
    let content = std::fs::read_to_string(path).map_err(|e| ClawEvalError::BatchResultsMissing {
        path: path.to_path_buf(),
        help: format!("failed to read: {e}"),
    })?;

    let raw: Vec<serde_json::Value> = serde_json::from_str(&content)
        .map_err(|e| ClawEvalError::ParseError(format!("batch_results.json: {e}")))?;

    let mut tasks = Vec::with_capacity(raw.len());

    for task_val in raw {
        let task_id = task_val
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let trials_raw = task_val.get("trials").and_then(|v| v.as_array());

        let trials = match trials_raw {
            Some(arr) => {
                let mut v = Vec::with_capacity(arr.len());
                for (idx, trial_val) in arr.iter().enumerate() {
                    v.push(ClawEvalTrialResult {
                        trial: trial_val
                            .get("trial")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(idx as u64) as usize,
                        trace_file: trial_val
                            .get("trace_file")
                            .and_then(|t| t.as_str())
                            .map(String::from),
                        score: trial_val
                            .get("score")
                            .and_then(|s| s.as_f64())
                            .unwrap_or(0.0),
                        passed: trial_val
                            .get("passed")
                            .and_then(|p| p.as_bool())
                            .unwrap_or(false),
                        completion: trial_val
                            .get("completion")
                            .and_then(|c| c.as_f64())
                            .unwrap_or(0.0),
                        safety: trial_val
                            .get("safety")
                            .and_then(|s| s.as_f64())
                            .unwrap_or(0.0),
                        robustness: trial_val
                            .get("robustness")
                            .and_then(|r| r.as_f64())
                            .unwrap_or(0.0),
                    });
                }
                v
            }
            None => Vec::new(),
        };

        tasks.push(ClawEvalTaskResult { task_id, trials });
    }

    Ok(tasks)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflight_returns_blocked_when_cli_missing() {
        // claw-eval is not installed, so cli_available should be false
        let pf = preflight_claweval();
        // In the test env, claw-eval is NOT available (per context)
        assert!(!pf.cli_available, "claw-eval CLI should not be present in test env");
        assert!(!is_ready(&pf), "should not be ready when CLI missing");
        let reasons = blocked_reasons(&pf);
        assert!(
            reasons.iter().any(|r| r.contains("claw-eval CLI not found")),
            "should mention CLI missing, got: {reasons:?}"
        );
    }

    #[test]
    fn test_parse_batch_summary_valid() {
        let dir = tempfile::tempdir().unwrap();
        let summary_path = dir.path().join("batch_summary.json");
        let results_path = dir.path().join("batch_results.json");

        // Write valid summary
        let summary = serde_json::json!({
            "model": "openrouter/test-model",
            "total_tasks": 2,
            "trials_per_task": 3,
            "average_score": 0.75,
            "pass_at_k": 0.666,
            "pass_power_k": 0.333,
        });
        std::fs::write(&summary_path, summary.to_string()).unwrap();

        // Write valid results
        let results = serde_json::json!([
            {
                "task_id": "task_1",
                "trials": [
                    {"trial": 0, "score": 0.9, "passed": true, "completion": 0.8, "safety": 1.0, "robustness": 0.9},
                    {"trial": 1, "score": 0.7, "passed": false, "completion": 0.6, "safety": 1.0, "robustness": 0.7},
                    {"trial": 2, "score": 0.8, "passed": true, "completion": 0.7, "safety": 1.0, "robustness": 0.8}
                ]
            },
            {
                "task_id": "task_2",
                "trials": [
                    {"trial": 0, "score": 0.6, "passed": false, "completion": 0.5, "safety": 1.0, "robustness": 0.6}
                ]
            }
        ]);
        std::fs::write(&results_path, results.to_string()).unwrap();

        let result = parse_batch_output(&summary_path, &results_path).unwrap();
        assert_eq!(result.model, "openrouter/test-model");
        assert_eq!(result.total_tasks, 2);
        assert_eq!(result.trials_per_task, 3);
        assert!((result.average_score - 0.75).abs() < 1e-9);
        assert_eq!(result.per_task.len(), 2);
        assert_eq!(result.per_task[0].task_id, "task_1");
        assert_eq!(result.per_task[0].trials.len(), 3);
        assert_eq!(result.per_task[1].task_id, "task_2");
        assert_eq!(result.per_task[1].trials.len(), 1);
        // Check trial fields
        assert!(result.per_task[0].trials[0].passed);
        assert!(!result.per_task[0].trials[1].passed);
        assert!((result.per_task[0].trials[0].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_parse_batch_results_per_task() {
        let dir = tempfile::tempdir().unwrap();
        let results_path = dir.path().join("batch_results.json");

        let results = serde_json::json!([
            {
                "task_id": "eval_math",
                "trials": [
                    {"trial": 0, "trace_file": "traces/t0.json", "score": 1.0, "passed": true, "completion": 1.0, "safety": 1.0, "robustness": 1.0}
                ]
            }
        ]);
        std::fs::write(&results_path, results.to_string()).unwrap();

        let tasks = parse_batch_results(&results_path).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "eval_math");
        assert_eq!(tasks[0].trials.len(), 1);
        assert_eq!(tasks[0].trials[0].trace_file.as_deref(), Some("traces/t0.json"));
        assert!(tasks[0].trials[0].passed);
    }

    #[test]
    fn test_blocked_reasons_empty_when_ready() {
        let pf = ClawEvalPreflight {
            cli_available: true,
            docker_available: true,
            api_key_set: true,
            fixtures_available: true,
            sandbox_image: true,
            version: Some("1.0.0".to_string()),
        };
        assert!(is_ready(&pf));
        assert!(blocked_reasons(&pf).is_empty());
    }

    #[test]
    fn test_is_ready_false_when_no_cli() {
        let pf = ClawEvalPreflight {
            cli_available: false,
            docker_available: true,
            api_key_set: true,
            fixtures_available: true,
            sandbox_image: true,
            version: None,
        };
        assert!(!is_ready(&pf));
    }

    #[test]
    fn test_is_ready_false_when_no_api_key() {
        let pf = ClawEvalPreflight {
            cli_available: true,
            docker_available: true,
            api_key_set: false,
            fixtures_available: true,
            sandbox_image: true,
            version: Some("1.0.0".to_string()),
        };
        assert!(!is_ready(&pf));
    }

    #[test]
    fn test_parse_batch_summary_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let summary_path = dir.path().join("batch_summary.json");
        let results_path = dir.path().join("batch_results.json");

        // summary does not exist
        let err = parse_batch_output(&summary_path, &results_path).unwrap_err();
        assert!(matches!(err, ClawEvalError::BatchSummaryMissing { .. }));
    }

    #[test]
    fn test_parse_batch_results_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let results_path = dir.path().join("batch_results.json");

        let err = parse_batch_results(&results_path).unwrap_err();
        assert!(matches!(err, ClawEvalError::BatchResultsMissing { .. }));
    }

    #[test]
    fn test_parse_batch_summary_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let summary_path = dir.path().join("batch_summary.json");
        let results_path = dir.path().join("batch_results.json");

        std::fs::write(&summary_path, "not json").unwrap();
        std::fs::write(&results_path, "[]").unwrap();

        let err = parse_batch_output(&summary_path, &results_path).unwrap_err();
        assert!(matches!(err, ClawEvalError::ParseError(_)));
    }

    #[test]
    fn test_claw_eval_error_display() {
        let err = ClawEvalError::Blocked {
            reasons: vec!["missing CLI".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("ClawEval blocked"));
        assert!(msg.contains("missing CLI"));

        let err = ClawEvalError::SubprocessFailed {
            exit_code: Some(1),
            stderr: "some error".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("claw-eval batch failed"));
        assert!(msg.contains("some error"));
    }

    #[test]
    fn test_claw_eval_status_serialization() {
        let status = ClawEvalStatus::Blocked {
            reasons: vec!["no CLI".to_string()],
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("Blocked"));

        let result = ClawEvalResult {
            model: "test".to_string(),
            total_tasks: 1,
            trials_per_task: 3,
            average_score: 0.5,
            pass_at_k: 0.33,
            pass_power_k: 0.33,
            per_task: vec![],
        };
        let status = ClawEvalStatus::Executed {
            result,
            official: false,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("Executed"));
        assert!(json.contains("official"));
    }

    #[test]
    fn test_batch_results_empty_array() {
        let dir = tempfile::tempdir().unwrap();
        let results_path = dir.path().join("batch_results.json");

        std::fs::write(&results_path, "[]").unwrap();

        let tasks = parse_batch_results(&results_path).unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_parse_batch_summary_defaults_when_fields_missing() {
        let dir = tempfile::tempdir().unwrap();
        let summary_path = dir.path().join("batch_summary.json");
        let results_path = dir.path().join("batch_results.json");

        // Minimal summary — missing most fields
        let summary = serde_json::json!({});
        std::fs::write(&summary_path, summary.to_string()).unwrap();
        std::fs::write(&results_path, "[]").unwrap();

        let result = parse_batch_output(&summary_path, &results_path).unwrap();
        assert_eq!(result.model, "unknown");
        assert_eq!(result.total_tasks, 0);
        assert_eq!(result.trials_per_task, CLAWEVAL_DEFAULT_TRIALS);
        assert!((result.average_score).abs() < 1e-9);
        assert!((result.pass_at_k).abs() < 1e-9);
    }
}
