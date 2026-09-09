//! Terminal-Bench 2.1 runner — wrapper around Harbor CLI.
//!
//! Invokes `harbor run` with the official TB2 dataset and parses results.
//! Requires: harbor CLI, Docker, Ollama running.

use anyhow::{Context, Result};
use std::process::Command;

/// Official TB2.1 dataset id (Harbor registry).
///
/// NOTE: the local `-p bench/official/terminal-bench-2` tree vendored in this
/// repo is the 2.0 dataset — it is NOT the 2.1 source. Official runs always
/// use the `-d` registry id below.
pub const TB2_DATASET: &str = "terminal-bench/terminal-bench-2-1";

/// Hub tag pinned on `-d <name@version>` (`harbor run --help` documents
/// `-d` as `Dataset name@version`, e.g. `dataset@1.0`, with no digest form;
/// fleet harbor is 0.22.0, so the `latest` tag is used and the exact Hub
/// rev-6 / sha256:7d7bdc1c… must be recorded in the run manifest).
pub const TB2_DATASET_VERSION: &str = "latest";

/// Pinned `-d` value (`<name@version>`); never run bare `-d` floating on
/// Hub latest without recording rev-6 / sha256:7d7bdc1c… in the manifest.
pub fn tb2_dataset_pinned() -> String {
    format!("{TB2_DATASET}@{TB2_DATASET_VERSION}")
}

/// Full TB2.1 task count: evidence runs cover all 89 tasks (no `-l` flag);
/// smaller `--limit` values select at most N tasks after filters
/// (order-dependent smoke-only selection, never evidence-grade).
/// NOTE: a Hub rev bump changing this count voids prior baselines —
/// void-and-re-baseline, never auto-pass.
pub const TB2_FULL_TASK_COUNT: usize = 89;

/// TB2 run configuration.
#[derive(Debug, Clone)]
pub struct Tb2Config {
    pub model: String,
    pub api_base: String,
    pub trials: usize,
    pub parallel: usize,
    pub timeout_multiplier: u64,
    pub max_turns: usize,
    /// Bounded task selection (`-l/--n-tasks`): `Some(n)` runs at most N
    /// tasks after filters (order-dependent smoke-only selection);
    /// `None` runs the full 89-task set.
    pub task_limit: Option<usize>,
}

/// Map the shared `--limit` flag to a Harbor task selection: bounded N for
/// smokes (maximum N after filters, order-dependent, never evidence-grade);
/// the full 89-task set (no `-l` flag) for evidence runs.
/// NOTE: `Some(0)` is returned for `--limit 0`; callers (tb2-official path)
/// must reject 0 before building the command.
pub fn task_limit_for(limit: usize) -> Option<usize> {
    if limit >= TB2_FULL_TASK_COUNT {
        None
    } else {
        Some(limit)
    }
}

/// Resolved task count for a manifest: `task_limit.unwrap_or(FULL)`.
/// Record alongside `TB2_FULL_TASK_COUNT` everywhere a manifest/output is
/// written; a Hub rev bump changing the count voids prior baselines —
/// void-and-re-baseline, never auto-pass.
pub fn resolved_task_count(task_limit: Option<usize>) -> usize {
    task_limit.unwrap_or(TB2_FULL_TASK_COUNT)
}

/// TB2 run result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Tb2Result {
    pub model: String,
    pub total_tasks: usize,
    pub resolved: usize,
    pub resolution_rate: f64,
    pub per_task: Vec<Tb2TaskResult>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Tb2TaskResult {
    pub task_id: String,
    pub passed: bool,
    pub reward: f64,
}

/// Preflight check status.
#[derive(Debug, Clone)]
pub struct Preflight {
    pub harbor_ok: bool,
    pub harbor_version: Option<String>,
    pub docker_ok: bool,
}

/// Check if Harbor CLI is available.
pub fn check_harbor() -> (bool, Option<String>) {
    match Command::new("harbor").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let version = String::from_utf8_lossy(&o.stdout).trim().to_string();
            (true, Some(version))
        }
        _ => (false, None),
    }
}

/// Check if Docker is available.
pub fn check_docker() -> bool {
    Command::new("docker")
        .arg("info")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run preflight checks.
pub fn preflight() -> Preflight {
    let (harbor_ok, harbor_version) = check_harbor();
    let docker_ok = check_docker();
    Preflight {
        harbor_ok,
        harbor_version,
        docker_ok,
    }
}

/// Check if preflight passes.
pub fn is_ready(pf: &Preflight) -> bool {
    pf.harbor_ok && pf.docker_ok
}

/// List blocking reasons from preflight.
pub fn blocked_reasons(pf: &Preflight) -> Vec<String> {
    let mut issues = Vec::new();
    if !pf.harbor_ok {
        issues.push("harbor CLI not found — install: uv tool install harbor".to_string());
    }
    if !pf.docker_ok {
        issues.push("Docker not running or not accessible".to_string());
    }
    issues
}

/// Build the harbor run command.
pub fn build_command(config: &Tb2Config) -> Vec<String> {
    // Pinned `-d <name@version>` (`harbor run --help`: `Dataset
    // name@version`); `latest` floats on Hub, so rev-6 / sha256:7d7bdc1c…
    // must be recorded in the run manifest — never bare `-d` on Hub latest.
    let mut cmd = vec![
        "harbor".to_string(),
        "run".to_string(),
        "-d".to_string(),
        tb2_dataset_pinned(),
        "-a".to_string(),
        "terminus-2".to_string(),
        "-m".to_string(),
        format!("ollama_chat/{}", config.model),
        "--ak".to_string(),
        format!("api_base={}", config.api_base),
        // Ornith eval-params pin: Harbor sends no sampling default,
        // so temperature + top_p must be passed explicitly (1.0 / 1.0).
        "--ak".to_string(),
        "temperature=1.0".to_string(),
        "--ak".to_string(),
        "top_p=1.0".to_string(),
        "--ak".to_string(),
        format!("max_turns={}", config.max_turns),
        "--ak".to_string(),
        "model_info={\"max_input_tokens\":16384,\"max_output_tokens\":8192,\"input_cost_per_token\":0,\"output_cost_per_token\":0}".to_string(),
        "--agent-timeout-multiplier".to_string(),
        config.timeout_multiplier.to_string(),
        "-n".to_string(),
        config.parallel.to_string(),
    ];

    // N-selection: bounded N for smokes (`-l` is Harbor `--n-tasks`,
    // maximum N after filters, order-dependent smoke-only selection);
    // omitted for full evidence runs.
    if let Some(n) = config.task_limit {
        // Caller rejects --limit 0; 0 must never reach Harbor as `-l 0`.
        debug_assert!(n > 0);
        cmd.push("-l".to_string());
        cmd.push(n.to_string());
    }

    if config.trials > 1 {
        cmd.push("-k".to_string());
        cmd.push(config.trials.to_string());
    }

    cmd
}

/// Parse harbor output for results.
pub fn parse_output(output: &str) -> Tb2Result {
    let mut resolved = 0;
    let mut total = 0;
    let mut per_task = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Look for task result lines. Harbor outputs patterns like:
        //   "task-name ... PASSED" or "task-name ... FAILED"
        // or just standalone PASSED/FAILED markers.
        if trimmed.contains("PASSED") || trimmed.contains("passed") {
            resolved += 1;
            total += 1;
            let task_id = extract_task_id(trimmed);
            per_task.push(Tb2TaskResult {
                task_id,
                passed: true,
                reward: 1.0,
            });
        } else if trimmed.contains("FAILED") || trimmed.contains("failed") {
            total += 1;
            let task_id = extract_task_id(trimmed);
            per_task.push(Tb2TaskResult {
                task_id,
                passed: false,
                reward: 0.0,
            });
        }
    }

    let resolution_rate = if total > 0 {
        resolved as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    Tb2Result {
        model: String::new(),
        total_tasks: total,
        resolved,
        resolution_rate,
        per_task,
    }
}

/// Try to extract a task ID from a harbor output line.
fn extract_task_id(line: &str) -> String {
    // Try to get the part before "..." or "PASSED" or "FAILED"
    if let Some(pos) = line.find("...") {
        let candidate = line[..pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    if let Some(pos) = line.rfind("PASSED") {
        let candidate = line[..pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    if let Some(pos) = line.rfind("passed") {
        let candidate = line[..pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    if let Some(pos) = line.rfind("FAILED") {
        let candidate = line[..pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    if let Some(pos) = line.rfind("failed") {
        let candidate = line[..pos].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    line.to_string()
}

/// Execute a full TB2 run: preflight, build command, run, parse.
pub async fn run_tb2(config: &Tb2Config) -> Result<Tb2Result> {
    // Preflight
    let pf = preflight();
    if !is_ready(&pf) {
        let reasons = blocked_reasons(&pf);
        anyhow::bail!("TB2 preflight failed:\n  {}", reasons.join("\n  "));
    }

    let version_str = pf.harbor_version.as_deref().unwrap_or("unknown");
    println!("  Harbor version: {version_str}");

    // Build and display command
    let cmd = build_command(config);
    println!("  Running: {}", cmd.join(" "));

    // Execute harbor
    let output = tokio::task::spawn_blocking({
        let cmd = cmd.clone();
        move || {
            Command::new(&cmd[0])
                .args(&cmd[1..])
                .output()
                .context("failed to execute harbor")
        }
    })
    .await
    .context("spawn_blocking failed")?
    .context("harbor execution failed")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        eprintln!("Harbor stderr:\n{stderr}");
        anyhow::bail!("harbor exited with status {}", output.status);
    }

    // Parse results
    let mut result = parse_output(&stdout);
    result.model = config.model.clone();

    // If no results parsed from stdout, try stderr
    if result.total_tasks == 0 {
        result = parse_output(&stderr);
        result.model = config.model.clone();
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_command_basic() {
        let config = Tb2Config {
            model: "ornith:9b".to_string(),
            api_base: "http://127.0.0.1:11434".to_string(),
            trials: 1,
            parallel: 1,
            timeout_multiplier: 3,
            max_turns: 50,
            task_limit: None,
        };
        let cmd = build_command(&config);
        assert!(cmd.contains(&"harbor".to_string()));
        assert!(cmd.contains(&"ollama_chat/ornith:9b".to_string()));
        // Exact `-d` value: pinned `<name@version>`, never bare Hub latest
        // without a rev-6 / sha256:7d7bdc1c… manifest record.
        let d_idx = cmd.iter().position(|x| x == "-d").expect("missing -d flag");
        assert_eq!(cmd[d_idx + 1], tb2_dataset_pinned());
        assert_eq!(
            cmd[d_idx + 1],
            format!("{TB2_DATASET}@{TB2_DATASET_VERSION}")
        );
        assert!(cmd[d_idx + 1].contains('@'));
        // Ornith eval-params pin: explicit sampling, never the old temp 0.
        assert!(cmd.contains(&"temperature=1.0".to_string()));
        assert!(cmd.contains(&"top_p=1.0".to_string()));
        assert!(!cmd.iter().any(|x| x == "temperature=0"));
        // Full run: no `-l` flag.
        assert!(!cmd.iter().any(|x| x == "-l"));
    }

    #[test]
    fn build_command_with_trials() {
        let config = Tb2Config {
            model: "ornith:9b".to_string(),
            api_base: "http://127.0.0.1:11434".to_string(),
            trials: 3,
            parallel: 2,
            timeout_multiplier: 5,
            max_turns: 100,
            task_limit: None,
        };
        let cmd = build_command(&config);
        let k_idx = cmd.iter().position(|x| x == "-k").expect("missing -k flag");
        assert_eq!(cmd[k_idx + 1], "3");
        let n_idx = cmd.iter().position(|x| x == "-n").expect("missing -n flag");
        assert_eq!(cmd[n_idx + 1], "2");
    }

    #[test]
    fn build_command_task_limit_smoke() {
        let config = Tb2Config {
            model: "ornith:9b".to_string(),
            api_base: "http://127.0.0.1:11434".to_string(),
            trials: 1,
            parallel: 1,
            timeout_multiplier: 3,
            max_turns: 50,
            task_limit: Some(5),
        };
        let cmd = build_command(&config);
        let l_idx = cmd.iter().position(|x| x == "-l").expect("missing -l flag");
        assert_eq!(cmd[l_idx + 1], "5");
    }

    #[test]
    fn task_limit_for_maps_limit_flag() {
        // Bounded N for smokes (maximum N after filters, never evidence);
        // full 89-task set for evidence runs.
        // `--limit 0` maps to `Some(0)` here so the tb2-official path can
        // bail loudly instead of silently running `-l 0`.
        assert_eq!(task_limit_for(0), Some(0));
        assert_eq!(task_limit_for(5), Some(5));
        assert_eq!(task_limit_for(15), Some(15));
        assert_eq!(task_limit_for(88), Some(88));
        assert_eq!(task_limit_for(89), None);
        assert_eq!(task_limit_for(200), None);
    }

    #[test]
    fn default_cli_limit_15_is_smoke_not_evidence() {
        // Default CLI `--limit 15` is a smoke selection (maximum 15 after
        // filters, order-dependent), never evidence-grade; evidence needs
        // the full set (no `-l` flag).
        assert_eq!(task_limit_for(15), Some(15));
        assert!(task_limit_for(15).is_some());
        assert_eq!(resolved_task_count(task_limit_for(15)), 15);
    }

    #[test]
    fn resolved_task_count_pairs_with_full_count() {
        // Manifests record both counts; a Hub rev bump changing the count
        // voids prior baselines (void-and-re-baseline, never auto-pass).
        assert_eq!(resolved_task_count(Some(15)), 15);
        assert_eq!(resolved_task_count(None), TB2_FULL_TASK_COUNT);
    }

    #[test]
    fn dataset_pin_uses_hub_tag() {
        // `-d` pins `<name@version>`; `harbor run --help` documents only
        // `Dataset name@version` (no digest form), so Hub tag `latest` is
        // used and rev-6 / sha256:7d7bdc1c… goes in the run manifest.
        assert_eq!(TB2_DATASET_VERSION, "latest");
        assert_eq!(
            tb2_dataset_pinned(),
            "terminal-bench/terminal-bench-2-1@latest"
        );
    }

    #[test]
    fn parse_output_basic() {
        let output = "Task 1: PASSED\nTask 2: FAILED\nTask 3: PASSED";
        let result = parse_output(output);
        assert_eq!(result.total_tasks, 3);
        assert_eq!(result.resolved, 2);
        assert!((result.resolution_rate - 66.67).abs() < 0.1);
    }

    #[test]
    fn parse_output_with_dots() {
        let output = "task-a ...\n  PASSED\ntask-b ...\n  FAILED";
        let result = parse_output(output);
        assert_eq!(result.total_tasks, 2);
        assert_eq!(result.resolved, 1);
    }

    #[test]
    fn parse_output_empty() {
        let result = parse_output("");
        assert_eq!(result.total_tasks, 0);
        assert_eq!(result.resolved, 0);
        assert_eq!(result.resolution_rate, 0.0);
    }

    #[test]
    fn extract_task_id_from_harbor_line() {
        assert_eq!(extract_task_id("my-task ... PASSED"), "my-task");
        assert_eq!(extract_task_id("my-task ... FAILED"), "my-task");
        // Bare status line with no preceding task name → returns the line itself
        assert_eq!(extract_task_id("PASSED"), "PASSED");
        assert_eq!(extract_task_id("some task name FAILED"), "some task name");
    }

    #[test]
    fn blocked_reasons_for_missing_harbor() {
        let pf = Preflight {
            harbor_ok: false,
            harbor_version: None,
            docker_ok: true,
        };
        let reasons = blocked_reasons(&pf);
        assert_eq!(reasons.len(), 1);
        assert!(reasons[0].contains("harbor"));
    }

    #[test]
    fn blocked_reasons_for_missing_docker() {
        let pf = Preflight {
            harbor_ok: true,
            harbor_version: Some("0.1.0".to_string()),
            docker_ok: false,
        };
        let reasons = blocked_reasons(&pf);
        assert_eq!(reasons.len(), 1);
        assert!(reasons[0].contains("Docker"));
    }

    #[test]
    fn blocked_reasons_for_both_missing() {
        let pf = Preflight {
            harbor_ok: false,
            harbor_version: None,
            docker_ok: false,
        };
        let reasons = blocked_reasons(&pf);
        assert_eq!(reasons.len(), 2);
    }

    #[test]
    fn is_ready_requires_both() {
        let pf = Preflight {
            harbor_ok: true,
            harbor_version: Some("0.1.0".to_string()),
            docker_ok: true,
        };
        assert!(is_ready(&pf));
        let pf = Preflight {
            harbor_ok: false,
            harbor_version: None,
            docker_ok: true,
        };
        assert!(!is_ready(&pf));
        let pf = Preflight {
            harbor_ok: true,
            harbor_version: Some("0.1.0".to_string()),
            docker_ok: false,
        };
        assert!(!is_ready(&pf));
    }
}
