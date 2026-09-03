//! Terminal-Bench 2.1 runner — wrapper around Harbor CLI.
//!
//! Invokes `harbor run` with the official TB2 dataset and parses results.
//! Requires: harbor CLI, Docker, Ollama running.

use anyhow::{Context, Result};
use std::process::Command;

/// TB2 run configuration.
#[derive(Debug, Clone)]
pub struct Tb2Config {
    pub model: String,
    pub api_base: String,
    pub trials: usize,
    pub parallel: usize,
    pub timeout_multiplier: u64,
    pub max_turns: usize,
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
    let mut cmd = vec![
        "harbor".to_string(),
        "run".to_string(),
        "-d".to_string(),
        "terminal-bench/terminal-bench-2".to_string(),
        "-a".to_string(),
        "terminus-2".to_string(),
        "-m".to_string(),
        format!("ollama_chat/{}", config.model),
        "--ak".to_string(),
        format!("api_base={}", config.api_base),
        "--ak".to_string(),
        "temperature=0".to_string(),
        "--ak".to_string(),
        format!("max_turns={}", config.max_turns),
        "--ak".to_string(),
        "model_info={\"max_input_tokens\":16384,\"max_output_tokens\":8192,\"input_cost_per_token\":0,\"output_cost_per_token\":0}".to_string(),
        "--agent-timeout-multiplier".to_string(),
        config.timeout_multiplier.to_string(),
        "-n".to_string(),
        config.parallel.to_string(),
    ];

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
        };
        let cmd = build_command(&config);
        assert!(cmd.contains(&"harbor".to_string()));
        assert!(cmd.contains(&"ollama_chat/ornith:9b".to_string()));
        assert!(cmd.contains(&"terminal-bench/terminal-bench-2".to_string()));
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
        };
        let cmd = build_command(&config);
        let k_idx = cmd.iter().position(|x| x == "-k").expect("missing -k flag");
        assert_eq!(cmd[k_idx + 1], "3");
        let n_idx = cmd.iter().position(|x| x == "-n").expect("missing -n flag");
        assert_eq!(cmd[n_idx + 1], "2");
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
