//! Local JSONL telemetry for harness runs.
//!
//! Writes structured events to rotating JSONL files. NEVER logs prompts
//! or message content — only metadata (model, tokens, latency, errors).
//! Controlled by `KITTY_TELEMETRY` env var (default: enabled, "off" to disable).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::redact::redact_secrets;

/// Maximum file size before rotation (10 MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;
/// Number of rotated files to keep.
const MAX_ROTATIONS: usize = 5;

/// A single telemetry event for one turn.
#[derive(Debug, Clone, Serialize)]
pub struct TurnEvent {
    /// Unix timestamp (seconds since epoch).
    pub ts: u64,
    /// Model identifier.
    pub model: String,
    /// Turn number within the session.
    pub turn: usize,
    /// Number of tool calls in this turn.
    pub tool_calls: usize,
    /// Names of tools called.
    pub tool_names: Vec<String>,
    /// Total latency for the LLM response (milliseconds).
    pub latency_ms: u64,
    /// Tokens in the prompt (if available).
    pub tokens_in: Option<usize>,
    /// Tokens in the response (if available).
    pub tokens_out: Option<usize>,
    /// Error message if the turn failed (redacted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Benchmark suite name (if running a benchmark).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench_suite: Option<String>,
    /// Whether compaction was triggered this turn.
    pub compacted: bool,
    /// W0: autorun hops consumed so far (`None` = unknown/legacy).
    /// Omitted from JSON when `None` so old telemetry stays byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hops: Option<usize>,
    /// W0: tool-loop iterations consumed in the current cycle, 1-based
    /// (`None` = unknown/legacy). Omitted from JSON when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iters: Option<usize>,
    /// W0: top1-top2 margin when available. Always `None` for now — no
    /// logits plumbing in this lane (unlocked by the W3 logprobs spike).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f32>,
    /// W0 (SDD estado-arte): LLM-call wall time for this turn (ms).
    /// Same clock as `latency_ms`; the explicit name anchors the
    /// `wall = modelo/guard/sandbox/tools` breakdown. `None` = unknown/legacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_model_ms: Option<u64>,
    /// W0: accumulated `Guard::check_chain` time within this turn (ms,
    /// truncated — a single check is ~µs, so this is often 0; still a real
    /// measurement, not a placeholder). `None` = unknown/legacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_ms: Option<u64>,
    /// W0: accumulated Bash child-execution wall within this turn (ms) —
    /// the same quantity `SandboxOutcome::duration_ms` measures for the
    /// direct `Sandbox::run` API, taken here at the `ToolRegistry` spawn
    /// site (which owns its own timeout/kill-tree plumbing and does not
    /// route through `Sandbox::run`). Includes containment overhead when
    /// the sandbox is armed. 0/`None` when no Bash ran this turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_ms: Option<u64>,
    /// W0: `process_response` (tool execution + result handling) wall for
    /// this turn (ms). Always `>= guard_ms + sandbox_ms` (nested intervals,
    /// same clock; floor-truncation preserves the ordering).
    /// `None` = unknown/legacy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_ms: Option<u64>,
}

/// W0 (SDD estado-arte): per-turn latency breakdown, mirroring the four
/// `TurnEvent` breakdown fields. Kept on `Agent` as `last_turn_breakdown`
/// so tests/benches can observe the desglose without scraping JSONL
/// (the telemetry writer points at the process cwd).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnBreakdown {
    /// LLM-call wall (ms).
    pub wall_model_ms: u64,
    /// `Guard::check_chain` accumulation (ms).
    pub guard_ms: u64,
    /// Bash child-execution accumulation (ms).
    pub sandbox_ms: u64,
    /// `process_response` wall (ms).
    pub tools_ms: u64,
}

/// Telemetry writer with file rotation.
pub struct TelemetryWriter {
    dir: PathBuf,
    current_file: PathBuf,
    current_size: u64,
    enabled: bool,
}

impl TelemetryWriter {
    /// Create a new writer. If `dir` is None, uses `.kitty/telemetry/` in the workspace.
    /// Checks `KITTY_TELEMETRY=off` to disable.
    pub fn new(dir: Option<&Path>) -> Self {
        let enabled = crate::config::telemetry_enabled();

        let default_dir = PathBuf::from(".kitty/telemetry");
        let base = match dir {
            Some(p) => p.to_path_buf(),
            None => default_dir,
        };
        let file_path = base.join("current.jsonl");

        if enabled {
            std::fs::create_dir_all(&base).ok();
        }

        let current_size = if enabled {
            std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        Self {
            dir: base,
            current_file: file_path,
            current_size,
            enabled,
        }
    }

    /// Write a turn event. Applies redaction. Rotates if needed.
    /// W1 async: the append runs on the blocking pool via `spawn_blocking`
    /// (established pattern tools.rs:1338, response.rs:329) so per-turn
    /// telemetry never parks the reactor; rotation (rare, 10MB) stays sync.
    pub async fn write_turn(&mut self, event: &TurnEvent) {
        if !self.enabled {
            return;
        }

        let mut json = serde_json::to_string(event).unwrap_or_default();
        json = redact_secrets(&json);

        // Rotate if needed
        if self.current_size + json.len() as u64 + 1 > MAX_FILE_SIZE {
            self.rotate();
        }

        let path = self.current_file.clone();
        let line = format!("{json}\n");
        let appended = tokio::task::spawn_blocking(move || {
            use std::fs::OpenOptions;
            use std::io::Write;
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(mut file) => file.write_all(line.as_bytes()).is_ok(),
                Err(_) => false,
            }
        })
        .await
        .unwrap_or(false);
        if appended {
            self.current_size += json.len() as u64 + 1;
        }
    }

    /// Rotate current file to numbered backup, start fresh.
    fn rotate(&mut self) {
        // Remove oldest rotation if at limit
        let mut entries: Vec<_> = std::fs::read_dir(&self.dir)
            .ok()
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "jsonl")
                            .unwrap_or(false)
                            && e.path()
                                .file_name()
                                .map(|n| n != "current.jsonl")
                                .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();

        entries.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH)
        });

        while entries.len() >= MAX_ROTATIONS {
            if let Some(oldest) = entries.first() {
                let _ = std::fs::remove_file(oldest.path());
                entries.remove(0);
            }
        }

        // Rename current to numbered backup
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let backup = self.dir.join(format!("{ts}.jsonl"));
        let _ = std::fs::rename(&self.current_file, &backup);

        self.current_size = 0;
    }
}

/// Check if telemetry is enabled (for callers that want to skip expensive prep).
pub fn is_telemetry_enabled() -> bool {
    crate::config::telemetry_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Serializes env-mutating telemetry tests: KITTY_TELEMETRY is
    /// process-global, so the disabled-by-env case must not interleave
    /// with enabled cases (W1 async widened the race window).
    static ENV_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn tmp_dir() -> PathBuf {
        // Unique across parallel tests: nanos alone collide when tests
        // start within the same tick (latent flake, exposed by W1 async).
        static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "kitty_telemetry_test_{}_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            n
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ENV_SERIAL serializes env-mutating tests; per-test current-thread runtime
    async fn write_turn_creates_file() {
        let _env = ENV_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp_dir();
        let mut writer = TelemetryWriter::new(Some(&dir));
        let event = TurnEvent {
            ts: 1234567890,
            model: "test".into(),
            turn: 1,
            tool_calls: 2,
            tool_names: vec!["Read".into(), "Bash".into()],
            latency_ms: 150,
            tokens_in: Some(100),
            tokens_out: Some(50),
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        writer.write_turn(&event).await;

        let content = fs::read_to_string(dir.join("current.jsonl")).unwrap();
        assert!(content.contains("\"model\":\"test\""));
        assert!(content.contains("\"turn\":1"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ENV_SERIAL serializes env-mutating tests; per-test current-thread runtime
    async fn telemetry_disabled_by_env() {
        let _env = ENV_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp_dir();
        std::env::set_var("KITTY_TELEMETRY", "off");
        let mut writer = TelemetryWriter::new(Some(&dir));
        let event = TurnEvent {
            ts: 1234567890,
            model: "test".into(),
            turn: 1,
            tool_calls: 0,
            tool_names: vec![],
            latency_ms: 0,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        writer.write_turn(&event).await;

        assert!(!dir.join("current.jsonl").exists());
        std::env::set_var("KITTY_TELEMETRY", "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ENV_SERIAL serializes env-mutating tests; per-test current-thread runtime
    async fn redaction_applied_to_telemetry() {
        let _env = ENV_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp_dir();
        let mut writer = TelemetryWriter::new(Some(&dir));
        let event = TurnEvent {
            ts: 1234567890,
            model: "test".into(),
            turn: 1,
            tool_calls: 1,
            tool_names: vec!["Bash".into()],
            latency_ms: 100,
            tokens_in: None,
            tokens_out: None,
            error: Some("API key sk-abc123def456ghi789jkl012mno failed".into()),
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        writer.write_turn(&event).await;

        let content = fs::read_to_string(dir.join("current.jsonl")).unwrap();
        assert!(content.contains("[REDACTED]"));
        assert!(!content.contains("sk-abc123"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ENV_SERIAL serializes env-mutating tests; per-test current-thread runtime
    async fn rotation_triggered_at_max_size() {
        let _env = ENV_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tmp_dir();
        // Create a file close to max size to trigger rotation
        let file_path = dir.join("current.jsonl");
        let big_content = "x".repeat((MAX_FILE_SIZE - 100) as usize);
        fs::write(&file_path, &big_content).unwrap();

        let mut writer = TelemetryWriter::new(Some(&dir));
        writer.current_size = MAX_FILE_SIZE - 50; // Almost at limit

        let event = TurnEvent {
            ts: 1234567890,
            model: "test".into(),
            turn: 1,
            tool_calls: 0,
            tool_names: vec![],
            latency_ms: 0,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        writer.write_turn(&event).await;

        // Should have rotated: old file renamed, new current.jsonl created
        let entries: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "jsonl")
                    .unwrap_or(false)
            })
            .collect();
        assert!(entries.len() >= 2, "should have rotated file");
        let _ = fs::remove_dir_all(&dir);
    }

    /// W0: `None` effort fields are omitted, so legacy telemetry JSON is
    /// byte-compatible (no new keys unless there is real data).
    #[test]
    fn effort_fields_omitted_when_none_keeps_old_format() {
        let event = TurnEvent {
            ts: 1,
            model: "m".into(),
            turn: 0,
            tool_calls: 0,
            tool_names: vec![],
            latency_ms: 0,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            !json.contains("hops"),
            "old format must not gain keys: {json}"
        );
        assert!(
            !json.contains("iters"),
            "old format must not gain keys: {json}"
        );
        assert!(
            !json.contains("margin"),
            "old format must not gain keys: {json}"
        );
    }

    /// W0: present effort fields serialize with their values.
    #[test]
    fn effort_fields_serialize_when_present() {
        let event = TurnEvent {
            ts: 1,
            model: "m".into(),
            turn: 3,
            tool_calls: 1,
            tool_names: vec!["Read".into()],
            latency_ms: 10,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: Some(2),
            iters: Some(4),
            margin: Some(0.07),
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        let v: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(v["hops"], serde_json::json!(2));
        assert_eq!(v["iters"], serde_json::json!(4));
        assert!((v["margin"].as_f64().unwrap_or(-1.0) - 0.07).abs() < 1e-6);
    }

    /// W0: the latency-breakdown fields are omitted when `None`, so legacy
    /// telemetry JSON stays byte-identical (no migrations).
    #[test]
    fn breakdown_fields_omitted_when_none_keeps_old_format() {
        let event = TurnEvent {
            ts: 1,
            model: "m".into(),
            turn: 0,
            tool_calls: 0,
            tool_names: vec![],
            latency_ms: 5,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: None,
            guard_ms: None,
            sandbox_ms: None,
            tools_ms: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        for key in ["wall_model_ms", "guard_ms", "sandbox_ms", "tools_ms"] {
            assert!(
                !json.contains(key),
                "old format must not gain {key}: {json}"
            );
        }
    }

    /// W0: present breakdown fields serialize with their values.
    #[test]
    fn breakdown_fields_serialize_when_present() {
        let event = TurnEvent {
            ts: 1,
            model: "m".into(),
            turn: 0,
            tool_calls: 1,
            tool_names: vec!["Bash".into()],
            latency_ms: 120,
            tokens_in: None,
            tokens_out: None,
            error: None,
            bench_suite: None,
            compacted: false,
            hops: None,
            iters: None,
            margin: None,
            wall_model_ms: Some(120),
            guard_ms: Some(0),
            sandbox_ms: Some(95),
            tools_ms: Some(100),
        };
        let v: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(v["wall_model_ms"], serde_json::json!(120));
        assert_eq!(v["guard_ms"], serde_json::json!(0));
        assert_eq!(v["sandbox_ms"], serde_json::json!(95));
        assert_eq!(v["tools_ms"], serde_json::json!(100));
    }
}
