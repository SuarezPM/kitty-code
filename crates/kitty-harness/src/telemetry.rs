//! Local JSONL telemetry for harness runs.
//!
//! Writes structured events to rotating JSONL files. NEVER logs prompts
//! or message content — only metadata (model, tokens, latency, errors).
//! Controlled by `KITTY_TELEMETRY` env var (default: enabled, "off" to disable).

use std::fs::OpenOptions;
use std::io::Write;
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
        let enabled = std::env::var("KITTY_TELEMETRY")
            .map(|v| v != "off")
            .unwrap_or(true);

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
    pub fn write_turn(&mut self, event: &TurnEvent) {
        if !self.enabled {
            return;
        }

        let mut json = serde_json::to_string(event).unwrap_or_default();
        json = redact_secrets(&json);

        // Rotate if needed
        if self.current_size + json.len() as u64 + 1 > MAX_FILE_SIZE {
            self.rotate();
        }

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.current_file)
        {
            let _ = writeln!(file, "{json}");
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
                            && e.path().file_name().map(|n| n != "current.jsonl").unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();

        entries.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(UNIX_EPOCH));

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
    std::env::var("KITTY_TELEMETRY")
        .map(|v| v != "off")
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kitty_telemetry_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_turn_creates_file() {
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
        };
        writer.write_turn(&event);

        let content = fs::read_to_string(dir.join("current.jsonl")).unwrap();
        assert!(content.contains("\"model\":\"test\""));
        assert!(content.contains("\"turn\":1"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn telemetry_disabled_by_env() {
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
        };
        writer.write_turn(&event);

        assert!(!dir.join("current.jsonl").exists());
        std::env::set_var("KITTY_TELEMETRY", "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn redaction_applied_to_telemetry() {
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
        };
        writer.write_turn(&event);

        let content = fs::read_to_string(dir.join("current.jsonl")).unwrap();
        assert!(content.contains("[REDACTED]"));
        assert!(!content.contains("sk-abc123"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rotation_triggered_at_max_size() {
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
        };
        writer.write_turn(&event);

        // Should have rotated: old file renamed, new current.jsonl created
        let entries: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "jsonl").unwrap_or(false))
            .collect();
        assert!(entries.len() >= 2, "should have rotated file");
        let _ = fs::remove_dir_all(&dir);
    }
}
