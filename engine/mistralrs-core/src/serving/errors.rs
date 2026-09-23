//! Friendly errors with actionable hints + bounded request log ring.
//!
//! Every error maps to `{ error, hint }` so `kitty run` users get a next
//! step, not a stack trace. Hints cover OOM/KV/engine-core; unknown errors
//! say so instead of guessing.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Wire body for failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
    pub hint: String,
}

/// Classified serving failure.
#[derive(Debug, Clone)]
pub enum KittyError {
    /// GPU/CPU out of memory while loading or stepping.
    Oom { detail: String },
    /// KV-cache exhausted (context too long for resident allocation).
    KvExhausted { detail: String },
    /// Engine core failure (pipeline/load/scheduler).
    EngineCore { detail: String },
    /// Unknown model id.
    NotFound { model: String },
    /// Bad request (validation).
    Validation { detail: String },
}


/// Shared OOM predicate (registry retry + friendly errors stay in sync).
/// Explicit parens; a bare `memory` only counts with a companion verb
/// (`alloc`/`exhaust`/`failed`/`could not allocate`) so benign mentions
/// like "memory usage 5GB" stay EngineCore.
pub(crate) fn is_oom_message(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("oom")
        || lower.contains("out of memory")
        || lower.contains("out_of_memory")
        || lower.contains("vram pre-check failed")
        || (lower.contains("memory")
            && (lower.contains("alloc")
                || lower.contains("exhaust")
                || lower.contains("failed")
                || lower.contains("could not allocate")))
}

impl KittyError {
    pub fn oom(detail: impl Into<String>) -> Self {
        Self::Oom { detail: detail.into() }
    }

    pub fn kv(detail: impl Into<String>) -> Self {
        Self::KvExhausted { detail: detail.into() }
    }

    pub fn engine(detail: impl Into<String>) -> Self {
        Self::EngineCore { detail: detail.into() }
    }

    /// Classify a free-form anyhow/engine error string. Never invents a
    /// cause: falls back to EngineCore when nothing matches.
    pub fn classify(msg: &str) -> Self {
        if is_oom_message(msg) {
            Self::oom(msg)
        } else {
            let lower = msg.to_lowercase();
            if (lower.contains("kv")
                && (lower.contains("exhaust")
                    || lower.contains("cache")
                    || lower.contains("paged")))
                || (lower.contains("context length") && lower.contains("exceed"))
            {
                Self::kv(msg)
            } else {
                Self::engine(msg)
            }
        }
    }
    pub fn status_code(&self) -> u16 {
        match self {
            Self::Oom { .. } => 507,
            Self::KvExhausted { .. } => 507,
            Self::EngineCore { .. } => 500,
            Self::NotFound { .. } => 404,
            Self::Validation { .. } => 400,
        }
    }

    pub fn hint(&self) -> String {
        match self {
            Self::Oom { .. } => "GPU OOM: reload with a smaller quant (Q4_K_M), fewer GPU layers (--gpu-layers), or shorter context (--ctx 4096). Check `kitty gpu status` first.".to_string(),
            Self::KvExhausted { .. } => "KV cache exhausted: shorten --ctx, reduce concurrent requests, or reload with a smaller model.".to_string(),
            Self::EngineCore { .. } => "Engine failure: retry once; if it repeats, capture `kitty logs --tail 50` and reload the model.".to_string(),
            Self::NotFound { model } => format!("Model `{model}` is not resident. Load it with `kitty load <spec>` first."),
            Self::Validation { .. } => "Invalid request: check model, messages, and token limits.".to_string(),
        }
    }

    pub fn body(&self) -> ErrorBody {
        let error = match self {
            Self::Oom { detail } => format!("out of memory: {detail}"),
            Self::KvExhausted { detail } => format!("kv cache exhausted: {detail}"),
            Self::EngineCore { detail } => format!("engine error: {detail}"),
            Self::NotFound { model } => format!("model not found: {model}"),
            Self::Validation { detail } => format!("invalid request: {detail}"),
        };
        ErrorBody { error, hint: self.hint() }
    }
}

impl std::fmt::Display for KittyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = self.body();
        write!(f, "{} ({})", b.error, b.hint)
    }
}

/// One line in the bounded request log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub seq: u64,
    pub route: String,
    pub model: String,
    pub status: u16,
    pub detail: String,
}

/// Fixed-capacity ring for recent requests. Oldest entries drop first.
pub struct RequestLog {
    ring: Mutex<VecDeque<LogEntry>>,
    cap: usize,
    next: Mutex<u64>,
}

impl RequestLog {
    pub fn new(cap: usize) -> Self {
        Self {
            ring: Mutex::new(VecDeque::with_capacity(cap.max(1))),
            cap: cap.max(1),
            next: Mutex::new(1),
        }
    }

    pub fn push(&self, route: &str, model: &str, status: u16, detail: &str) {
        let mut next = self.next.lock().unwrap();
        let seq = *next;
        *next += 1;
        drop(next);
        let mut ring = self.ring.lock().unwrap();
        if ring.len() >= self.cap {
            ring.pop_front();
        }
        ring.push_back(LogEntry {
            seq,
            route: route.to_string(),
            model: model.to_string(),
            status,
            detail: detail.to_string(),
        });
    }

    pub fn tail(&self, n: usize) -> Vec<LogEntry> {
        let ring = self.ring.lock().unwrap();
        let n = n.min(ring.len()).max(0);
        ring.iter().skip(ring.len() - n).cloned().collect()
    }
}

impl Default for RequestLog {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_and_hint() {
        let e = KittyError::classify("CUDA OOM allocating 2GiB");
        assert_eq!(e.status_code(), 507);
        assert!(e.hint().contains("kitty gpu status"));
        let e = KittyError::classify("paged kv cache exhausted");
        assert!(e.hint().contains("KV cache"));
        // The VRAM pre-check message must classify as OOM so the registry
        // retry (0.9 -> 0.6) engages instead of failing fast.
        let e = KittyError::classify("VRAM pre-check failed: need ~11GB but only 7GB free");
        assert_eq!(e.status_code(), 507);
        // Bare `memory` without a companion verb is NOT OOM.
        let e = KittyError::classify("memory usage 5GB");
        assert_eq!(e.status_code(), 500);
        // ...but with one it is.
        let e = KittyError::classify("failed to allocate memory");
        assert_eq!(e.status_code(), 507);
    }

    #[test]
    fn ring_bounded() {
        let log = RequestLog::new(2);
        log.push("a", "m", 200, "ok");
        log.push("b", "m", 200, "ok");
        log.push("c", "m", 500, "boom");
        let tail = log.tail(10);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].route, "b");
        assert_eq!(tail[1].route, "c");
    }
}
