//! Central configuration for `KITTY_*` environment variables.
//!
//! Single source of truth for variable NAMES and typed resolution
//! (audit 2026-09 Fase 1.4). Rules:
//!
//! - Every harness-owned `KITTY_*` name is a `pub const` here; readers import
//!   the const or call the typed accessor — never a string literal.
//! - Runtime-owned vars (`KITTY_NUM_CTX`, `KITTY_NO_THINK`,
//!   `KITTY_KV_QUANT_ACK`, `KITTY_CLOUD_*`, `KITTY_LLAMA_MODEL`) keep their
//!   consts in `kitty-runtime` (dependency direction: harness → runtime) and
//!   are re-exported here for discovery.
//! - Sandbox child-process vars (`KITTY_SANDBOX_CHILD_*`,
//!   `KITTY_SANDBOX_MEM_MB`, …) are read by the sandboxed child / arming path
//!   and stay in `sandbox.rs` (documented exception).
//! - Test-only reads (env save/restore inside `#[cfg(test)]`) manipulate env
//!   directly and are unaffected.
//!
//! Typed accessors follow a pure-core + env-wrapper split: the `_raw`
//! functions take `Option<&str>` and are unit-testable without mutating
//! process env; the env wrappers read once at call time.

/// Kernel containment opt-out for Bash children (`=off` disables, loud).
pub const ENV_SANDBOX: &str = "KITTY_SANDBOX";
/// Telemetry JSONL toggle (`=off` disables the per-turn writer).
pub const ENV_TELEMETRY: &str = "KITTY_TELEMETRY";
/// Directory where oversized tool outputs are spilled (E3).
pub const ENV_TRUNCATION_DIR: &str = "KITTY_TRUNCATION_DIR";
/// Verification-loop toggle (`=off` disables workspace test grading).
pub const ENV_VERIFY_LOOP: &str = "KITTY_VERIFY_LOOP";
/// Bench overwrite allowance (`=1` permits Write on existing files).
pub const ENV_BENCH_MODE: &str = "KITTY_BENCH_MODE";
/// Post-Write syntax check gate (`=1` enables).
pub const ENV_EDIT_CHECK: &str = "KITTY_EDIT_CHECK";
/// Post-Write lint gate (`=1` enables).
pub const ENV_LINT_GATE: &str = "KITTY_LINT_GATE";
/// Interactive autorun continuation hops (bench paths are flag-only).
pub const ENV_AUTORUN_HOPS: &str = "KITTY_AUTORUN_HOPS";
/// Per-exercise bench wall-clock fallback seconds.
pub const ENV_BENCH_TIMEOUT_SECS: &str = "KITTY_BENCH_TIMEOUT_SECS";
/// GPQA question-cache directory override (CLI evaluation concern).
pub const ENV_GPQA_CACHE: &str = "KITTY_GPQA_CACHE";
/// GPQA sandboxed-child workspace root (CLI evaluation concern).
pub const ENV_WORKSPACE_ROOT: &str = "KITTY_WORKSPACE_ROOT";

pub use kitty_runtime::client::{ENV_NO_THINK, ENV_NUM_CTX};
/// Per-model turn-cap override spec (`exact=value[,family=value…]`).
pub const ENV_MODEL_TURN_CAPS: &str = "KITTY_MODEL_TURN_CAPS";

// ── Sandbox opt-out ─────────────────────────────────────────────────

/// Pure core: `KITTY_SANDBOX=off` (exact match) disables containment.
pub fn sandbox_off_raw(raw: Option<&str>) -> bool {
    raw == Some("off")
}

/// `true` when `KITTY_SANDBOX=off` is set (explicit containment opt-out).
pub fn sandbox_off() -> bool {
    sandbox_off_raw(std::env::var(ENV_SANDBOX).ok().as_deref())
}

// ── Telemetry ───────────────────────────────────────────────────────

/// Pure core: telemetry is on unless `KITTY_TELEMETRY=off`.
pub fn telemetry_enabled_raw(raw: Option<&str>) -> bool {
    raw != Some("off")
}

/// Telemetry enabled for the current process.
pub fn telemetry_enabled() -> bool {
    telemetry_enabled_raw(std::env::var(ENV_TELEMETRY).ok().as_deref())
}

// ── Verification loop ───────────────────────────────────────────────

/// Pure core: verification is on unless `KITTY_VERIFY_LOOP=off`.
pub fn verify_loop_enabled_raw(raw: Option<&str>) -> bool {
    raw != Some("off")
}

/// Verification loop enabled for the current process.
pub fn verify_loop_enabled() -> bool {
    verify_loop_enabled_raw(std::env::var(ENV_VERIFY_LOOP).ok().as_deref())
}

// ── Bench / gates ───────────────────────────────────────────────────

/// Pure core: `KITTY_BENCH_MODE=1` permits Write-on-existing (benchmarks).
pub fn bench_mode_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Bench overwrite mode enabled.
pub fn bench_mode() -> bool {
    bench_mode_raw(std::env::var(ENV_BENCH_MODE).ok().as_deref())
}

/// Pure core: `KITTY_EDIT_CHECK=1` enables the post-Write syntax check.
pub fn edit_check_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Post-Write syntax check enabled.
pub fn edit_check() -> bool {
    edit_check_raw(std::env::var(ENV_EDIT_CHECK).ok().as_deref())
}

/// Pure core: `KITTY_LINT_GATE=1` enables the post-Write lint gate.
pub fn lint_gate_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Post-Write lint gate enabled.
pub fn lint_gate() -> bool {
    lint_gate_raw(std::env::var(ENV_LINT_GATE).ok().as_deref())
}

// ── Truncation spill ────────────────────────────────────────────────

/// Directory for oversized tool-output spills (`KITTY_TRUNCATION_DIR` or
/// `.kitty/truncation`).
pub fn truncation_dir() -> std::path::PathBuf {
    std::env::var(ENV_TRUNCATION_DIR)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(".kitty/truncation"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_off_requires_exact_match() {
        assert!(sandbox_off_raw(Some("off")));
        assert!(!sandbox_off_raw(Some("OFF")));
        assert!(!sandbox_off_raw(Some("on")));
        assert!(!sandbox_off_raw(None));
    }

    #[test]
    fn toggles_default_on_and_off_with_exact_value() {
        assert!(telemetry_enabled_raw(None));
        assert!(!telemetry_enabled_raw(Some("off")));
        assert!(verify_loop_enabled_raw(None));
        assert!(!verify_loop_enabled_raw(Some("off")));
        assert!(!bench_mode_raw(None));
        assert!(bench_mode_raw(Some("1")));
        assert!(!edit_check_raw(None));
        assert!(edit_check_raw(Some("1")));
        assert!(!lint_gate_raw(None));
        assert!(lint_gate_raw(Some("1")));
    }
}
