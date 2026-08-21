//! Contract — Definition of Done guard (DoD done-guard).
//!
//! Minimal port of `smallcode/src/session/contract.js:39-266` +
//! `contract_guard.js:26-98` for the harness. Keeps contracts in-memory only;
//! persistence to `.smallcode/contracts/<id>/contract.md` is optional via
//! `to_markdown()` / `save_markdown()`.

use regex::Regex;
use std::sync::OnceLock;

// ── Core types ──────────────────────────────────────────────────────────────

/// Contract state (lifecycle).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractState {
    Pending,
    Active,
    Completed,
    Failed,
}

/// Per-assertion verification kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssertionCheck {
    FileExists(String),
    FileContains(String, String),
    CommandSucceeds(String),
}

/// Per-assertion execution status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssertionStatus {
    Pending,
    Passed,
    Skipped,
    Failed(String),
}

/// Single assertion within a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assertion {
    pub id: String,
    pub description: String,
    pub check: AssertionCheck,
    pub status: AssertionStatus,
}

/// A contract (Definition of Done): a set of verifiable assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    pub id: String,
    pub assertions: Vec<Assertion>,
    pub state: ContractState,
}

impl Contract {
    /// Create a new contract with the given id, in Pending state and no assertions.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            assertions: Vec::new(),
            state: ContractState::Pending,
        }
    }

    /// Add an assertion derived from `check`. Generates an id `a01`, `a02`, …
    /// and a human description from the check. Returns the new assertion id.
    pub fn add_assertion(&mut self, check: AssertionCheck) -> String {
        let idx = self.assertions.len() + 1;
        let id = format!("a{idx:02}");
        let description = match &check {
            AssertionCheck::FileExists(p) => format!("File exists: {p}"),
            AssertionCheck::FileContains(p, needle) => {
                format!("File '{p}' contains '{needle}'")
            }
            AssertionCheck::CommandSucceeds(cmd) => format!("Command succeeds: {cmd}"),
        };
        let assertion = Assertion {
            id: id.clone(),
            description,
            check,
            status: AssertionStatus::Pending,
        };
        self.assertions.push(assertion);
        id
    }

    /// Mark the assertion with `id` as passed. Returns true if found.
    pub fn mark_passed(&mut self, id: &str) -> bool {
        if let Some(a) = self.assertions.iter_mut().find(|a| a.id == id) {
            a.status = AssertionStatus::Passed;
            true
        } else {
            false
        }
    }

    /// Mark the assertion with `id` as failed with `reason`. Returns true if found.
    pub fn mark_failed(&mut self, id: &str, reason: impl Into<String>) -> bool {
        if let Some(a) = self.assertions.iter_mut().find(|a| a.id == id) {
            a.status = AssertionStatus::Failed(reason.into());
            true
        } else {
            false
        }
    }

    /// Mark an assertion as skipped (genuinely out of scope).
    pub fn mark_skipped(&mut self, id: &str) -> bool {
        if let Some(a) = self.assertions.iter_mut().find(|a| a.id == id) {
            a.status = AssertionStatus::Skipped;
            true
        } else {
            false
        }
    }

    /// True when every assertion is `Passed` or `Skipped`. Empty contract is not done
    /// (mirrors `smallcode` `isDone()`).
    pub fn is_done(&self) -> bool {
        if self.assertions.is_empty() {
            return false;
        }
        self.assertions
            .iter()
            .all(|a| matches!(a.status, AssertionStatus::Passed | AssertionStatus::Skipped))
    }

    /// Blockers are assertions still `Pending` or `Failed` — they prevent completion.
    pub fn blockers(&self) -> Vec<&Assertion> {
        self.assertions
            .iter()
            .filter(|a| {
                matches!(
                    a.status,
                    AssertionStatus::Pending | AssertionStatus::Failed(_)
                )
            })
            .collect()
    }

    /// Render a Markdown snapshot suitable for `.smallcode/contracts/<id>/contract.md`.
    /// Pure in-memory helper; the caller decides whether to persist it.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Contract {}\n\n", self.id));
        out.push_str(&format!("- **state**: {:?}\n", self.state));
        out.push_str(&format!("- **assertions**: {}\n", self.assertions.len()));
        out.push('\n');
        if self.assertions.is_empty() {
            out.push_str("_No assertions yet._\n");
            return out;
        }
        out.push_str("## Assertions\n\n");
        for a in &self.assertions {
            let mark = match &a.status {
                AssertionStatus::Passed => "✅",
                AssertionStatus::Failed(_) => "❌",
                AssertionStatus::Skipped => "⊘",
                AssertionStatus::Pending => "⏳",
            };
            let status_str = match &a.status {
                AssertionStatus::Pending => "Pending".to_string(),
                AssertionStatus::Passed => "Passed".to_string(),
                AssertionStatus::Skipped => "Skipped".to_string(),
                AssertionStatus::Failed(reason) => format!("Failed: {reason}"),
            };
            out.push_str(&format!(
                "- {mark} **{}** — {} _({})_\n",
                a.id, a.description, status_str
            ));
            match &a.check {
                AssertionCheck::FileExists(p) => {
                    out.push_str(&format!("  - check: FileExists({p})\n"));
                }
                AssertionCheck::FileContains(p, needle) => {
                    out.push_str(&format!("  - check: FileContains({p}, {needle})\n"));
                }
                AssertionCheck::CommandSucceeds(cmd) => {
                    out.push_str(&format!("  - check: CommandSucceeds({cmd})\n"));
                }
            }
        }
        out
    }

    /// Convenience: write markdown to `.smallcode/contracts/<id>/contract.md` under `base_dir`.
    /// Best-effort, returns the path written.
    pub fn save_markdown(&self, base_dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        let dir = base_dir.join(".smallcode").join("contracts").join(&self.id);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("contract.md");
        std::fs::write(&path, self.to_markdown())?;
        Ok(path)
    }
}

// ── Done-guard ─────────────────────────────────────────────────────────────
//
// Regex set ported from `smallcode/src/session/contract_guard.js:26-34`.
// All patterns are case-insensitive (`(?i)`). The guard only cares whether
// *any* pattern matches and the text is not a question.

/// Raw pattern strings (case-insensitive via `(?i)`). Expanded slightly to
/// cover `ready for review` and standalone `completed`/`finished` while
/// preserving the original 7-pattern intent.
pub const DONE_PATTERNS_RAW: &[&str] = &[
    r"(?i)\b(all\s+)?(done|completed|finished)\b",
    r"(?i)\btask (is\s+)?(now\s+)?(complete|completed|finished|done)\b",
    r"(?i)\b(everything|all)\s+(is\s+)?(done|working|set|complete|completed|finished)\b",
    r"(?i)\b(finished|completed)\s+(the\s+)?task\b",
    r"(?i)\b(implementation|feature|fix)\s+(is\s+)?(complete|done|finished|completed)\b",
    r"(?i)\bsuccessfully\s+(implemented|completed|finished)\b",
    r"(?i)\bready\s+(to\s+(ship|merge|use|review)|for\s+review)\b",
];

fn done_patterns() -> &'static Vec<Regex> {
    static CELL: OnceLock<Vec<Regex>> = OnceLock::new();
    CELL.get_or_init(|| {
        DONE_PATTERNS_RAW
            .iter()
            .map(|p| Regex::new(p).expect("invalid DONE pattern"))
            .collect()
    })
}

/// Returns true when `text` looks like a completion claim.
/// - returns false for empty / non-string-like / question endings (`?`).
/// - case-insensitive, matches any of the 7 DONE_PATTERNS.
pub fn looks_like_done_claim(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    let trimmed = text.trim();
    if trimmed.ends_with('?') {
        return false;
    }
    for re in done_patterns() {
        if re.is_match(text) {
            return true;
        }
    }
    false
}

/// Whether the contract guard is enabled. Respects `SMALLCODE_CONTRACT=false`.
pub fn contract_guard_enabled() -> bool {
    !matches!(std::env::var("SMALLCODE_CONTRACT"), Ok(v) if v == "false")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn contract_with_n_assertions(n: usize) -> Contract {
        let mut c = Contract::new("test");
        for i in 0..n {
            let id = c.add_assertion(AssertionCheck::FileExists(format!("file{i}.rs")));
            // leave Pending by default
            let _ = id;
        }
        c
    }

    #[test]
    fn is_done_all_passed_is_done() {
        let mut c = contract_with_n_assertions(2);
        assert!(!c.is_done(), "pending contract should not be done");
        let ids: Vec<String> = c.assertions.iter().map(|a| a.id.clone()).collect();
        for id in &ids {
            c.mark_passed(id);
        }
        assert!(c.is_done(), "all passed should be done");
    }

    #[test]
    fn is_done_with_skipped_counts_as_done() {
        let mut c = Contract::new("skipped");
        let a1 = c.add_assertion(AssertionCheck::FileExists("a.rs".into()));
        let a2 = c.add_assertion(AssertionCheck::FileExists("b.rs".into()));
        c.mark_passed(&a1);
        c.mark_skipped(&a2);
        assert!(c.is_done(), "Passed+Skipped should be done");
    }

    #[test]
    fn is_done_pending_not_done() {
        let mut c = contract_with_n_assertions(2);
        let ids: Vec<String> = c.assertions.iter().map(|a| a.id.clone()).collect();
        c.mark_passed(&ids[0]);
        // one pending left
        assert!(!c.is_done(), "pending remaining should not be done");
        assert_eq!(c.blockers().len(), 1);
        assert_eq!(c.blockers()[0].id, ids[1]);
    }

    #[test]
    fn is_done_failed_not_done() {
        let mut c = contract_with_n_assertions(1);
        let id = c.assertions[0].id.clone();
        c.mark_failed(&id, "missing file");
        assert!(!c.is_done());
        assert_eq!(c.blockers().len(), 1);
        assert!(matches!(c.blockers()[0].status, AssertionStatus::Failed(_)));
    }

    #[test]
    fn is_done_empty_not_done() {
        let c = Contract::new("empty");
        assert!(!c.is_done(), "empty contract is not done (mirrors JS)");
        assert!(c.blockers().is_empty());
    }

    #[test]
    fn blockers_lists_pending_and_failed() {
        let mut c = Contract::new("blockers");
        let a1 = c.add_assertion(AssertionCheck::FileExists("a".into()));
        let a2 = c.add_assertion(AssertionCheck::FileExists("b".into()));
        let a3 = c.add_assertion(AssertionCheck::FileExists("c".into()));
        c.mark_passed(&a1);
        c.mark_failed(&a2, "nope");
        // a3 stays Pending
        let blockers = c.blockers();
        assert_eq!(blockers.len(), 2);
        let ids: Vec<&str> = blockers.iter().map(|b| b.id.as_str()).collect();
        assert!(ids.contains(&a2.as_str()));
        assert!(ids.contains(&a3.as_str()));
        assert!(!ids.contains(&a1.as_str()));
    }

    #[test]
    fn looks_like_done_claim_positive_cases() {
        // standalone done / completed / finished
        assert!(looks_like_done_claim("done"), "standalone done");
        assert!(looks_like_done_claim("All done"), "All done");
        assert!(looks_like_done_claim("all completed"), "all completed");
        assert!(looks_like_done_claim("finished"), "finished");
        // task patterns
        assert!(looks_like_done_claim("task completed"), "task completed");
        assert!(looks_like_done_claim("task is now complete"));
        assert!(looks_like_done_claim("finished the task"));
        assert!(looks_like_done_claim("completed the task"));
        // implementation/fix patterns
        assert!(looks_like_done_claim("implementation is complete"));
        assert!(looks_like_done_claim("feature is done"), "feature is done");
        // successfully
        assert!(looks_like_done_claim("successfully implemented"));
        assert!(looks_like_done_claim("Successfully completed"));
        // ready variants
        assert!(looks_like_done_claim("ready to ship"));
        assert!(
            looks_like_done_claim("ready for review"),
            "ready for review"
        );
        assert!(looks_like_done_claim("Ready to merge"));
        // everything / all
        assert!(looks_like_done_claim("everything is done"));
        assert!(looks_like_done_claim("all is working"));
        // case insensitive
        assert!(looks_like_done_claim("DONE"), "uppercase DONE");
        assert!(looks_like_done_claim("Task Is Completed"), "mixed case");
    }

    #[test]
    fn looks_like_done_claim_negative_cases() {
        assert!(!looks_like_done_claim("hello world"), "unrelated");
        assert!(!looks_like_done_claim(""), "empty");
        assert!(!looks_like_done_claim("   "), "whitespace");
        // question should be skipped
        assert!(!looks_like_done_claim("done?"), "question");
        assert!(
            !looks_like_done_claim("Is this done?"),
            "question with done"
        );
        assert!(
            !looks_like_done_claim("Are we done?  "),
            "question with spaces"
        );
        // similar but not matching ready for review question
        assert!(!looks_like_done_claim("ready for review?"));
        // random words
        assert!(!looks_like_done_claim("I did some work"));
    }

    #[test]
    fn looks_like_done_claim_question_trim() {
        // ensure trailing ? after spaces counts as question
        assert!(!looks_like_done_claim("All done ?"));
        // but declarative still true
        assert!(looks_like_done_claim("All done."));
        assert!(looks_like_done_claim("All done! All working"));
    }

    #[test]
    fn to_markdown_contains_id_and_assertions() {
        let mut c = Contract::new("md-test");
        c.add_assertion(AssertionCheck::FileContains(
            "src/main.rs".into(),
            "fn main".into(),
        ));
        c.mark_passed("a01");
        let md = c.to_markdown();
        assert!(
            md.contains("md-test"),
            "markdown should contain contract id"
        );
        assert!(md.contains("a01"), "markdown should contain assertion id");
        assert!(md.contains("Passed"), "markdown should contain status");
    }

    #[test]
    fn mark_passed_and_failed_return_values() {
        let mut c = Contract::new("ret");
        let a1 = c.add_assertion(AssertionCheck::CommandSucceeds("cargo test".into()));
        assert!(c.mark_passed(&a1));
        assert!(!c.mark_passed("nonexistent"));
        let mut c2 = Contract::new("ret2");
        let a2 = c2.add_assertion(AssertionCheck::FileExists("x".into()));
        assert!(c2.mark_failed(&a2, "reason"));
        assert_eq!(
            c2.assertions[0].status,
            AssertionStatus::Failed("reason".into())
        );
        assert!(!c2.mark_failed("ghost", "x"));
    }

    #[test]
    fn add_assertion_generates_description_and_ids() {
        let mut c = Contract::new("desc");
        let id1 = c.add_assertion(AssertionCheck::FileExists("foo.txt".into()));
        assert_eq!(id1, "a01");
        assert_eq!(c.assertions[0].description, "File exists: foo.txt");
        let id2 = c.add_assertion(AssertionCheck::FileContains(
            "bar.rs".into(),
            "hello".into(),
        ));
        assert_eq!(id2, "a02");
        assert!(c.assertions[1].description.contains("bar.rs"));
        let id3 = c.add_assertion(AssertionCheck::CommandSucceeds("make".into()));
        assert_eq!(id3, "a03");
        assert!(c.assertions[2].description.contains("make"));
    }
}
