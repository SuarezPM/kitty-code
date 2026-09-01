//! Secret redaction for tool outputs, errors, and telemetry.
//!
//! Redacts known secret patterns before they leave the harness boundary.
//! Never redacts user content — only matches specific secret formats.

use regex::Regex;
use std::sync::OnceLock;

/// Patterns that indicate secrets/credentials.
fn redaction_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // API keys: sk-*, ghp_*, gho_*, glpat-*, xox[bpas]-*
            Regex::new(r"(?i)(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|glpat-[A-Za-z0-9\-]{20,}|xox[bpas]-[A-Za-z0-9\-]{10,})").unwrap(),
            // Bearer tokens
            Regex::new(r"(?i)(Bearer\s+[A-Za-z0-9\-._~+/]+=*)").unwrap(),
            // Generic KEY=SECRET in env vars (only in command output context)
            Regex::new(r"(?i)((?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIALS?)\s*[=:]\s*\S+)").unwrap(),
            // HuggingFace tokens
            Regex::new(r"hf_[A-Za-z0-9]{34}").unwrap(),
            // AWS keys
            Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
        ]
    })
}

/// Redact all known secret patterns in `input`, replacing matches with `[REDACTED]`.
pub fn redact_secrets(input: &str) -> String {
    let mut result = input.to_string();
    for pattern in redaction_patterns() {
        result = pattern.replace_all(&result, "[REDACTED]").to_string();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_api_keys() {
        let input = "Using key sk-abc123def456ghi789jkl012mno";
        assert_eq!(redact_secrets(input), "Using key [REDACTED]");
    }

    #[test]
    fn redacts_github_tokens() {
        // ghp_ + 36 alphanumeric chars (GitHub PAT format)
        let input = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        assert_eq!(redact_secrets(input), "[REDACTED]");
    }

    #[test]
    fn redacts_bearer() {
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9";
        assert_eq!(redact_secrets(input), "Authorization: [REDACTED]");
    }

    #[test]
    fn redacts_hf_tokens() {
        let input = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh";
        assert_eq!(redact_secrets(input), "[REDACTED]");
    }

    #[test]
    fn preserves_normal_text() {
        let input = "Hello world, this is a normal message";
        assert_eq!(redact_secrets(input), input);
    }

    #[test]
    fn redacts_env_var_secrets() {
        let input = "API_KEY=supersecret123";
        assert_eq!(redact_secrets(input), "[REDACTED]");
    }
}
